from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np

is_cuda = torch.cuda.is_available()

class CRF_Loss(nn.Module):
    def __init__(self, tagset_size, pad_token_id, tag_pad_id):
        super(CRF_Loss, self).__init__()
        self.start_tag = tagset_size
        self.end_tag = tagset_size + 1
        self.num_tags = tagset_size + 2
        self.tag_pad_id = tag_pad_id
        self.pad_token_id = pad_token_id

        self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))
        nn.init.constant_(self.transitions, -np.log(self.num_tags))

        self.transitions.data[self.end_tag, :] = -10000
        self.transitions.data[:, self.start_tag] = -10000

    def get_log_p_z(self, emissions, mask):
        seq_len = emissions.shape[1]
        log_alpha = emissions[:, 0].clone()
        log_alpha += self.transitions[self.start_tag, : self.start_tag].unsqueeze(0)

        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, idx].unsqueeze(1)
            broadcast_transitions = self.transitions[ : self.start_tag, : self.start_tag].unsqueeze(0)
            broadcast_logprob = log_alpha.unsqueeze(2)
            score = broadcast_logprob + broadcast_emissions + broadcast_transitions

            score = torch.logsumexp(score, 1)
            log_alpha = score * mask[:, idx].unsqueeze(1) + log_alpha.squeeze(1) * (1.0 - mask[:, idx].unsqueeze(1))

        log_alpha += self.transitions[: self.start_tag, self.end_tag].unsqueeze(0)
        return torch.logsumexp(log_alpha.squeeze(1), 1)

    def get_log_p_Y_X(self, emissions, mask, orig_tags):
        seq_len = emissions.shape[1]
        tags = orig_tags.clone()
        tags[tags < 0] = 0

        llh = self.transitions[self.start_tag, tags[:, 0]].unsqueeze(1)
        llh += emissions[:, 0, :].gather(1, tags[:, 0].view(-1, 1)) * mask[:, 0].unsqueeze(1)

        for idx in range(1, seq_len):
            old_state, new_state = (
                tags[:, idx - 1].view(-1, 1),
                tags[:, idx].view(-1, 1),
            )
            emission_scores = emissions[:, idx, :].gather(1, new_state)
            transition_scores = self.transitions[old_state, new_state]
            llh += (emission_scores + transition_scores) * mask[:, idx].unsqueeze(1)

        last_tag_indices = mask.sum(1, dtype=torch.long) - 1
        last_tags = tags.gather(1, last_tag_indices.view(-1, 1))

        llh += self.transitions[last_tags.squeeze(1), self.end_tag].unsqueeze(1)

        return llh.squeeze(1)

    def log_likelihood(self, emissions, tags, mask):
        log_z = self.get_log_p_z(emissions, mask)
        log_p_y_x = self.get_log_p_Y_X(emissions, mask, tags)
        return log_p_y_x - log_z

    def get_crf_loss(self, logits, y):
        mask = y.ne(self.tag_pad_id)
        s_lens = mask.sum(1)
        loss = -1 * self.log_likelihood(logits, y, mask.float())
        loss = loss / s_lens.float()
        loss = loss.mean()
        return loss

    def viterbi_decode(self, emissions, mask):
        mask = mask.float()
        b, seq_len, d = emissions.shape
        log_prob = emissions[:, 0].clone()
        log_prob += self.transitions[self.start_tag, : self.start_tag].unsqueeze(0)

        end_scores = log_prob + self.transitions[: self.start_tag, self.end_tag].unsqueeze(0)

        best_scores_list = []
        best_scores_list.append(end_scores.unsqueeze(1))

        best_paths_0 = torch.Tensor().long()
        if is_cuda:
            best_paths_0 = best_paths_0.cuda()
        best_paths_list = [best_paths_0]

        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, idx].unsqueeze(1)
            broadcast_transmissions = self.transitions[: self.start_tag, : self.start_tag].unsqueeze(0)
            broadcast_log_prob = log_prob.unsqueeze(2)
            score = broadcast_emissions + broadcast_transmissions + broadcast_log_prob
            max_scores, max_score_indices = torch.max(score, 1)
            best_paths_list.append(max_score_indices.unsqueeze(1))
            end_scores = max_scores + self.transitions[: self.start_tag, self.end_tag].unsqueeze(0)

            best_scores_list.append(end_scores.unsqueeze(1))
            log_prob = max_scores

        best_scores = torch.cat(best_scores_list, 1).float()
        best_paths = torch.cat(best_paths_list, 1)

        max_scores, max_indices_from_scores = torch.max(best_scores, 2)

        valid_index_tensor = torch.tensor(0).long()
        padding_tensor = torch.tensor(self.tag_pad_id).long()
        
        if is_cuda:
            valid_index_tensor = valid_index_tensor.cuda()
            padding_tensor = padding_tensor.cuda()
        #alternative to where
        #curr_mask = mask[:, seq_len - 1].float()
        #labels = max_indices_from_scores[:, seq_len - 1] * curr_mask + torch.logical_not(curr_mask) * padding_tensor

        labels = max_indices_from_scores[:, seq_len - 1]
        labels = torch.where(mask[:, seq_len - 1] != 1.0, padding_tensor, labels)
        all_labels = labels.unsqueeze(1).long()
        #####
        labels_score = max_scores[:, seq_len - 1]
        all_labels_score = labels_score.unsqueeze(1)
        ####
        for idx in range(seq_len - 2, -1, -1):
            indices_for_lookup = all_labels[:, -1].clone()
            indices_for_lookup = torch.where(indices_for_lookup == self.tag_pad_id,
                                             valid_index_tensor,
                                             indices_for_lookup)

            indices_from_prev_pos = best_paths[:, idx, :].gather(1, indices_for_lookup.view(-1, 1).long()).squeeze(1)
            indices_from_prev_pos = torch.where(mask[:, idx + 1] != 1.0, padding_tensor, indices_from_prev_pos)

            indices_from_max_scores = max_indices_from_scores[:, idx]
            indices_from_max_scores = torch.where(mask[:, idx + 1] == 1.0, padding_tensor, indices_from_max_scores)

            labels = torch.where(indices_from_max_scores == self.tag_pad_id,
                                 indices_from_prev_pos,
                                 indices_from_max_scores)
            # Set to ignore_index if present state is not valid.
            labels = torch.where(mask[:, idx] != 1.0, padding_tensor, labels)
            all_labels = torch.cat((all_labels, labels.view(-1, 1).long()), 1)
            ######
            labels_score = max_scores[:, idx]
            all_labels_score = torch.cat((all_labels_score, labels_score.view(-1, 1)), 1)
            ####
        #think about squeezing this score between 0 and 1
        last_tag_indices = mask.sum(1, dtype=torch.long) - 1
        sentence_score = max_scores.gather(1, last_tag_indices.view(-1, 1)).squeeze(1)
        all_labels = torch.flip(all_labels, [1])
        all_labels_score = torch.flip(all_labels_score, [1])

        return sentence_score, all_labels, all_labels_score

    def structural_perceptron_loss(self, emissions, tags):
        mask = tags.ne(self.tag_pad_id).float()

        best_scores, pred = self.viterbi_decode(emissions, mask, is_cuda)
        log_p_y_x = self.get_log_p_Y_X(emissions, mask, tags)

        delta = torch.sum(tags.ne(pred).float()*mask, 1)

        margin_loss = torch.clamp(best_scores + delta - log_p_y_x, min=0.0)
        return margin_loss

    def bert_output2crf_input(self, logits_ner, labels):
        mask = labels.ne(self.tag_pad_id)
        lens = mask.sum(1).view(-1).tolist()

        logits_selected = torch.masked_select(logits_ner, mask.unsqueeze(2)).view(-1, logits_ner.size()[-1])
        logits_split = torch.split(logits_selected, lens)
        logits_padded = pad_sequence(logits_split, batch_first=True, padding_value=self.pad_token_id)

        labels_selected = torch.masked_select(labels, mask)
        labels_split = torch.split(labels_selected, lens)
        labels_padded = pad_sequence(labels_split, batch_first=True, padding_value=self.tag_pad_id)

        return logits_padded, labels_padded
