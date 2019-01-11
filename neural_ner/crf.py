from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from data_utils.sentence_utils import Constants


def get_mask(lengths):
    seq_lens = lengths.view(-1, 1)
    max_len = torch.max(seq_lens)
    range_tensor = torch.arange(max_len).unsqueeze(0)
    range_tensor = range_tensor.expand(seq_lens.size(0), range_tensor.size(1))
    mask = (range_tensor < seq_lens).float()
    return mask

class CRF_Loss(nn.Module):
    def __init__(self, tagset_size):
        super(CRF_Loss, self).__init__()
        self.start_tag_idx = tagset_size
        self.stop_tag_idx = tagset_size + 1
        self.num_tags = tagset_size + 2

        #transition from y_i-1 to y_i, T[y_i, y_j] = y_i <= y_j
        #+2 added for start and end indices
        self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        #no transition to start_tag, not transition from end tag
        self.transitions.data[self.start_tag_idx, :] = -10000
        self.transitions.data[:, self.stop_tag_idx] = -10000

    def get_log_p_z(self, emissions, mask, seq_len):
        log_alpha = emissions[:, 0].clone()
        log_alpha += self.transitions[self.start_tag_idx, : self.start_tag_idx].unsqueeze(0)

        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, idx].unsqueeze(1)
            broadcast_transitions = self.transitions[
                : self.start_tag, : self.start_tag
            ].unsqueeze(0)
            broadcast_logprob = log_alpha.unsqueeze(2)
            score = broadcast_logprob + broadcast_emissions + broadcast_transitions

            score = torch.logsumexp(score, 1)
            log_alpha = score * mask[:, idx].unsqueeze(1) + log_alpha.squeeze(1) * (
                1.0 - mask[:, idx].unsqueeze(1)
            )

            log_alpha += self.transitions[: self.start_tag, self.end_tag].unsqueeze(0)
        return torch.logsumexp(log_alpha.squeeze(1), 1)

    def get_log_p_Y_X(self, emissions, mask, seq_len, tags):
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

    def log_likelihood(self, emissions, tags):
        mask = tags.ne(Constants.TAG_PAD_ID).float()
        seq_len = emissions.shape[1]
        log_z = self.get_log_p_z(emissions, mask, seq_len)
        log_p_y_x = self.get_log_p_Y_X(emissions, mask, seq_len, tags)
        return log_p_y_x - log_z

    def forward(self, emissions, tags):
        return self.log_likelihood(emissions, tags)

    def inference(self, emissions, lengths):
        return self.viterbi_decode(emissions, lengths)

    def viterbi_decode(self, emissions, lengths):
        mask = get_mask(lengths)
        seq_len = emissions.shape[1]

        log_prob = emissions[:, 0].clone()
        log_prob += self.transitions[self.start_tag, : self.start_tag].unsqueeze(0)


        end_scores = log_prob + self.transitions[
            : self.start_tag, self.end_tag
        ].unsqueeze(0)

        best_scores_list = []
        best_scores_list.append(end_scores.unsqueeze(1))

        best_paths_list = [torch.Tensor().long()]

        for idx in range(1, seq_len):
            broadcast_emissions = emissions[:, idx].unsqueeze(1)
            broadcast_transmissions = self.transitions[
                                      : self.start_tag, : self.start_tag
                                      ].unsqueeze(0)
            broadcast_log_prob = log_prob.unsqueeze(2)

            score = broadcast_emissions + broadcast_transmissions + broadcast_log_prob

            max_scores, max_score_indices = torch.max(score, 1)

            best_paths_list.append(max_score_indices.unsqueeze(1))

            end_scores = max_scores + self.transitions[
                                      : self.start_tag, self.end_tag
                                      ].unsqueeze(0)

            best_scores_list.append(end_scores.unsqueeze(1))
            log_prob = max_scores

        best_scores = torch.cat(best_scores_list, 1).float()
        best_paths = torch.cat(best_paths_list, 1)

        _, max_indices_from_scores = torch.max(best_scores, 2)

        valid_index_tensor = torch.tensor(0).long()
        padding_tensor = torch.tensor(Constants.PAD_ID).long()

        labels = max_indices_from_scores[:, seq_len - 1]
        labels = self._mask_tensor(labels, 1.0 - mask[:, seq_len - 1], padding_tensor)

        all_labels = labels.unsqueeze(1).long()

        for idx in range(seq_len - 2, -1, -1):
            indices_for_lookup = all_labels[:, -1].clone()
            indices_for_lookup = torch.where(
                indices_for_lookup == self.ignore_index,
                valid_index_tensor,
                indices_for_lookup
            )

            indices_from_prev_pos = (
                best_paths[:, idx, :]
                    .gather(1, indices_for_lookup.view(-1, 1).long())
                    .squeeze(1)
            )
            indices_from_prev_pos = torch.where(
                (1.0 - mask[:, idx + 1]),
                padding_tensor,
                indices_from_prev_pos
            )

            indices_from_max_scores = max_indices_from_scores[:, idx]
            indices_from_max_scores = torch.where(
                mask[:, idx + 1],
                padding_tensor,
                indices_from_max_scores
            )

            labels = torch.where(
                indices_from_max_scores == self.ignore_index,
                indices_from_prev_pos,
                indices_from_max_scores,
            )

            # Set to ignore_index if present state is not valid.
            labels = torch.where(
                (1 - mask[:, idx]),
                padding_tensor,
                labels
            )
            all_labels = torch.cat((all_labels, labels.view(-1, 1).long()), 1)

        return best_scores, torch.flip(all_labels, [1])

