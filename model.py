from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import logging

logging.basicConfig(level=logging.INFO)

def test_one_batch(s, s_lengths, model):
    model.eval()
    logits = model(s, s_lengths)
    _, pred = model.get_argmax(logits)
    return logits, pred

def get_model(dataset, config, use_cuda, model_file_path, is_eval=False):
    embd_vector = dataset.init_emb()
    tagset_size = len(dataset.labels)
    model = NER_SOFTMAX(embd_vector, config.hidden_dim, tagset_size,
                        dataset.get_pad_labelid(), config.reg_lambda)

    if is_eval:
        model = model.eval()
    if use_cuda:
        model = model.cuda()

    if model_file_path is not None:
        state = torch.load(model_file_path, map_location=lambda storage, location: storage)
        model.load_state_dict(state['model'], strict=False)

    return model

class NER_SOFTMAX(nn.Module):
    def __init__(self, embd_vector, hidden_dim, tagset_size, pad_labelid, reg_lambda):
        super(NER_SOFTMAX, self).__init__()
        self.pad_labelid = pad_labelid
        self.reg_lambda = reg_lambda

        self.word_embeds = nn.Embedding.from_pretrained(embd_vector)
        embedding_dim = self.word_embeds.embedding_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence, lengths):
        embedded = self.word_embeds(sentence)
        lengths = lengths.view(-1).tolist()
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        output, hidden = self.lstm(packed)

        lstm_feats, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        lstm_feats = lstm_feats.contiguous()

        b, t_k, d = list(lstm_feats.size())

        logits = self.hidden2tag(lstm_feats.view(-1, d))
        logits = logits.view(b, t_k, -1)

        return logits

    def neg_log_likelihood(self, logits, y, s_len):
        log_smx = F.log_softmax(logits, dim=2)
        loss = F.nll_loss(log_smx.transpose(1, 2), y, ignore_index=self.pad_labelid, reduction='none')
        loss = loss.sum(dim=1) / s_len.float()
        loss = loss.mean()

        # might be reduction='sum' and divide s_lengths
        l2_reg = sum(p.norm(2) for p in self.parameters() if p.requires_grad)

        loss += self.reg_lambda * l2_reg
        return loss

    def get_argmax(self, logits):
        max_value, max_idx = torch.max(logits, dim=2)
        return max_value, max_idx

class NER_CRF(nn.Module):
    def __init__(self, embd_vector, hidden_dim, tagset_size, pad_labelid,
                 reg_lambda):
        super(NER_CRF, self).__init__()
        self.pad_labelid = pad_labelid
        self.reg_lambda = reg_lambda
        self.start_tag_idx = tagset_size
        self.stop_tag_idx = tagset_size + 1
        self.all_tagset_size = tagset_size + 2

        self.word_embeds = nn.Embedding.from_pretrained(embd_vector)
        embedding_dim = self.word_embeds.embedding_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        #transition from y_i-1 to y_i, T[y_i, y_j] = y_i <= y_j
        #+2 added for start and end indices
        self.transitions = nn.Parameter(torch.randn(self.all_tagset_size, self.all_tagset_size))

        #no transition to start_tag, not transition from end tag
        self.transitions.data[self.start_tag_idx, :] = -10000
        self.transitions.data[:, self.stop_tag_idx] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def get_emission_prob(self, sentence, lengths):
        embedded = self.word_embeds(sentence)
        lengths = lengths.view(-1).tolist()
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)

        self.hidden = self.init_hidden()
        output, self.hidden = self.lstm(packed, self.hidden)

        lstm_feats, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        lstm_feats = lstm_feats.contiguous()

        b, t_k, d = list(lstm_feats.size())

        logits = self.hidden2tag(lstm_feats.view(-1, d))
        logits = logits.view(b, t_k, -1)

        return logits

    def get_argmax(self, logits):
        max_value, max_idx = torch.max(logits, dim=2)
        return max_value, max_idx

    def log_sum_exp(self, vec):
        max_score, _ = self.get_argmax(vec)
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def forward_algo(self, emission_prob):
        init_alphas = torch.full((1, self.all_tagset_size), -10000.)
        init_alphas[0][self.start_tag_idx] = 0.
        forward_var = init_alphas
        for e_i in emission_prob:
            alphas_t = []
            for next_tag in range(self.all_tagset_size):
                emit_score = e_i[next_tag].view(
                    1, -1).expand(1, self.all_tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)

                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.stop_tag_idx]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def score_sentence(self, feats, lengths, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.start_tag_idx], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.stop_tag_idx, tags[-1]]
        return score

    def neg_log_likelihood(self, sentence, lengths, tags):
        feats = self.get_emission_prob(sentence, lengths)
        forward_score = self.forward_alg(feats, lengths)
        gold_score = self.score_sentence(feats, tags, lengths)
        return forward_score - gold_score

    def forward(self, sentence, lengths):
        feats = self.get_emission_prob(sentence, lengths)
        score, tag_seq = self.viterbi_decode(feats, lengths)
        return score, tag_seq

    def viterbi_decode(self, feats, lengths):
        backpointers = []

        init_vvars = torch.full((1, self.all_tagset_size), -10000.)
        init_vvars[0][self.start_tag_idx] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.all_tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                _, best_tag_id = self.get_argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.stop_tag_idx]
        _, best_tag_id = self.get_argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.start_tag_idx
        best_path.reverse()
        return path_score, best_path