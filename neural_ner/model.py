from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_utils.sentence_utils import Constants
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

np.random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                drange = np.sqrt(6. / (np.sum(wt.size())))
                wt.data.uniform_(-drange, drange)

            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    drange = np.sqrt(6. / (np.sum(linear.weight.size())))
    linear.weight.data.uniform_(-drange, drange)

    if linear.bias is not None:
        linear.bias.data.fill_(0.)

def test_one_batch(batch, model):
    model.eval()
    logits = model(batch)
    _, pred = model.get_argmax(logits)
    return logits, pred

def get_model(vocab, config, model_file_path, is_eval=False):
    model = NER_SOFTMAX_CHAR(vocab,  config)

    if is_eval:
        model = model.eval()
    if config.is_cuda:
        model = model.cuda()

    if model_file_path is not None:
        state = torch.load(model_file_path, map_location=lambda storage, location: storage)
        model.load_state_dict(state['model'], strict=False)

    return model

class NER_SOFTMAX_CHAR(nn.Module):
    def __init__(self, vocab, config):
        super(NER_SOFTMAX_CHAR, self).__init__()

        embd_vector = torch.from_numpy(vocab.get_word_embd()).float()
        tagset_size = len(vocab.id_to_tag)

        self.word_embeds = nn.Embedding.from_pretrained(embd_vector, freeze=False)
        self.char_embeds = nn.Embedding(len(vocab.char_to_id), config.char_embd_dim, padding_idx=Constants.PAD_ID)
        self.caps_embeds = nn.Embedding(vocab.get_caps_cardinality(), config.caps_embd_dim, padding_idx=Constants.PAD_ID)

        self.lstm_char = nn.LSTM(self.char_embeds.embedding_dim,
                            config.char_lstm_dim,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.lstm = nn.LSTM(self.word_embeds.embedding_dim + config.char_embd_dim * 2 + config.caps_embd_dim,
                            config.word_lstm_dim,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.hidden_layer = nn.Linear(config.word_lstm_dim * 2, config.word_lstm_dim)
        self.tanh_layer = torch.nn.Tanh()
        self.hidden2tag = nn.Linear(config.word_lstm_dim, tagset_size)

        self.config = config

        init_lstm_wt(self.lstm_char)
        init_lstm_wt(self.lstm)
        init_linear_wt(self.hidden_layer)
        init_linear_wt(self.hidden2tag)
        self.char_embeds.weight.data.uniform_(-1., 1.)
        self.caps_embeds.weight.data.uniform_(-1., 1.)

    def forward(self, batch):
        sentence = batch['words']
        lengths = batch['words_lens']
        caps = batch['caps']

        char_emb = []
        word_embed = self.word_embeds(sentence)
        for chars, char_len in batch['chars']:
            seq_embed = self.char_embeds(chars)
            seq_lengths, sort_idx = torch.sort(char_len, descending=True)
            _, unsort_idx = torch.sort(sort_idx)
            seq_embed = seq_embed[sort_idx]

            packed = pack_padded_sequence(seq_embed, seq_lengths, batch_first=True)
            output, hidden = self.lstm_char(packed)
            lstm_feats, _ = pad_packed_sequence(output, batch_first=True)
            lstm_feats = lstm_feats.contiguous()
            b, t_k, d = list(lstm_feats.size())

            seq_rep = lstm_feats.view(b, t_k, 2, -1) #0 is fwd and 1 is bwd

            last_idx = char_len - 1
            seq_rep_fwd = seq_rep[unsort_idx, 0, 0]
            seq_rep_bwd = seq_rep[unsort_idx, last_idx, 1]

            seq_out = torch.cat([seq_rep_fwd, seq_rep_bwd], 1)
            char_emb.append(seq_out.unsqueeze(0))
        char_emb = torch.cat(char_emb, 0) #b x n x c_dim

        caps_embd = self.caps_embeds(caps)

        word_embed = torch.cat([char_emb, word_embed, caps_embd], 2)
        word_embed = self.dropout(word_embed)

        lengths = lengths.view(-1).tolist()
        packed = pack_padded_sequence(word_embed, lengths, batch_first=True)
        output, hidden = self.lstm(packed)

        lstm_feats, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        lstm_feats = lstm_feats.contiguous()

        b, t_k, d = list(lstm_feats.size())

        h = self.hidden_layer(lstm_feats.view(-1, d))
        h = self.tanh_layer(h)
        logits = self.hidden2tag(h)
        logits = logits.view(b, t_k, -1)

        return logits

    def neg_log_likelihood(self, logits, y, s_len):
        log_smx = F.log_softmax(logits, dim=2)
        loss = F.nll_loss(log_smx.transpose(1, 2), y, ignore_index=Constants.TAG_PAD_ID, reduction='none')
        loss = loss.squeeze(1).sum(dim=1) / s_len.float()
        loss = loss.mean()
        if self.config.is_l2_loss:
            l2_reg = sum(p.norm(2) for p in self.parameters() if p.requires_grad)
            loss += self.config.reg_lambda * l2_reg
        return loss

    def get_argmax(self, logits):
        max_value, max_idx = torch.max(logits, dim=2)
        return max_value, max_idx

class NER_CRF(nn.Module):
    def __init__(self, embd_vector, hidden_dim, tagset_size,
                 reg_lambda):
        super(NER_CRF, self).__init__()

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
        nn.init.uniform_(self.transitions, -0.1, 0.1)

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

    def get_log_z(self, emission_prob):
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

    def get_log_p_y_x(self, feats, lengths, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.start_tag_idx], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.stop_tag_idx, tags[-1]]
        return score

    def neg_log_likelihood(self, sentence, lengths, tags):
        feats = self.get_emission_prob(sentence, lengths)
        log_z = self.get_log_z(feats, lengths)
        log_p_y_x = self.get_log_p_y_x(feats, tags, lengths)
        return -(log_p_y_x - log_z)

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

