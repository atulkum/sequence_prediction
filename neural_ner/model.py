from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_utils.sentence_utils import Constants
import logging
import numpy as np

from crf import CRF_Loss

print('pytorch version', torch.__version__)

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

def get_word_embd(vocab, config):
    word_to_vector = vocab.glove_vectors
    word_emb_matrix = np.random.uniform(low=-1.0, high=1.0,
                                        size=(len(vocab.word_to_id), config.word_emdb_dim))
    pretrained_init = 0
    for w, wid in vocab.word_to_id.items():
        if w in word_to_vector:
            word_emb_matrix[wid, :] = word_to_vector[w]
            pretrained_init += 1

    "Total words %i (%i pretrained initialization)" % (
        len(vocab.word_to_id), pretrained_init
    )
    return word_emb_matrix

def test_one_batch(batch, model):
    model.eval()
    logits = model(batch)
    lengths = batch['words_lens']
    pred = model.predict(logits, lengths)
    return logits, pred

def get_model(vocab, config, model_file_path, is_eval=False):
    model = NER_SOFTMAX_CHAR_CRF(vocab,  config)

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
        word_emb_matrix = get_word_embd(vocab, config)
        embd_vector = torch.from_numpy(word_emb_matrix).float()

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

        self.hidden2tag = nn.Linear(config.word_lstm_dim, len(vocab.id_to_tag))

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

    def neg_log_likelihood(self, logits, y, s_lens):
        log_smx = F.log_softmax(logits, dim=2)
        loss = F.nll_loss(log_smx.transpose(1, 2), y, ignore_index=Constants.TAG_PAD_ID, reduce=False)
        loss = loss.squeeze(1).sum(dim=1) / s_lens.float()
        loss = loss.mean()
        return loss

    def get_loss(self, logits, y, s_lens):
        loss = self.neg_log_likelihood(logits, y, s_lens)
        if self.config.is_l2_loss:
            loss += self.get_l2_loss()
        return loss

    def get_l2_loss(self):
        l2_reg = sum(p.norm(2) for p in self.parameters() if p.requires_grad)
        return self.config.reg_lambda * l2_reg

    def predict(self, logit, lengths):
        max_value, pred = torch.max(logit, dim=2)
        return pred

class NER_SOFTMAX_CHAR_CRF(nn.Module):
    def __init__(self, vocab, config):
        super(NER_SOFTMAX_CHAR_CRF, self).__init__()

        self.featurizer = NER_SOFTMAX_CHAR(vocab, config)
        self.crf = CRF_Loss(len(vocab.id_to_tag))
        self.config = config

    def forward(self, batch):
        emissions = self.featurizer(batch)
        return emissions

    def crf_loss(self, emissions, target):
        a = self.crf(emissions, target)
        loss = -1 * a
        loss = loss.mean()
        return loss

    def get_loss(self, logits, y, s_lens):
        loss = self.crf_loss(logits, y)
        if self.config.is_l2_loss:
            loss += self.get_l2_loss()
        return loss

    def predict(self, emissions, lengths):
        best_scores, pred = self.crf.viterbi_decode(emissions, lengths)
        return pred