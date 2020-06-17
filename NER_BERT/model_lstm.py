from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Conv1d, ReLU

from crf import CRF_Loss
import model_utils

class CHAR_CONV(torch.nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_filters,
                 ngram_filter_sizes=(2, 3, 4, 5)):
        super(CHAR_CONV, self).__init__()

        self.convolution_layers = torch.nn.ModuleList()
        for ngram_size in ngram_filter_sizes:
            conv_maxpool = torch.nn.ModuleList()
            conv_maxpool.extend([Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=ngram_size),
                ReLU(),
                torch.nn.MaxPool1d(kernel_size=num_filters)])

            self.convolution_layers.append(conv_maxpool)

    def forward(self, tokens, mask):
        tokens = tokens * mask.unsqueeze(-1).float()
        tokens = torch.transpose(tokens, 1, 2)

        filter_outputs = []
        for conv_maxpool in self.convolution_layers:
            filter_outputs.append(conv_maxpool(tokens))

        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        return maxpool_output

class NER_SOFTMAX_CHAR(nn.Module):
    def __init__(self, emb_matrix, config, num_tags):
        super(NER_SOFTMAX_CHAR, self).__init__()

        embd_vector = torch.from_numpy(emb_matrix['word']).float()
        self.word_embeds = nn.Embedding.from_pretrained(embd_vector, freeze=False)

        embd_vector = torch.from_numpy(emb_matrix['char']).float()
        self.char_embeds = nn.Embedding.from_pretrained(embd_vector, freeze=False)

        self.lstm_char = nn.LSTM(self.char_embeds.embedding_dim,
                            config.char_lstm_dim,
                            num_layers=1, bidirectional=True, batch_first=True)

        input_size = self.word_embeds.embedding_dim + config.char_lstm_dim * 2

        self.lstm = nn.LSTM(input_size,
                            config.word_lstm_dim,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.hidden_layer = nn.Linear(config.word_lstm_dim * 2, config.word_lstm_dim)
        self.tanh_layer = torch.nn.Tanh()

        self.hidden2tag = nn.Linear(config.word_lstm_dim, num_tags)

        self.config = config

        model_utils.init_lstm_wt(self.lstm_char)
        model_utils.init_lstm_wt(self.lstm)
        model_utils.init_linear_wt(self.hidden_layer)
        model_utils.init_linear_wt(self.hidden2tag)

    def forward(self, word_ids, mask, char_ids):
        lengths = mask.sum(1, dtype=torch.long)

        max_length = torch.max(lengths)
        char_emb = []
        word_embed = self.word_embeds(word_ids)
        for chars, char_mask in char_ids:
            char_len = char_mask.sum(1, dtype=torch.long)
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
            # fill up the dummy char embedding for padding
            seq_out = F.pad(seq_out, (0, 0, 0, max_length - seq_out.size(0)))
            char_emb.append(seq_out.unsqueeze(0))

        char_emb = torch.cat(char_emb, 0) #b x n x c_dim

        word_embed = torch.cat([char_emb, word_embed], 2)
        word_embed = self.dropout(word_embed)

        lengths = lengths.view(-1).tolist()
        packed = pack_padded_sequence(word_embed, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(packed)

        lstm_feats, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        lstm_feats = lstm_feats.contiguous()

        b, t_k, d = list(lstm_feats.size())

        h = self.hidden_layer(lstm_feats.view(-1, d))
        h = self.tanh_layer(h)
        logits = self.hidden2tag(h)
        logits = logits.view(b, t_k, -1)

        return logits


class NER_SOFTMAX_CHAR_CRF(nn.Module):
    def __init__(self, emb_matrix, config, tag_pad_id, num_tags):
        super(NER_SOFTMAX_CHAR_CRF, self).__init__()
        self.featurizer = NER_SOFTMAX_CHAR(emb_matrix, config, num_tags)
        self.crf = CRF_Loss(num_tags, config, tag_pad_id)
        self.config = config

    def forward(self, word_ids, mask, char_ids, labels=None):
        output = self.featurizer(word_ids, mask, char_ids)
        if labels is not None:
            loss = self.get_loss(output, labels, mask)
            output = (loss, output)
        return output

    def get_loss(self, logits, y, mask):
        if self.config.is_structural_perceptron_loss:
            loss = self.crf.structural_perceptron_loss(logits, y)
        else:
            loss = -1 * self.crf.log_likelihood(logits, y)

        s_lens = mask.sum(1, dtype=torch.long)

        loss = loss / s_lens.float()
        loss = loss.mean()
        return loss

    def predict(self, emissions, mask):
        best_scores, pred = self.crf.viterbi_decode_batch(emissions, mask)
        return best_scores, pred
