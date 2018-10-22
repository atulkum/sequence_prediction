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
    pred = torch.argmax(logits, dim=2)
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

    def get_loss(self, logits, y, s_len):
        log_smx = F.log_softmax(logits, dim=2)
        loss = F.nll_loss(log_smx.transpose(1, 2), y, ignore_index=self.pad_labelid, reduction='none')
        loss = loss.sum(dim=1) / s_len.float()
        loss = loss.mean()

        # might be reduction='sum' and divide s_lengths
        l2_reg = sum(p.norm(2) for p in self.parameters() if p.requires_grad)

        loss += self.reg_lambda * l2_reg
        return loss
