from __future__ import unicode_literals, print_function, division

import torch
import numpy as np

def get_mask(lengths, config):
    seq_lens = lengths.view(-1, 1)
    max_len = torch.max(seq_lens)
    range_tensor = torch.arange(max_len).unsqueeze(0)
    range_tensor = range_tensor.expand(seq_lens.size(0), range_tensor.size(1))
    if config.is_cuda:
        range_tensor = range_tensor.cuda()
    mask = (range_tensor < seq_lens).float()
    return mask

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

