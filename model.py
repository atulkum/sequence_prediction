from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def get_model(config, use_cuda, model_file_path=None, is_eval=False):
    model = NCRF(config.hidden_dim, config.maxout_pool_size,
                 self.emb_matrix, config.max_dec_steps, config.dropout_ratio)

    if is_eval:
        model = model.eval()
    if use_cuda:
        model = model.cuda()

    if model_file_path is not None:
        state = torch.load(model_file_path, map_location=lambda storage, location: storage)
        model.load_state_dict(state['model'], strict=False)

    return model

class NCRF(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, emb_matrix, max_dec_steps, dropout_ratio):
        super(NCRF, self).__init__()
        self.hidden_dim = hidden_dim


    def forward(self):
      pass