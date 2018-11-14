import numpy as np

def get_mask_from_seq_len(self, seq_mask):
    seq_lens = np.sum(seq_mask, 1)
    max_len = np.max(seq_lens)
    indices = np.arange(0, max_len)
    mask = (indices < np.expand_dims(seq_lens, 1)).astype(int)
    return mask