import re
import numpy as np

class Constants(object):
    _UNK = b"<UNK>"
    _PAD = b"<PAD>"
    _START_VOCAB = [_UNK, _PAD]
    UNK_ID = 0
    PAD_ID = 1
    TAG_PAD_ID = -1

def pad_items(items, is_tag=False):
    padded_items = []
    padded_items_len = [len(item) for item in items]
    max_length = max(padded_items_len)

    pad_id = Constants.TAG_PAD_ID if is_tag else Constants.PAD_ID
    for item in items:
        padding = [pad_id] * (max_length - len(item))
        padded_items.append(item + padding)

    return np.array(padded_items), np.array(padded_items_len)

def pad_chars(items, max_word_len):
    padded_items = []
    padded_items_len = [len(item) for item in items]
    max_length = max(padded_items_len)

    pad_id = Constants.PAD_ID
    for item in items:
        padding = [pad_id] * (max_length - len(item))
        padded_items.append(item + padding)
    for i in xrange(len(items), max_word_len):
        padding = [pad_id] * max_length
        padded_items.append(padding)
        padded_items_len.append(1)

    return np.array(padded_items), np.array(padded_items_len)

def zero_digits(s):
    return re.sub('\d', '0', s)

def cap_feature(s):
    if s.lower() == s:
        return 2
    elif s.upper() == s:
        return 3
    elif s[0].upper() == s[0]:
        return 4
    else:
        return Constants.UNK_ID

def get_char_word_seq(str_words, lower, zeros):
    word_seq = []
    word_char_seq = []

    for w in str_words:
        w = zero_digits(w) if zeros else w
        w_lower = w.lower() if lower else w
        word_seq.append(w_lower)

        char_seq = [c for c in w]
        word_char_seq.append(char_seq)

    return word_seq, word_char_seq

def prepare_sentence(s, vocab, config):
    str_words = [w[0] for w in s]
    word_seq, word_char_seq = get_char_word_seq(str_words, config.lower, config.zeros)

    words = [vocab.word_to_id[w] if w in vocab.word_to_id else Constants.UNK_ID for w in word_seq]
    chars = [[vocab.char_to_id[c] for c in char_seq if c in vocab.char_to_id] for char_seq in word_char_seq]
    caps = [cap_feature(w) for w in str_words]

    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps,
        'raw_sentence':str_words
    }

