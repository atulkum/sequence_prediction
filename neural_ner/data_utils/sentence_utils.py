import re
import numpy as np

from data_utils.constant import Constants

def pad_items(items, is_tag=False):
    padded_items = []
    padded_items_len = [len(item) for item in items]
    max_length = max(padded_items_len)

    pad_id = Constants.TAG_PAD_ID if is_tag else Constants.PAD_ID
    for item in items:
        padding = [pad_id] * (max_length - len(item))
        padded_items.append(item + padding)

    return np.array(padded_items), np.array(padded_items_len)

def pad_chars(items):
    padded_items = []
    padded_items_len = [len(item) for item in items]
    max_length = max(padded_items_len)

    pad_id = Constants.PAD_ID
    for item in items:
        padding = [pad_id] * (max_length - len(item))
        padded_items.append(item + padding)

    return np.array(padded_items), np.array(padded_items_len)

def zero_digits(s):
    return re.sub('\d', '0', s)

def cap_feature(s):
    if s.lower() == s:
        return Constants.MAX_CAPS_FEATURE - 2
    elif s.upper() == s:
        return Constants.MAX_CAPS_FEATURE - 1
    elif s[0].upper() == s[0]:
        return Constants.MAX_CAPS_FEATURE
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

    words = [vocab.word_to_id.get(w, Constants.UNK_ID) for w in word_seq]
    chars = [[vocab.char_to_id.get(c, Constants.UNK_ID) for c in char_seq] for char_seq in word_char_seq]
    caps = [cap_feature(w) for w in str_words]

    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps,
        'raw_sentence':str_words
    }

if __name__ == '__main__':
    from config import config
    from data_utils.vocab import Vocab
    from data_utils.utils import prepare_dataset

    vocab = Vocab(config)

    sentences = [['a O', 'b O', 'c O', '| O']]

    data = prepare_dataset(sentences, vocab, config)
    datum = data[0]
    chars = datum['chars']
    chars_padded, chars_padded_lens = pad_chars(chars)
