import re

def pad_word_chars(words, pad_id):
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [pad_id] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos

def zero_digits(s):
    return re.sub('\d', '0', s)


def cap_feature(s):
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

def get_char_word_seq(str_words, lower=False, zeros=False):
    word_seq = []
    word_char_seq = []

    for w in str_words:
        w = zero_digits(w) if zeros else w

        w_lower = w.lower() if lower else w
        word_seq.append(w_lower)

        char_seq = [c for c in w]
        word_char_seq.append(char_seq)

    return word_seq, word_char_seq

def prepare_sentence(str_words, word_to_id, char_to_id, lower=False, zeros=False):
    word_seq, word_char_seq = get_char_word_seq(str_words, lower, zeros)

    words = [w if w in word_to_id else '<UNK>' for w in word_seq]
    chars = [[char_to_id[c] for c in char_seq if c in char_to_id] for char_seq in word_char_seq]
    caps = [cap_feature(w) for w in word_seq]

    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }

