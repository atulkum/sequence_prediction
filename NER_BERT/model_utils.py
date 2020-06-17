import numpy as np
import codecs
import re
from collections import Counter

import const

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


def get_glove(glove_path):
    print("Loading GLoVE vectors from file: {}".format(glove_path))
    word_to_vector = {}

    # go through glove vecs
    with codecs.open(glove_path, 'r', 'utf-8') as fh:
        for line in fh:
            line = re.split('\s+', line.strip())
            word = line[0]
            vector = list(map(float, line[1:]))
            word_to_vector[word] = vector

    return word_to_vector

def get_word_embd(config, glove_path, examples):
    word_to_vector = get_glove(glove_path)

    word_freq_map = Counter()
    for tokens, tags in examples:
        word_freq_map.update(tokens)

    word_freq_map.update(word_to_vector.keys())
    orig_tokens = set([w for w, ct in word_freq_map.most_common()])
    orig_tokens = sorted(list(orig_tokens.union(word_to_vector.keys())))

    id_to_word = const._START_VOCAB.copy()
    id_to_word.extend(orig_tokens)

    word_to_id = {v: k for k, v in enumerate(id_to_word)}

    word_emb_matrix = np.random.uniform(low=-1.0, high=1.0,
                                        size=(len(id_to_word), config.word_emdb_dim))
    pretrained_init = 0
    for wid, w in enumerate(id_to_word):
        if w in word_to_vector:
            word_emb_matrix[wid, :] = word_to_vector[w]
            pretrained_init += 1

    return word_emb_matrix, word_to_id, id_to_word

def get_char_embd(config, id_to_word):
    char_freq_map = Counter()
    for w in id_to_word:
        char_freq_map.update([c for c in w])

    id_to_char = const._START_VOCAB.copy()
    id_to_char.extend([c for c, ct in char_freq_map.most_common()])

    char_to_id = {v: k for k, v in enumerate(id_to_char)}

    char_emb_matrix = np.random.uniform(low=-1.0, high=1.0,
                                        size=(len(id_to_char), config.char_embd_dim))

    return char_emb_matrix, char_to_id, id_to_char

def get_random_embedding(vocab, config):
    word_emb_matrix = np.random.uniform(low=-1.0, high=1.0,
                                        size=(len(vocab['id2word']), config.word_emdb_dim))

    char_emb_matrix = np.random.uniform(low=-1.0, high=1.0,
                                        size=(len(vocab['id2char']), config.char_embd_dim))

    embd = {
        'word': word_emb_matrix,
        'char': char_emb_matrix
    }
    return embd
