from tqdm import tqdm

import os
import re
import codecs
import numpy as np

from utils import create_freq_map, create_mapping
from sentence_utils import get_char_word_seq


class Vocab(object):
    _PAD = b"<pad>"
    _UNK = b"<unk>"
    _START_VOCAB = [_PAD, _UNK]
    PAD_ID = 0
    UNK_ID = 1

    def word_mapping(self, sentences, lower, zeros):
        words = []
        chars = []
        for s in sentences:
            str_words = [w[0] for w in s]
            word_seq, word_char_seq = get_char_word_seq(str_words, lower, zeros)
            words.append(word_seq)

            chars.append("".join("".join(char_seq) for char_seq in word_char_seq))

        word_freq_map = create_freq_map(words)
        word_freq_map['<UNK>'] = 10000000
        word_to_id, id_to_word = create_mapping(word_freq_map)
        print "Found %i unique words (%i in total)" % (
            len(word_freq_map), sum(len(x) for x in words)
        )

        char_freq_map = create_freq_map(chars)
        char_to_id, id_to_char = create_mapping(char_freq_map)
        print "Found %i unique characters" % len(char_freq_map)

        self.word_freq_map = word_freq_map
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

        self.char_freq_map = char_freq_map
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char

    def tag_mapping(self, sentences):
        tags = [[word[-1] for word in s] for s in sentences]
        freq_map = create_freq_map(tags)
        tag_to_id, id_to_tag = create_mapping(freq_map)
        print "Found %i unique named entity tags" % len(freq_map)

        self.tag_freq_map = freq_map
        self.tag_to_id = tag_to_id
        self.id_to_tag = id_to_tag

def augment_with_pretrained(dictionary, ext_emb_path, words):
    print 'Loading pretrained embeddings from %s...' % ext_emb_path
    assert os.path.isfile(ext_emb_path)

    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word

def get_glove(glove_path, glove_dim):
    print "Loading GLoVE vectors from file: %s" % glove_path
    vocab_size = int(4e5)

    emb_matrix = np.zeros((vocab_size + len(Vocab._START_VOCAB), glove_dim))
    word_to_id = {}
    id_to_word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(Vocab._START_VOCAB), :] = np.random.randn(len(Vocab._START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in Vocab._START_VOCAB:
        word_to_id[word] = idx
        id_to_word[idx] = word
        idx += 1

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if glove_dim != len(vector):
                raise Exception("glove_path=%s embedding_size=%i." % (glove_path, glove_dim))
            emb_matrix[idx, :] = vector
            word_to_id[word] = idx
            id_to_word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(Vocab._START_VOCAB)
    assert len(word_to_id) == final_vocab_size
    assert len(id_to_word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word_to_id, id_to_word


