from tqdm import tqdm

import os
import codecs
import re
import numpy as np

from .sentence_utils import get_char_word_seq, Constants
from .utils import load_sentences
from collections import Counter

def create_freq_map(item_list):
    assert type(item_list) is list
    freq_map = Counter()
    for items in item_list:
        freq_map.update(items)
    return freq_map

class Vocab(object):
    def __init__(self, config):
        self.config = config
        train_sentences = load_sentences(os.path.join(config.data_dir, config.train_file), config.label_type)
        self.word_mapping(train_sentences)
        self.tag_mapping(train_sentences)

    def get_caps_cardinality(self):
        return Constants.MAX_CAPS_FEATURE + 1

    def word_mapping(self, sentences):
        words = []
        chars = []
        for s in sentences:
            str_words = [w[0] for w in s]
            word_seq, word_char_seq = get_char_word_seq(str_words, self.config.lower, self.config.zeros)
            words.append(word_seq)

            chars.append("".join("".join(char_seq) for char_seq in word_char_seq))

        start_vocab_len = len(Constants._START_VOCAB)
        word_freq_map = create_freq_map(words)
        self.orig_word_freq_map = word_freq_map.copy()
        print ("Found {} unique words ({} in total)".format(
            len(word_freq_map), sum(len(x) for x in words)
        ))
        '''
        self.config.vocab_size = min(self.config.vocab_size, len(word_freq_map))
        sorted_items = word_freq_map.most_common(self.config.vocab_size)
        id_to_word = {i + start_vocab_len: v[0] for i, v in enumerate(sorted_items)}
        '''
        #augmnet with pretrained words
        self.glove_vectors = self.get_glove()
        word_freq_map.update(self.glove_vectors.keys())

        id_to_word = {}
        for i, v in enumerate(Constants._START_VOCAB):
            id_to_word[i] = v

        for v in word_freq_map:
            id_to_word[len(id_to_word)] = v

        self.word_to_id = {v: k for k, v in id_to_word.items()}
        self.id_to_word = id_to_word

        id_to_char = {}
        for i, v in enumerate(Constants._START_VOCAB):
            id_to_char[i] = v

        char_freq_map = create_freq_map(chars)

        for v in char_freq_map:
            id_to_char[len(id_to_char)] = v

        print("Found {} unique characters".format(len(char_freq_map)))

        self.char_to_id = {v: k for k, v in id_to_char.items()}
        self.id_to_char = id_to_char

    def tag_mapping(self, sentences):
        tags = [[word[-1] for word in s] for s in sentences]
        freq_map = create_freq_map(tags)
        id_to_tag = {i: v for i, v in enumerate(freq_map)}
        print ("Found {} unique named entity tags" .format(len(freq_map)))

        self.tag_to_id = {v: k for k, v in id_to_tag.items()}
        self.id_to_tag = id_to_tag

    def get_glove(self):
        print ("Loading GLoVE vectors from file: {}".format(self.config.glove_path))
        vocab_size = int(4e5)
        word_to_vector = {}

        # go through glove vecs
        with codecs.open(self.config.glove_path, 'r', 'utf-8') as fh:
            for line in tqdm(fh, total=vocab_size):
                line = re.split('\s+', line.strip())
                word = line[0]
                vector = list(map(float, line[1:]))
                if self.config.word_emdb_dim != len(vector):
                    raise Exception("glove_path=%s embedding_size=%i." %
                                    (self.config.glove_path, self.config.word_emdb_dim))

                word_to_vector[word] = vector

        return word_to_vector

if __name__ == '__main__':
    from config import config
    vocab = Vocab(config)
    print (len(vocab.word_to_id))
    #emb_matrix = vocab.get_word_embd()
    #print len(emb_matrix)