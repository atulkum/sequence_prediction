from tqdm import tqdm

import os
import numpy as np

from sentence_utils import get_char_word_seq, Constants
from utils import load_sentences
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
        self.config.vocab_size = min(self.config.vocab_size, len(word_freq_map))
        sorted_items = word_freq_map.most_common(self.config.vocab_size)

        id_to_word = {i + start_vocab_len: v[0] for i, v in enumerate(sorted_items)}
        for i, v in enumerate(Constants._START_VOCAB):
            id_to_word[i] = v

        print "Found %i unique words (%i in total)" % (
            len(word_freq_map), sum(len(x) for x in words)
        )
        self.word_to_id = {v: k for k, v in id_to_word.items()}
        self.id_to_word = id_to_word

        char_freq_map = create_freq_map(chars)

        id_to_char = {i+start_vocab_len: v for i, v in enumerate(char_freq_map)}
        for i, v in enumerate(Constants._START_VOCAB):
            id_to_char[i] = v

        print "Found %i unique characters" % len(char_freq_map)

        self.char_to_id = {v: k for k, v in id_to_char.items()}
        self.id_to_char = id_to_char

    def tag_mapping(self, sentences):
        tags = [[word[-1] for word in s] for s in sentences]
        freq_map = create_freq_map(tags)
        id_to_tag = {i: v for i, v in enumerate(freq_map)}
        print "Found %i unique named entity tags" % len(freq_map)

        self.tag_to_id = {v: k for k, v in id_to_tag.items()}
        self.id_to_tag = id_to_tag

    def get_word_embd(self):
        gemb_matrix, gword_to_id, gid_to_word = Vocab.get_glove(self.config.glove_path, self.config.word_emdb_dim)

        start_vocab_len = len(Constants._START_VOCAB)
        word_emb_matrix = np.random.uniform(low=-1.0, high=1.0, size=(self.config.vocab_size + start_vocab_len, self.config.word_emdb_dim)) #np.zeros((self.config.vocab_size + start_vocab_len, self.config.word_emdb_dim))

        # randomly initialize the special tokens otherwise 0 init
        #if self.config.random_init:
        #    word_emb_matrix[:start_vocab_len, :] = np.random.uniform(low=-1.0, high=1.0, size=(start_vocab_len, self.config.word_emdb_dim)) #np.random.randn(start_vocab_len, self.config.word_emdb_dim)

        pretrained_init = 0
        for wid in range(start_vocab_len, len(self.word_to_id)):
            w = self.id_to_word[wid]
            if w in gword_to_id:
                word_emb_matrix[wid, :] = gemb_matrix[gword_to_id[w], :]
                pretrained_init += 1
            #elif self.config.random_init:
            #    word_emb_matrix[wid, :] = np.random.uniform(low=-1.0, high=1.0, size=(1, self.config.word_emdb_dim)) #np.random.randn(1, self.config.word_emdb_dim)

        print "Total words %i (%i pretrained initialization)" % (
            len(self.word_to_id), pretrained_init
        )
        return word_emb_matrix

    @staticmethod
    def get_glove(glove_path, glove_dim):
        print "Loading GLoVE vectors from file: %s" % glove_path
        vocab_size = int(4e5)

        emb_matrix = np.zeros((vocab_size, glove_dim))
        word_to_id = {}
        id_to_word = {}

        # put start tokens in the dictionaries
        idx = 0
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

        return emb_matrix, word_to_id, id_to_word

if __name__ == '__main__':
    from config import config
    vocab = Vocab(config)

    emb_matrix = vocab.get_word_embd()
    print len(emb_matrix)