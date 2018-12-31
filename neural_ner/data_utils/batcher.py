#conll2003

import codecs
import numpy as np
from sentence_utils import prepare_sentence, pad_word_chars


class DatasetConll2003(object):
    def __init__(self):
        self.vocab = None


    def load_sentences(self, path):
        sentences = []
        sentence = []
        for line in codecs.open(path, 'r', 'utf8'):
            line = line.rstrip()
            if not line:
                if len(sentence) > 0:
                    if 'DOCSTART' not in sentence[0][0]:
                        sentences.append(sentence)
                    sentence = []
            else:
                word = line.split()
                assert len(word) >= 2
                sentence.append(word)
        if len(sentence) > 0:
            if 'DOCSTART' not in sentence[0][0]:
                sentences.append(sentence)
        return sentences


    def prepare_dataset(self, sentences, word_to_id, char_to_id, tag_to_id, lower=False):
        data = []
        for s in sentences:
            str_words = [w[0] for w in s]
            datum = prepare_sentence(str_words, word_to_id, char_to_id, lower)
            tags = [tag_to_id[w[-1]] for w in s]
            datum['tags'] = tags
            data.append(datum)
        return data


    def create_input(self, data, parameters, add_label, singletons=None):
        input = []
        if parameters['word_dim']:
            words = data['words']
            if singletons is not None:
                words = self.insert_singletons(words, singletons)
            input.append(words)
        if parameters['char_dim']:
            chars = data['chars']
            char_for, char_rev, char_pos = pad_word_chars(chars, VOCAB.PAD_ID)
            input.append(char_for)
            if parameters['char_bidirect']:
                input.append(char_rev)
            input.append(char_pos)
        if parameters['cap_dim']:
            caps = data['caps']
            input.append(caps)
        if add_label:
            input.append(data['tags'])
        return input


    def insert_singletons(self, words, singletons, p=0.5):
        new_words = []
        for word in words:
            if word in singletons and np.random.uniform() < p:
                new_words.append(0)
            else:
                new_words.append(word)
        return new_words

