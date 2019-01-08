#conll2003

import os
import numpy as np
from utils import load_sentences, prepare_dataset
from vocab import Vocab
from sentence_utils import pad_items, pad_chars
import torch

class DatasetConll2003(object):
    DATA_TYPE_TRAIN = 'train'
    DATA_TYPE_TEST = 'test'
    DATA_TYPE_EVAL = 'eval'

    def __init__(self, data_type, config, vocab, is_train):
        self.config = config
        self.vocab = vocab
        filepath = DatasetConll2003.get_data_file( data_type, config)

        sentences = load_sentences(os.path.join(config.data_dir, filepath), config.label_type)
        self.data = prepare_dataset(sentences, self.vocab, config)

        self.i = 0
        self.is_train = is_train
        self.epoch = 0
        self.iterations = 0

    @staticmethod
    def get_data_file(data_type, config):
        if data_type == DatasetConll2003.DATA_TYPE_TEST:
            filepath = config.test_file
        elif data_type == DatasetConll2003.DATA_TYPE_EVAL:
            filepath = config.validation_file
        else:
            filepath = config.train_file

        return filepath

    def __iter__(self):
        return self

    def next(self):
        self.iterations += 1

        if self.is_train and self.i >= len(self.data):
            np.random.shuffle(self.data)
            self.i = 0
            self.epoch += 1

        if self.i < len(self.data):
            batch = self.data[self.i:self.i + self.config.batch_size]
            self.i += self.config.batch_size

            return self.prepare_batch(batch)
        else:
            raise StopIteration()

    def prepare_batch(self, batch):
        words_len = np.array([len(datum['words']) for datum in batch])
        idx = np.argsort(words_len)[::-1]

        features = ['words', 'caps', 'tags']
        prepared_batch = {v:[] for v in features}
        prepared_batch['words_lens'] = words_len[idx]

        all_chars = []
        raw_sentences = []

        max_length = max(words_len)

        for i in idx:
            datum = batch[i]
            for v in features:
                prepared_batch[v].append(datum[v])

            raw_sentences.append(datum['raw_sentence'])

            chars = datum['chars']
            chars_padded, chars_padded_lens = pad_chars(chars, max_length)
            chars_padded = torch.Tensor(chars_padded).long()
            chars_padded_lens = torch.Tensor(chars_padded_lens).long()
            if self.config.is_cuda:
                chars_padded = chars_padded.cuda()
                chars_padded_lens = chars_padded_lens.cuda()

            all_chars.append((chars_padded, chars_padded_lens))

        for v in features:
            padded, _ = pad_items(prepared_batch[v], (v == 'tags'))
            prepared_batch[v] = padded

        for v in features + ['words_lens']:
            prepared_batch[v] = torch.Tensor(prepared_batch[v]).long()
            if self.config.is_cuda:
                prepared_batch[v] = prepared_batch[v].cuda()

        prepared_batch['chars'] = all_chars
        prepared_batch['raw_sentence'] = raw_sentences

        return prepared_batch

if __name__ == '__main__':
    from config import config

    vocab = Vocab(config)
    train_iter = DatasetConll2003(config.test_file, config, vocab, False)
    batch = next(iter(train_iter))

    print (batch)