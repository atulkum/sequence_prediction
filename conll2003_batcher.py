from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torchtext import data
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import GloVe

import logging

logging.getLogger().setLevel(logging.INFO)

class DatasetConll2003(object):
    def __init__(self, config, is_cuda=False):
        self.batch_size = config.batch_size
        self.is_cuda = is_cuda

        TEXT_WORD = data.Field(pad_token=None, unk_token='<unk>',
                               batch_first=True, lower=True, include_lengths=True)
        #CHAR_NESTING = data.Field(pad_token='<c>',
        #                          tokenize=list, batch_first=True)
        #TEXT_CHAR = data.NestedField(CHAR_NESTING, include_lengths=True)
        NER_LABELS = data.Field(unk_token=None, pad_token=None, batch_first=True, is_target=True)

        #fields = ([(('word', 'char'), (TEXT_WORD, TEXT_CHAR))] +
        fields = ([('word', TEXT_WORD)] +
                  [(None, None), (None, None), ('ner', NER_LABELS)])

        train, val, test = SequenceTaggingDataset.splits(
            path=config.data_dir,
            train=config.train_file,
            validation=config.validation_file,
            test=config.test_file,
            separator=' ',
            fields=fields)
        train.examples = [ex for ex in train.examples if ex.word != ['-DOCSTART-'.lower()]]
        val.examples = [ex for ex in val.examples if ex.word != ['-DOCSTART-'.lower()]]
        test.examples = [ex for ex in test.examples if ex.word != ['-DOCSTART-'.lower()]]

        self.train_ds = train
        self.val_ds = val
        self.test_ds = test

        logging.info('Train size: %d' % (len(train)))
        logging.info('Validation size: %d' % (len(val)))
        logging.info('Test size: %d' % (len(test)))

        #TEXT_CHAR.build_vocab(train.char, val.char, test.char)
        TEXT_WORD.build_vocab(train.word, val.word, test.word, max_size=50000,
                                vectors=[GloVe(name='6B', dim='200')])

        NER_LABELS.build_vocab(train.ner)

        self.TEXT_WORD = TEXT_WORD
        #self.char_vocab = TEXT_CHAR.vocab
        self.NER_LABELS = NER_LABELS

        logging.info('Input word vocab size:%d' % (len(self.TEXT_WORD.vocab)))
        #logging.info('Input char vocab size:%d' % (len(self.char_vocab)))
        logging.info('NER Tagset size: %d' % (len(self.NER_LABELS.vocab)))

        self.sort_key = lambda x: len(x.word)

    def get_train_iterator(self):
        train_iter = data.BucketIterator(
            self.train_ds, batch_size=self.batch_size, sort_key=self.sort_key,
            shuffle=True, device=torch.device("cuda:0" if self.is_cuda else "cpu"))
        train_iter.repeat = True
        return train_iter

    def get_val_iterator(self):
        return data.BucketIterator(
            self.val_ds, batch_size=self.batch_size, sort_key=self.sort_key,
            device=torch.device("cuda:0" if self.is_cuda else "cpu"))

    def get_test_iterator(self):
        return data.BucketIterator(
            self.test_ds, batch_size=self.batch_size, sort_key=self.sort_key,
            device=torch.device("cuda:0" if self.is_cuda else "cpu"))

    def get_default_labelid(self):
        return self.NER_LABELS.vocab.itos['O']

    def get_default_label(self):
        return 'O'

    def get_unk_wordid(self):
        return self.TEXT_WORD.vocab.itos[self.TEXT_WORD.unk_token]

    def label_ids2labels(self, y, s_len):
        return [self.NER_LABELS.vocab.itos[y[j]] for j in range(s_len)]

    def word_ids2words(self, s, s_len):
        return [self.TEXT_WORD.vocab.itos[s[j]] for j in range(s_len)]

    def get_labels(self):
        return self.NER_LABELS.vocab.itos

    def get_chunks(self, seq):
        default_label = 'O'
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default_label and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
            # End of a chunk + start of a chunk!
            elif tok != default_label:
                if chunk_type is None:
                    chunk_type, chunk_start = tok, i
                elif tok != chunk_type:
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok, i
            else:
                pass
        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)
        return chunks

if __name__ == '__main__':
    from config import config
    ds = DatasetConll2003(config)
    train_iter = ds.get_train_iterator()
    batch = next(iter(train_iter))

    (s, s_lengths), y = batch.word, batch.ner
    batch_examples =[]
    for i, ilen in enumerate(s_lengths):
        w = ds.word_ids2words(s[i], ilen)
        lb = ds.label_ids2labels(y[i], ilen)
        ch = ds.get_chunks(lb)
        batch_examples.append((w, lb, ch))
    for a in batch_examples:
        print (a)



