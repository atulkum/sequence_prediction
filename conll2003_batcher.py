from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torchtext import data
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import GloVe, CharNGram

import logging

logger = logging.getLogger(__name__)


def conll2003_dataset(tag_type, batch_size, root='./CoNLL-2003',
                      train_file='eng.train',
                      validation_file='eng.testa',
                      test_file='eng.testb',
                      convert_digits=False):
    TEXT_WORD = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, lower=True,
                             preprocessing=data.Pipeline(
                                 lambda w: '0' if convert_digits and w.isdigit() else w))

    CHAR_NESTING = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>",
                                     batch_first=True)

    TEXT_CHAR = data.NestedField(CHAR_NESTING,
                                   init_token="<bos>", eos_token="<eos>")

    LABELS = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

    fields = ([(('word', 'char'), (TEXT_WORD, TEXT_CHAR))] +
              [('labels', LABELS) if label == tag_type else (None, None)
               for label in ['pos', 'chunk', 'ner']])

    train, val, test = SequenceTaggingDataset.splits(
        path=root,
        train=train_file,
        validation=validation_file,
        test=test_file,
        separator=' ',
        fields=tuple(fields))

    logger.info('---------- CONLL 2003 %s ---------' % tag_type)
    logger.info('Train size: %d' % (len(train)))
    logger.info('Validation size: %d' % (len(val)))
    logger.info('Test size: %d' % (len(test)))


    TEXT_CHAR.build_vocab(train.inputs_char, val.inputs_char, test.inputs_char)
    TEXT_WORD.build_vocab(train.inputs_word, val.inputs_word, test.inputs_word, max_size=50000,
                            vectors=[GloVe(name='6B', dim='200'), CharNGram()])

    LABELS.build_vocab(train.labels)
    logger.info('Input vocab size:%d' % (len(TEXT_WORD.vocab)))
    logger.info('Tagset size: %d' % (len(LABELS.vocab)))

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    train_iter.repeat = False

    return {
        'task': 'conll2003.%s' % tag_type,
        'iters': (train_iter, val_iter, test_iter),
        'vocabs': (TEXT_WORD.vocab, TEXT_CHAR.vocab, LABELS.vocab)
    }

if __name__ == '__main__':
    batcher = conll2003_dataset('ner', 2)

    train_iter, val_iter, test_iter = batcher['iters']
    for batch in train_iter:
        print (batch)
        exit()