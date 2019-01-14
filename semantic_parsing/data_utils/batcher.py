import os
import gzip
import cPickle

from config import config

for fold in range(5):
    filename = os.path.join(config.data_dir, 'atis.fold' + str(fold) + '.pkl.gz')
    with gzip.open(filename, 'rb') as f:
        train_set, valid_set, test_set, dicts = cPickle.load(f)
        labels2idx_, tables2idx_, words2idx_ = dicts['labels2idx'], dicts['tables2idx'], dicts['words2idx']

        idx2labels = {v: k for k, v in labels2idx_.items()}
        idx2tables = {v: k for k, v in tables2idx_.items()}
        idx2words = {v: k for k, v in words2idx_.items()}

        train_x, train_ne, train_label = train_set

        for sentence, ne, label in zip(train_x, train_ne, train_label):
            print(sentence, ne, label)
            print (' '.join([idx2labels[i] for i in label])); print ('\n')
            print (' '.join([idx2tables[i] for i in ne])); print ('\n')
            print (' '.join([idx2words[i] for i in sentence])); print ('\n')

            exit()
