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
    DATA_TYPE_TRAIN = 'train'
    DATA_TYPE_TEST = 'test'
    DATA_TYPE_VAL = 'val'

    def __init__(self, config, is_cuda=False):
        self.batch_size = config.batch_size
        self.is_cuda = is_cuda
        self.num_special_toks = 2 #for '<pad>' and '<unk>'
        self.label_type = config.label_type

        TEXT_WORD = data.Field(pad_token='<pad>', unk_token='<unk>',
                               batch_first=True, lower=True, include_lengths=True)
        #CHAR_NESTING = data.Field(pad_token='<c>',
        #                          tokenize=list, batch_first=True)
        #TEXT_CHAR = data.NestedField(CHAR_NESTING, include_lengths=True)
        NER_LABELS = data.Field(pad_token='<pad>', unk_token=None, batch_first=True,
                                is_target=True, postprocessing=lambda arr, _: [[x-1 for x in ex] for ex in arr])

        #fields = ([(('word', 'char'), (TEXT_WORD, TEXT_CHAR))] +
        fields = ([('word', TEXT_WORD)] + [('ner', NER_LABELS)])

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

        self.labels = self.NER_LABELS.vocab.itos[1:]

        logging.info('Input word vocab size:%d' % (len(self.TEXT_WORD.vocab)))
        #logging.info('Input char vocab size:%d' % (len(self.char_vocab)))
        logging.info('NER Tagset size: %d' % (len(self.labels)))

        self.sort_key = lambda x: len(x.word)


    def get_train_iterator(self):
        return data.BucketIterator(
            self.train_ds, batch_size=self.batch_size, sort_key=self.sort_key,
            sort_within_batch = True, repeat = True,
            shuffle=True, device=torch.device("cuda:0" if self.is_cuda else "cpu"))

    def get_val_iterator(self):
        return data.BucketIterator(
            self.val_ds, batch_size=self.batch_size, sort_key=self.sort_key,
            sort_within_batch=True, repeat = False,
            device=torch.device("cuda:0" if self.is_cuda else "cpu"))


    def get_test_iterator(self):
        return data.BucketIterator(
            self.test_ds, batch_size=self.batch_size, sort_key=self.sort_key,
            sort_within_batch=True, repeat = False,
            device=torch.device("cuda:0" if self.is_cuda else "cpu"))

    def get_data_iterator(self, data_type):
        if data_type == DatasetConll2003.DATA_TYPE_TRAIN:
            return self.get_train_iterator()
        elif data_type == DatasetConll2003.DATA_TYPE_VAL:
            return self.get_val_iterator()
        elif data_type == DatasetConll2003.DATA_TYPE_TEST:
            return self.get_test_iterator()
        else:
            return None

    def get_default_labelid(self):
        return self.NER_LABELS.vocab.stoi['O']

    def get_pad_labelid(self):
        #as the post process substract 1 to each element the
        #pad has -1 index
        return -1

    def get_default_label(self):
        return 'O'

    def get_unk_wordid(self):
        return self.TEXT_WORD.vocab.itos[self.TEXT_WORD.unk_token]

    def label_ids2labels(self, y, s_len):
        return [self.labels[y[j]] for j in range(s_len)]

    def word_ids2words(self, s, s_len):
        return [self.TEXT_WORD.vocab.itos[s[j]] for j in range(s_len)]

    def get_labels(self):
        return self.labels

    @staticmethod
    def reverse_style(input_string):
        target_position = input_string.index('[')
        input_len = len(input_string)
        output_string = input_string[target_position:input_len] + input_string[0:target_position]
        return output_string

    @staticmethod
    def get_ner_BIOES(label_list):
        list_len = len(label_list)
        begin_label = 'B-'
        end_label = 'E-'
        single_label = 'S-'
        whole_tag = ''
        index_tag = ''
        tag_list = []
        stand_matrix = []
        for i in range(0, list_len):
            # wordlabel = word_list[i]
            current_label = label_list[i].upper()
            if begin_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

            elif single_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
                tag_list.append(whole_tag)
                whole_tag = ""
                index_tag = ""
            elif end_label in current_label:
                if index_tag != '':
                    tag_list.append(whole_tag + ',' + str(i))
                whole_tag = ''
                index_tag = ''
            else:
                continue
        if (whole_tag != '') & (index_tag != ''):
            tag_list.append(whole_tag)
        tag_list_len = len(tag_list)

        for i in range(0, tag_list_len):
            if len(tag_list[i]) > 0:
                tag_list[i] = tag_list[i] + ']'
                insert_list = DatasetConll2003.reverse_style(tag_list[i])
                stand_matrix.append(insert_list)
        # print stand_matrix
        return stand_matrix

    @staticmethod
    def get_ner_BIO(label_list):
        list_len = len(label_list)
        begin_label = 'B-'
        inside_label = 'I-'
        whole_tag = ''
        index_tag = ''
        tag_list = []
        stand_matrix = []
        for i in range(0, list_len):
            # wordlabel = word_list[i]
            current_label = label_list[i].upper()
            if begin_label in current_label:
                if index_tag == '':
                    whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                    index_tag = current_label.replace(begin_label, "", 1)
                else:
                    tag_list.append(whole_tag + ',' + str(i - 1))
                    whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                    index_tag = current_label.replace(begin_label, "", 1)

            elif inside_label in current_label:
                if current_label.replace(inside_label, "", 1) == index_tag:
                    whole_tag = whole_tag
                else:
                    if (whole_tag != '') & (index_tag != ''):
                        tag_list.append(whole_tag + ',' + str(i - 1))
                    whole_tag = ''
                    index_tag = ''
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''

        if (whole_tag != '') & (index_tag != ''):
            tag_list.append(whole_tag)
        tag_list_len = len(tag_list)

        for i in range(0, tag_list_len):
            if len(tag_list[i]) > 0:
                tag_list[i] = tag_list[i] + ']'
                insert_list = DatasetConll2003.reverse_style(tag_list[i])
                stand_matrix.append(insert_list)
        return stand_matrix

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

    def init_emb(self, init="randn"):
        num_special_toks = 2
        emb_vectors = self.TEXT_WORD.vocab.vectors
        sweep_range = len(self.TEXT_WORD.vocab)
        running_norm = 0.
        num_non_zero = 0
        total_words = 0
        for i in range(num_special_toks, sweep_range):
            if len(emb_vectors[i, :].nonzero()) == 0:
                # std = 0.05 is based on the norm of average GloVE 100-dim word vectors
                if init == "randn":
                    torch.nn.init.normal_(emb_vectors[i], mean=0, std=0.05)
            else:
                num_non_zero += 1
                running_norm += torch.norm(emb_vectors[i])
            total_words += 1

        logging.info("average GloVE norm is {}, number of known words are {}, total number of words are {}".format(
            running_norm / num_non_zero, num_non_zero, total_words))
        return emb_vectors

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
        ch = ds.get_ner_BIOES(lb)
        batch_examples.append((w, lb, ch))
    for a in batch_examples:
        print (a)




