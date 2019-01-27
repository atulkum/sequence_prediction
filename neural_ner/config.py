import os
import torch
import numpy as np

print('pytorch version', torch.__version__)

np.random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class Config(object):
    pass

config = Config()
root_dir = os.path.join(os.path.expanduser('~'), 'sequence_prediction')
config.data_dir = os.path.join(root_dir, 'CoNLL-2003')
config.log_root = os.path.join(root_dir, 'log')

config.train_file='eng.train'
config.validation_file='eng.testa'
config.test_file='eng.testb'

config.label_type= 'iobes'
#config.label_type= 'iob'

config.lr = 0.0015
config.lr_decay = 0.05
config.dropout_ratio = 0.5
config.momentum = 0.9

config.max_grad_norm = 5.0
config.batch_size = 32
config.num_epochs = 100

config.print_every = 100

config.reg_lambda = 1e-8

config.dropout_rate = 0.5

config.lower = True
config.zeros = True
config.random_init = True

config.char_embd_dim = 30
config.char_lstm_dim = 30
config.word_emdb_dim = 100
config.word_lstm_dim = 100
config.caps_embd_dim = 3

config.glove_path = os.path.join(root_dir, 'glove.6B/glove.6B.100d.txt')

config.vocab_size = int(4e5)

config.is_cuda = True

config.is_l2_loss = True

config.model_name = 'model.NER_SOFTMAX_CHAR_CRF'
config.optimizer = 'adam'

config.use_pretrain_embd = True

config.verbose = False
config.is_caps=True
config.is_structural_perceptron_loss=False

config.input_format='conll2003' #crfsuite


# config postprocess
config.is_cuda = config.is_cuda and torch.cuda.is_available()

