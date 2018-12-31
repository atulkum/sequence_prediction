import os

class Config(object):
    pass

config = Config()
root_dir = os.path.join(os.path.expanduser('~'), 'Downloads/sequence_prediction-master')
config.data_dir = os.path.join(root_dir, 'CoNLL-2003')
config.log_root = os.path.join(root_dir, 'log')

config.train_file='eng.train.bio'
config.validation_file='eng.testa.bio'
config.test_file='eng.testb.bio'
config.label_type='BIO'

config.hidden_dim = 300

config.lr = 0.001
config.dropout_ratio = 0.15

config.max_grad_norm = 5.0
config.batch_size = 32
config.num_epochs = 10

config.print_every = 10

config.reg_lambda = 0.00007

config.dropout_rate = 0.5