import os

class Config(object):
    pass

config = Config()
root_dir = os.path.join(os.path.expanduser('~'), 'Downloads/nlp_class')
config.data_dir = os.path.join(root_dir, 'neural_ner/CoNLL-2003')
config.log_root = os.path.join(root_dir, 'neural_ner/log')

config.train_file='eng.train.bioes'
config.validation_file='eng.testa.bioes'
config.test_file='eng.testb.bioes'
config.label_type='BIOES'

config.hidden_dim = 200
config.embedding_size=100

config.lr = 0.001
config.dropout_ratio = 0.15

config.max_grad_norm = 5.0
config.batch_size = 4
config.num_epochs = 50

config.print_every = 1000

config.reg_lambda = 0.00007