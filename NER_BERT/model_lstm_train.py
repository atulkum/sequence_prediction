from __future__ import absolute_import, division, print_function

import os.path

import os
import torch
import torch.nn as nn
import numpy as np
import random
import time
import json
import codecs

import const
import eval_util
import bert_data_util
import dataset_utils
import model_lstm
import model_utils
import lstm_data_util

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

is_cuda = torch.cuda.is_available()

class Config(object):
    def __init__(self):
        self.num_epoch = 10
        self.weight_decay = 0.0  # 1e-8
        self.batch_size = 8
        self.eval_batch_size = 8
        self.max_grad_norm = 1.0
        self.learning_rate = 5e-5
        self.adam_epsilon = 1e-8
        self.print_interval = 1000 * self.batch_size
        self.warmup_steps = 0.0
        self.max_seq_length = 100  # =32 - 2
        self.seed = 42
        self.tagging_type = 'B'
        self.word_emdb_dim = 100
        self.word_lstm_dim = 100
        self.char_embd_dim = 30
        self.char_lstm_dim = 64
        self.dropout_rate = 0.15
        self.is_structural_perceptron_loss = False

config = Config()

#####set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if is_cuda > 0:
    torch.cuda.manual_seed_all(config.seed)
#####set seed end
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_optimizer(model, config, t_total):
    optimizer = Adam(model.parameters(), amsgrad=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=t_total)
    return optimizer, scheduler


def predictions(dev_data, model, config):
    data_size = len(dev_data)
    ids = np.arange(data_size)
    eval_loss = 0
    model.eval()

    gold_label_list = []
    preds_list = []

    for i in range(0, data_size, config.eval_batch_size):
        batch_ids = ids[i:i + config.eval_batch_size]

        inputs = lstm_data_util.create_batch(dev_data, batch_ids, is_cuda)
        outputs_t = model(**inputs)
        loss_t, scores_ner_t = outputs_t[:2]
        best_scores, pred_tag_t = model.predict(scores_ner_t, inputs['mask'])

        eval_loss += loss_t.item()

        pred_tag = pred_tag_t.cpu().data.numpy()
        gold_tag = inputs['labels'].cpu().data.numpy()

        for k, bi in enumerate(batch_ids):
            s_len = len(dev_data[bi][0])
            predict_list = []
            gold_list = []
            for j in range(s_len):
                if gold_tag[k][j] == const.label_pad_id:
                    continue
                predict_list.append(pred_tag[k][j])
                gold_list.append(gold_tag[k][j])

            gold_label_list.append(gold_list)
            preds_list.append(predict_list)
    eval_loss /= data_size
    return preds_list, gold_label_list, eval_loss

def train(output_root_dir, embd, train_data, dev_data):
    num_tags = len(dataset_utils.get_tag_vocab(config))
    model = model_lstm.NER_SOFTMAX_CHAR_CRF(embd, config, const.label_pad_id, num_tags)

    if is_cuda:
        model = model.cuda()

    data_size = len(train_data)
    num_batch = np.ceil(data_size / config.batch_size)
    t_total = config.num_epoch * num_batch

    optimizer, scheduler = get_optimizer(model, config, t_total)

    exp_loss = None
    global_step = 0
    best_dev_f1 = 0
    model.zero_grad()
    ids = np.arange(data_size)
    for epoch in range(config.num_epoch):
        np.random.shuffle(ids)
        for i in range(0, data_size, config.batch_size):
            batch_ids = ids[i:i + config.batch_size]

            model.train()
            inputs = lstm_data_util.create_batch(train_data, batch_ids, is_cuda)
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            exp_loss = 0.99 * exp_loss + 0.01 * loss.item() if exp_loss else loss.item()

            if global_step > 0 and global_step % config.print_interval == 0:
                print(f'{global_step} / {t_total} train loss: {exp_loss} lr: {scheduler.get_lr()[0]}', flush=True)

        preds_list, gold_label_list, eval_loss = predictions(dev_data, model, config)
        results = eval_util.evaluate(preds_list, gold_label_list, config)
        print(f'{global_step}/{t_total} NER: p/r/f1 {results["precision"]:.5f}/{results["recall"]:.5f}/{results["f1"]:.5f}', flush=True)

        f1 = results['f1']
        if f1 > best_dev_f1:
            # output_dir = os.path.join(output_root_dir, 'checkpoint-{}'.format(epoch))
            output_dir = os.path.join(output_root_dir, 'checkpoint')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print(f"Saving model checkpoint to {output_dir}", flush=True)

            model.save_pretrained(output_dir)
            bert_data_util.bert_tokenizer.save_pretrained(output_dir)

            with open(os.path.join(output_dir, "training_config.json"), 'w') as fout:
                json.dump(vars(config), fout)

    print(f'{global_step} / {t_total} train loss: {exp_loss} lr: {scheduler.get_lr()[0]}', flush=True)


def process_train(root_dir, data_dir, glove_path):
    output_root_dir = os.path.join(root_dir, f'dl_model_{int(time.time())}')
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    print(f'model out dir {output_root_dir}', flush=True)

    train_data_seq = dataset_utils.get_data_train_dev(data_dir, 'train.jsonl', config)
    dev_data_seq = dataset_utils.get_data_train_dev(data_dir, 'dev.jsonl', config)

    word_emb_matrix, word2id, id2word = model_utils.get_word_embd(config, glove_path, train_data_seq)
    char_emb_matrix, char2id, id2char = model_utils.get_char_embd(config, id2word)

    vocab = {
        'word2id': word2id,
        'id2word': id2word,
        'char2id': char2id,
        'id2char': id2char
    }
    embd = {
        'word': word_emb_matrix,
        'char': char_emb_matrix
    }
    with codecs.open(os.path.join(output_root_dir, 'word.vocab'), "w", encoding="utf8") as f:
        f.write('\n'.join(id2word) + '\n')
    with codecs.open(os.path.join(output_root_dir, 'char.vocab'), "w", encoding="utf8") as f:
        f.write('\n'.join(id2char) + '\n')

    train_data = lstm_data_util.get_data(train_data_seq, vocab, config)
    dev_data = lstm_data_util.get_data(dev_data_seq, vocab, config)

    train(output_root_dir, embd, train_data, dev_data)


def get_model(model_dir, vocab, num_tag):
    embd = model_utils.get_random_embedding(vocab, config)
    model = model_lstm.NER_SOFTMAX_CHAR_CRF(embd, config, const.label_pad_id, num_tag)

    #load model
    save_directory = os.path.join(root_dir, model_dir + '/checkpoint')
    model_file_path = os.path.join(save_directory, "pytorch_model.bin")
    print(f'reading model from {model_file_path}')
    state_dict = torch.load(model_file_path, map_location=lambda storage, location: storage)
    model.eval()
    model.load_state_dict(state_dict, strict=False)

    if is_cuda:
        model = model.cuda()
    return model

def process_eval(root_dir, model_dir, data_dir, filename):
    tag_vocab = dataset_utils.get_tag_vocab(config)
    tag2id = {t: i for i, t in enumerate(tag_vocab)}

    with codecs.open(os.path.join(model_dir, 'word.vocab'), "r", encoding="utf8") as f:
        id2word = f.readlines().split('\n')
        word2id = {v: k for k, v in enumerate(id2word)}
    with codecs.open(os.path.join(model_dir, 'char.vocab'), "r", encoding="utf8") as f:
        id2char = f.readlines().split('\n')
        char2id = {v: k for k, v in enumerate(id2char)}
    vocab = {
        'word2id': word2id,
        'id2word': id2word,
        'char2id': char2id,
        'id2char': id2char
    }
    model = get_model(model_dir, vocab, len(tag2id))

    test_data_seq, metadata = dataset_utils.get_data_test(data_dir, filename)
    test_data = lstm_data_util.get_data(test_data_seq, vocab, config)

    preds_list, _, _ = predictions(test_data, model, config)
    eval_util.dump_result(preds_list, metadata, test_data, root_dir, 'boundary_model.txt', config)

if __name__ == "__main__":
    prefix = 'prop' #'private-projects/propganda'
    root_dir = os.path.join(os.path.expanduser("~"), prefix + '/exp')
    data_dir = os.path.join(os.path.expanduser("~"), prefix + '/datasets')
    glove_dir = os.path.join(os.path.expanduser("~"), 'dl_entity/glove.6B/glove.6B.100d.txt')
    process_train(root_dir, data_dir, glove_dir)
    #model_dir = os.path.join(os.path.expanduser("~"), 'private-projects/propganda/exp/dl_model_1579925959')
    #process_eval(root_dir, model_dir, data_dir, 'test_phase0.jsonl')

