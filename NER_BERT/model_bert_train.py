from __future__ import absolute_import, division, print_function

import os.path

import os
import torch
import torch.nn as nn
import numpy as np
import random
import time
import json

import const
import eval_util
import bert_data_util
import dataset_utils

from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

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

config = Config()

#####set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if is_cuda > 0:
    torch.cuda.manual_seed_all(config.seed)
#####set seed end

class MyBertForTokenClassification(nn.Module):
    def __init__(self, num_tags):
        super(MyBertForTokenClassification, self).__init__()
        self.bert = BertModel.from_pretrained(const.MODEL_TYPE)

        self.config = BertConfig.from_pretrained(const.MODEL_TYPE)
        self.config.num_tags = num_tags

        self.dropout_ner = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier_ner = nn.Linear(self.config.hidden_size, num_tags)

        self.classifier_ner.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.classifier_ner.bias.data.zero_()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # NER
        sequence_output = outputs[0]
        sequence_output = self.dropout_ner(sequence_output)
        logits_ner = self.classifier_ner(sequence_output)

        outputs = (logits_ner,) + outputs[2:]  # add hidden states and attention if they are here
        return outputs  # (loss), scores_ner, (hidden_states), (attentions)

    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), output_model_file)

    def get_loss_greedy(self, logits_ner, labels, attention_mask):
        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits_ner.view(-1, self.model_config.num_tags)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = loss_fct(active_logits, active_labels)
        return loss
    
    def get_loss_crf(self, logits_ner, labels, attention_mask):
        logits_padded, labels_padded = self.crf.bert_output2crf_input(logits_ner, labels)
        loss = self.crf.get_crf_loss(logits_padded, labels_padded)
        return loss

    def predict_ner_greedy(self, logits_ner, labels):
        logits_ner = logits_ner.softmax(dim=2)
        score_tags, pred_tags = logits_ner.max(dim=2)
        return -1.0, pred_tags, score_tags, labels
    
    def predict_ner_viterbi(self, logits_ner, labels):
        logits_padded, labels_padded = self.crf.bert_output2crf_input(logits_ner, labels)
        mask = labels_padded.ne(self.crf.tag_pad_id)
        score_sentence, pred_tags, score_tags = self.crf.viterbi_decode(logits_padded, mask)
        return score_sentence, pred_tags, score_tags, labels_padded

def get_optimizer(model, config, t_total):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
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

        inputs = bert_data_util.create_batch(dev_data, batch_ids, is_cuda)
        outputs_t = model(**inputs)
        loss_t, scores_ner_t = outputs_t[:2]

        eval_loss += loss_t.item()
        max_value_tag_t, pred_tag_t = torch.max(scores_ner_t, dim=2)

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

def train(output_root_dir, train_data_seq, dev_data_seq):
    tag_vocab = dataset_utils.get_tag_vocab(config)
    tag2id = {t: i for i, t in enumerate(tag_vocab)}

    train_data = bert_data_util.get_bert_data(train_data_seq, tag2id, config)
    dev_data = bert_data_util.get_bert_data(dev_data_seq, tag2id, config)

    model = MyBertForTokenClassification(num_tags=len(tag2id))
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
            inputs = bert_data_util.create_batch(train_data, batch_ids, is_cuda)
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
        results = eval_util.evaluate(preds_list, gold_label_list, tag_vocab)
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

def get_model(model_dir, num_tag):
    model = MyBertForTokenClassification(num_tags=num_tag)

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

def process_train(root_dir, data_dir):
    output_root_dir = os.path.join(root_dir, f'dl_model_{int(time.time())}')
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    print(f'model out dir {output_root_dir}', flush=True)

    train_data_seq = dataset_utils.get_data_train_dev(data_dir, 'train.jsonl', config)
    dev_data_seq = dataset_utils.get_data_train_dev(data_dir, 'dev.jsonl', config)

    train(output_root_dir, train_data_seq, dev_data_seq)

def process_eval(root_dir, model_dir, data_dir, filename):
    tag_vocab = dataset_utils.get_tag_vocab(config)
    tag2id = {t: i for i, t in enumerate(tag_vocab)}

    model = get_model(model_dir, len(tag2id))

    test_data, metadata = dataset_utils.get_data_test(data_dir, filename)
    test_data_bert = bert_data_util.get_bert_data(test_data, tag2id, config)

    preds_list, _, _ = predictions(test_data_bert, model, config)
    eval_util.dump_result(preds_list, metadata, tag_vocab, test_data, root_dir, 'boundary_model.txt')

if __name__ == "__main__":
    root_dir = os.path.join(os.path.expanduser("~"), 'private-projects/propganda/exp')
    data_dir = os.path.join(os.path.expanduser("~"), 'private-projects/propganda/datasets')
    #process_train(root_dir, data_dir)
    model_dir = os.path.join(os.path.expanduser("~"), 'private-projects/propganda/exp/dl_model_1579925959')
    process_eval(root_dir, model_dir, data_dir, 'test_phase0.jsonl')

