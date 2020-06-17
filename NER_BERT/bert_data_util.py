from __future__ import absolute_import, division, print_function

import os

from transformers import BertTokenizer

import torch
import torch.nn as nn

import const

bert_tokenizer = BertTokenizer.from_pretrained(const.MODEL_TYPE, do_lower_case=True)

def get_bert_data(examples, tag2id, config):
    bert_data = []
    for orig_tokens, orig_tag in examples:
        input_ids, label_ids, segment_ids, tokens = prepare_bert_input(orig_tokens, orig_tag, tag2id, config)
        bert_data.append((input_ids, label_ids, segment_ids))
    return bert_data

def prepare_bert_input(orig_tokens, orig_tag, tag2id, config):
    tokens = []
    label_ids = []

    assert len(orig_tag) == len(orig_tokens), orig_tag + orig_tokens

    for i, t in enumerate(orig_tokens):
        label_t = tag2id[orig_tag[i]]
        bert_tokens = bert_tokenizer.tokenize(t)
        bert_tokens_len = len(bert_tokens)
        if bert_tokens_len > 0:
            tokens.extend(bert_tokens)
            label_ids.append(label_t)

        # pad label if multiple tokens for a single word
        if bert_tokens_len > 1:
            label_ids.extend([const.label_pad_id] * (bert_tokens_len - 1))

    assert len(tokens) == len(label_ids)
    ###truncate large sequence###
    tokens = tokens[:config.max_seq_length]
    label_ids = label_ids[:config.max_seq_length]
    ##############################
    segment_ids = [const.sequence_a_segment_id] * len(tokens)

    tokens = [const.cls_token] + tokens + [const.sep_token]

    label_ids = [const.label_pad_id] + label_ids + [const.label_pad_id]
    segment_ids = [const.cls_token_segment_id] + segment_ids + [const.sequence_a_segment_id]
    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

    assert len(input_ids) == len(label_ids)
    assert len(input_ids) == len(segment_ids)

    input_ids = torch.tensor(input_ids).long()
    label_ids = torch.tensor(label_ids).long()
    segment_ids = torch.tensor(segment_ids).long()
    return input_ids, label_ids, segment_ids, tokens

def create_batch(train_data, batch_ids, is_cuda):
    max_len = max([len(train_data[bi][0]) for bi in batch_ids])
    batch_input_ids = []
    batch_label_ids = []
    batch_segment_ids = []
    for bi in batch_ids:
        input_ids, label_ids, segment_ids = train_data[bi]
        pad_len = max_len - len(input_ids)
        padding_op = nn.ConstantPad1d((0, pad_len), const.pad_token_id)
        batch_input_ids.append(padding_op(input_ids).unsqueeze(0))
        padding_op = nn.ConstantPad1d((0, pad_len), const.label_pad_id)
        batch_label_ids.append(padding_op(label_ids).unsqueeze(0))
        padding_op = nn.ConstantPad1d((0, pad_len), const.pad_token_segment_id)
        batch_segment_ids.append(padding_op(segment_ids).unsqueeze(0))

    batch_input_ids = torch.cat(batch_input_ids)
    batch_label_ids = torch.cat(batch_label_ids)
    batch_segment_ids = torch.cat(batch_segment_ids)

    att_mask = batch_input_ids.ne(const.pad_token_id)

    if is_cuda:
        batch_input_ids = batch_input_ids.cuda()
        batch_label_ids = batch_label_ids.cuda()
        batch_segment_ids = batch_segment_ids.cuda()
        att_mask = att_mask.cuda()

    inputs = {'input_ids': batch_input_ids,
              'attention_mask': att_mask,
              'token_type_ids': batch_segment_ids,
              'labels': batch_label_ids}
    return inputs

