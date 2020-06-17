import torch
import torch.nn as nn
import const
import dataset_utils

def get_data(examples, vocab, config):
    tag_vocab = dataset_utils.get_tag_vocab(config)
    tag2id = {t: i for i, t in enumerate(tag_vocab)}

    all_data = []
    for orig_tokens, orig_tag in examples:
        input_ids, char_ids, label_ids = prepare_input(orig_tokens, orig_tag, tag2id, vocab, config)
        all_data.append((input_ids, char_ids, label_ids))
    return all_data

def create_char_batch(char_id_seq):
    batch_char_ids = []
    max_len = max([len(char_ids) for char_ids in char_id_seq])
    for char_ids in char_id_seq:
        pad_len = max_len - len(char_ids)
        padding_op = nn.ConstantPad1d((0, pad_len), const.pad_token_id)
        batch_char_ids.append(padding_op(char_ids).unsqueeze(0))

    batch_char_ids = torch.cat(batch_char_ids)
    mask = batch_char_ids.ne(const.pad_token_id)

    return batch_char_ids, mask

def prepare_input(orig_tokens, orig_tag, tag2id, vocab, config):
    input_ids = []
    label_ids = []
    char_id_seq = []

    assert len(orig_tag) == len(orig_tokens), orig_tag + orig_tokens

    for i, w in enumerate(orig_tokens):
        label_id = tag2id[orig_tag[i]]
        w_id = vocab['word2id'][w] if w in vocab['word2id'] else const.UNK_ID
        input_ids.append(w_id)
        label_ids.append(label_id)
        w_char_ids = [vocab['char2id'][c] if c in vocab['char2id'] else const.UNK_ID for c in w]
        w_char_ids = torch.tensor(w_char_ids).long()

        char_id_seq.append(w_char_ids)

    char_ids = create_char_batch(char_id_seq)
    input_ids = torch.tensor(input_ids).long()
    label_ids = torch.tensor(label_ids).long()
    return input_ids, char_ids, label_ids

def create_batch(train_data, batch_ids, is_cuda):
    max_len = max([len(train_data[bi][0]) for bi in batch_ids])
    batch_input_ids = []
    batch_char_ids = []
    batch_label_ids = []

    for bi in batch_ids:
        input_ids, char_ids, label_ids = train_data[bi]
        batch_char_ids.append(char_ids)
        pad_len = max_len - len(input_ids)
        padding_op = nn.ConstantPad1d((0, pad_len), const.pad_token_id)
        batch_input_ids.append(padding_op(input_ids).unsqueeze(0))
        padding_op = nn.ConstantPad1d((0, pad_len), const.label_pad_id)
        batch_label_ids.append(padding_op(label_ids).unsqueeze(0))

    batch_input_ids = torch.cat(batch_input_ids)
    batch_label_ids = torch.cat(batch_label_ids)

    att_mask = batch_input_ids.ne(const.pad_token_id)

    if is_cuda:
        batch_input_ids = batch_input_ids.cuda()
        batch_label_ids = batch_label_ids.cuda()
        att_mask = att_mask.cuda()

    inputs = {'word_ids': batch_input_ids,
              'mask': att_mask,
              'char_ids':batch_char_ids,
              'labels': batch_label_ids}
    return inputs
