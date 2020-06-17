from __future__ import absolute_import, division, print_function
import glob
import os.path
import codecs
from pathlib import Path
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
import json
import re

import const

def tokenize(w):
    q = w
    contr_dict = {"’": "'",
                  "i\'m": "i am",
                  "won\'t": " will not",
                  "\'s": " s",
                  "\'ll": " will",
                  "\'ve": " have",
                  "n\'t": " not",
                  "\'re": " are",
                  "\'d": " would",
                  "y'all": " all of you"}

    for contr in contr_dict:
        q = q.replace(contr, contr_dict[contr])

    q_arr = re.findall(r"[\w]+[']*[\w]+|[\w]+|[.,!?;:]", q, re.UNICODE)

    q = ' '.join(q_arr)
    q = re.sub('[0-9]{5,}', '#####', q)
    q = re.sub('[0-9]{4}', '####', q)
    q = re.sub('[0-9]{3}', '###', q)
    q = re.sub('[0-9]{2}', '##', q)

    q = q.strip().lower().split()
    if len(q) == 0:
        return [w]
    return q

def tagid2tag_seq(tag_vocab, tagid_seq):
    return [[tag_vocab[t] for t in tag_seq] for tag_seq in tagid_seq]

def get_tokenize_tag(tagging_type, t, n):
    if n == 1:
        return [t]
    if t == const.ENTITY_OTHER or t.startswith(const.ENTITY_CONT) or tagging_type == 'B':
        return [t] * n

    name = t[2:]
    start = t[:2]
    tags = []
    if start == const.ENTITY_BEGIN:
        tags.append(t)
        for i in range(1, n):
            tags.append(const.ENTITY_CONT + name)
        return tags

    if tagging_type == "BIOES":
        if start == const.ENTITY_SINGLE:
            tags.append(t)
            for i in range(1, n-1):
                tags.append(const.ENTITY_CONT + name)
            tags.append(const.ENTITY_END + name)
        elif start == const.ENTITY_END:
            for i in range(n-1):
                tags.append(const.ENTITY_CONT + name)
            tags.append(t)

        return tags

def get_tag_vocab(config):
    tags = [const.ENTITY_OTHER]
    if config.tagging_type == 'B':
        tags.append(const.ENTITY_BEGIN)
    else:
        tags.extend([const.ENTITY_BEGIN + t for t in const.ENTITY_NAMES])
        tags.extend([const.ENTITY_CONT + t for t in const.ENTITY_NAMES])

        if config.tagging_type == "BIOES":
            tags.extend([const.ENTITY_END + t for t in const.ENTITY_NAMES])
            tags.extend([const.ENTITY_SINGLE + t for t in const.ENTITY_NAMES])

    return tags

def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles = []
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split(".")[0][7:]
        with codecs.open(filename, "r", encoding="utf8") as f:
            articles.append((article_id, f.read()))
    return articles


def parse_label(label_path):
    labels = []
    f = Path(label_path)

    if not f.exists():
        return labels

    for line in open(label_path):
        parts = line.strip().split('\t')
        labels.append({'start': int(parts[2]), 'end': int(parts[3]), 'type': parts[1]})

    labels.sort(key=lambda s: (s['start'], -s['end']))
    return labels

def clean_text(article):
    sentences = article.split('\n')
    end = -1
    res = []
    for sentence in sentences:
        start = end + 1
        end = start + len(sentence)  # length of sequence
        if sentence != "":  # if not empty line
            res.append({'start': start, 'end': end, 'sentence': sentence})
    return res

def get_overlapping_entities(entities):
    etree = defaultdict(set)
    # inefficeint but clean
    for a in range(len(entities)):
        ea = entities[a]
        for b in range(a + 1, len(entities)):
            eb = entities[b]
            overlap_start = max(ea['start'], eb['start'])
            overlap_end = min(ea['end'], eb['end'])
            if overlap_start <= overlap_end:
                # if eb['end'] > ea['end']:
                #    print('partial', ea, eb)
                etree[a].add(b)
                etree[b].add(a)
    assert all([(a in etree) for k in etree for a in etree[k]])
    # assert all([len(etree[k]) == 1 for k in etree])
    return etree

def get_per_sentence_entity(entities, ds, de):
    d_entities = []
    for a in range(len(entities)):
        ea = entities[a]
        overlap_start = max(ea['start'], ds)
        overlap_end = min(ea['end'], de)
        if overlap_start <= overlap_end:
            d_entities.append({
                'type': ea['type'],
                'start': overlap_start - ds,
                'end': overlap_end - ds
            })
    d_entities.sort(key=lambda s: (s['start'], -s['end']))
    return d_entities


def get_non_overlapping_seq(etree, n):
    # get non overlapping entity sequence
    ex = set()
    if len(etree) > 0:
        for k in etree:
            ex.add(tuple([i for i in range(n) if i not in etree[k]]))
    else:
        ex.add(tuple(range(n)))
    return ex


def get_tag_seq(n, name, tagging_type):
    tags = []
    for i in range(n):
        tag = None
        if tagging_type == 'IOBES':
            if i == 0:
                if n == 1:
                    tag = const.ENTITY_SINGLE
                else:
                    tag = const.ENTITY_BEGIN
            elif i == n - 1:
                tag = const.ENTITY_END
            else:
                tag = const.ENTITY_CONT
        elif tagging_type == 'IOB':
            if i == 0:
                tag = const.ENTITY_BEGIN
            else:
                tag = const.ENTITY_CONT
        elif tagging_type == 'B':
            tag = const.ENTITY_BEGIN
        else:
            raise Exception('tagging_type no recognized')
        assert tag is not None
        if tagging_type == 'B':
            tags.append(tag)
        else:
            tags.append(tag + name)

    return tags

def encode_tokens_json(e, sentence, d_entities):
    tokens = []
    labels = []
    curr = 0
    for a in sorted(e):
        ea = d_entities[a]
        pre = sentence[curr:ea['start']].split()
        span = sentence[ea['start']:ea['end']].split()

        tokens.extend(pre)
        start = len(tokens)
        tokens.extend(span)
        end = len(tokens)
        labels.append({ 'type': ea['type'],
                        'start': start,
                        'end' : end})
        curr = ea['end']

    pre = sentence[curr:].split()
    tokens.extend(pre)

    return {'tokens': tokens,
            'labels': labels}

def get_data_train_dev(root_dir, filename, config):
    examples = []
    for line in codecs.open(os.path.join(root_dir, filename), "r", encoding="utf8"):
        datum = json.loads(line)
        orig_tokens = datum['tokens']
        assert len(orig_tokens) > 0, line
        orig_tags = [const.ENTITY_OTHER]* len(datum['tokens'])
        for e in datum['labels']:
            orig_tags[e['start']:e['end']] = get_tag_seq(e['end'] - e['start'], e['type'], config.tagging_type)

        tokens = []
        tags = []
        for i, w in enumerate(orig_tokens):
            w_i = tokenize(w)
            t_i = orig_tags[i]
            tokens.extend(w_i)
            tokenized_tags = get_tokenize_tag(config.tagging_type, t_i, len(w_i))
            tags.extend(tokenized_tags)

            assert len(w_i) == len(tokenized_tags), f'{w_i} => {tokenized_tags}'
        assert len(tokens) == len(tags) and len(tokens) > 0, f'{tokens} => {tags}'
        examples.append((tokens, tags))
    return examples


def get_data_test(root_dir, filename):
    examples = []
    metadata = []

    for line in codecs.open(os.path.join(root_dir, filename), "r", encoding="utf8"):
        datum = json.loads(line)
        orig_tokens = datum['tokens']
        tokens = []
        ignore_mapping = []
        data_tokens = []
        for i, w in enumerate(orig_tokens):
            w_i = tokenize(w)
            tokens.extend(w_i)
            ignore = [0]*len(w_i)
            ignore[0] = 1
            ignore_mapping.extend(ignore)
            data_tokens.extend([w]*len(w_i))

        tags = [const.ENTITY_OTHER] * len(tokens)
        examples.append((tokens, tags))
        metadata.append({'article_id': datum['article_id'],
                         'start_sentence': datum['start_sentence'],
                         'end_sentence': datum['end_sentence'],
                         'ignore_mapping': ignore_mapping,
                         'data_tokens': data_tokens
                         })
    return examples, metadata


def dump_data(root_dir):
    train_data = read_articles_from_file_list(os.path.join(root_dir, 'train-articles'))
    label_dir = os.path.join(root_dir, 'train-labels-task2-technique-classification')

    examples = []
    for article_id, line in train_data:
        entities = parse_label(os.path.join(label_dir, f'article{article_id}.task2-TC.labels'))

        for d in clean_text(line):
            d_entities = get_per_sentence_entity(entities, d['start'], d['end'])
            etree = get_overlapping_entities(d_entities)
            ex = get_non_overlapping_seq(etree, len(d_entities))

            sentence = d['sentence']
            for e in ex:
                datum = encode_tokens_json(e, sentence, d_entities)
                datum['article_id'] = article_id
                examples.append(json.dumps(datum))

    train_data, dev_data = train_test_split(examples, test_size=0.2)
    with codecs.open(os.path.join(root_dir, 'train.jsonl'), "w", encoding="utf8") as f:
        f.write('\n'.join(train_data) + '\n')
    with codecs.open(os.path.join(root_dir, 'dev.jsonl'), "w", encoding="utf8") as f:
        f.write('\n'.join(dev_data) + '\n')

def dump_data_test(root_dir):
    examples = []
    dev_data = read_articles_from_file_list(os.path.join(root_dir, 'dev-articles'))

    for article_id, line in dev_data:
        for d in clean_text(line):
            datum = {'tokens':d['sentence'].split(),
                             'start_sentence': d['start'],
                             'end_sentence': d['end'],
                             'article_id':article_id}
            examples.append(json.dumps(datum))

    with codecs.open(os.path.join(root_dir, 'test_phase0.jsonl'), "w", encoding="utf8") as f:
        f.write('\n'.join(examples) + '\n')

if __name__ == "__main__":
    data_dir = os.path.join(os.path.expanduser("~"), 'prop/datasets')
    dump_data(data_dir)
    dump_data_test(data_dir)
