from __future__ import absolute_import, division, print_function

from seqeval.metrics import precision_score, recall_score, f1_score

import const
import os
import dataset_utils

# eval
def get_chunks(seq, ignore_I_mismatch=False):
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        if tok == const.ENTITY_OTHER:
            if chunk_type is not None:
                chunks.append((chunk_type, chunk_start, i))
                chunk_type, chunk_start = None, None
        else:
            curr_chunk_type = tok[2:]
            chunk_prefix = tok[:2]
            if chunk_prefix == const.ENTITY_BEGIN:
                if chunk_type is not None:
                    chunks.append((chunk_type, chunk_start, i))

                chunk_type, chunk_start = curr_chunk_type, i
            elif chunk_prefix == const.ENTITY_CONT and not ignore_I_mismatch:
                if chunk_type is not None and chunk_type != curr_chunk_type:
                    chunks.append((chunk_type, chunk_start, i))
                    chunk_type, chunk_start = None, None
    # end condition
    if chunk_type is not None:
        chunks.append((chunk_type, chunk_start, len(seq)))

    chunks = list(set(chunks))
    chunks.sort(key=lambda s: s[0])
    return chunks

def evaluate(gold_label_list, preds_list, config):
    tag_vocab = dataset_utils.get_tag_vocab(config)
    gold_label_list = dataset_utils.tagid2tag_seq(tag_vocab, gold_label_list)
    preds_list = dataset_utils.tagid2tag_seq(tag_vocab, preds_list)

    results = {
        "precision": precision_score(gold_label_list, preds_list),
        "recall": recall_score(gold_label_list, preds_list),
        "f1": f1_score(gold_label_list, preds_list)
    }
    return results


def dump_result(preds_list, metadata, test_data, root_dir, filename, config):
    tag_vocab = dataset_utils.get_tag_vocab(config)
    preds_list = dataset_utils.tagid2tag_seq(tag_vocab, preds_list)

    tc = os.path.join(root_dir, 'tc_' + filename)
    si = os.path.join(root_dir, 'si_' + filename)

    with open(tc, "w") as tc_writer, open(si, "w") as si_writer:
        for i, t in enumerate(preds_list):
            article = metadata[i]
            sen_start = article['start_sentence']
            article_id = article['article_id']
            ignore_mapping = article['ignore_mapping']
            data_tokens = article['data_tokens']

            for type, start, end in get_chunks(t):
                #adjust start end
                orig_tokens = [data_tokens[i] for i in range(start) if ignore_mapping[i] == 1]
                start_boundary = len(' '.join(orig_tokens)) + sen_start
                orig_tokens = [data_tokens[i] for i in range(start, end) if ignore_mapping[i] == 1]
                end_boundary = start_boundary + len(' '.join(orig_tokens))
                si_writer.write(f'{article_id}\t{start_boundary}\t{end_boundary}\n')
                tc_writer.write(f'{article_id}\t{type}\t{start_boundary}\t{end_boundary}\n')

