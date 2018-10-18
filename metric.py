from __future__ import division

import sys
import time
import logging
import StringIO
from collections import defaultdict, Counter, OrderedDict
import numpy as np
from numpy import array, zeros, allclose

class ConfusionMatrix(object):
    def __init__(self, labels, default_label=None):
        self.labels = labels
        self.default_label = default_label if default_label is not None else len(labels) -1
        self.counts = defaultdict(Counter)

    @staticmethod
    def to_table(data, row_labels, column_labels, precision=2, digits=4):
        # Convert data to strings
        data = [["%04.2f" % v for v in row] for row in data]
        cell_width = max(
            max(map(len, row_labels)),
            max(map(len, column_labels)),
            max(max(map(len, row)) for row in data))

        def c(s):
            """adjust cell output"""
            return s + " " * (cell_width - len(s))

        ret = ""
        ret += "\t".join(map(c, column_labels)) + "\n"
        for l, row in zip(row_labels, data):
            ret += "\t".join(map(c, [l] + row)) + "\n"
        return ret

    def as_table(self):
        data = [[self.counts[l][l_] for l_,_ in enumerate(self.labels)] for l,_ in enumerate(self.labels)]
        return ConfusionMatrix.to_table(data, self.labels, ["go\\gu"] + self.labels)

    def update(self, gold, guess):
        self.counts[gold][guess] += 1

    def summary(self, quiet=False):
        keys = range(len(self.labels))
        data = []
        macro = array([0., 0., 0., 0.])
        micro = array([0., 0., 0., 0.])
        default = array([0., 0., 0., 0.])
        for l in keys:
            tp = self.counts[l][l]
            fp = sum(self.counts[l_][l] for l_ in keys if l_ != l)
            tn = sum(self.counts[l_][l__] for l_ in keys if l_ != l for l__ in keys if l__ != l)
            fn = sum(self.counts[l][l_] for l_ in keys if l_ != l)

            acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
            prec = (tp)/(tp + fp) if tp > 0  else 0
            rec = (tp)/(tp + fn) if tp > 0  else 0
            f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0

            # update micro/macro averages
            micro += array([tp, fp, tn, fn])
            macro += array([acc, prec, rec, f1])
            if l != self.default_label: # Count count for everything that is not the default label!
                default += array([tp, fp, tn, fn])

            data.append([acc, prec, rec, f1])

        # micro average
        tp, fp, tn, fn = micro
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])
        # Macro average
        data.append(macro / len(keys))

        # default average
        tp, fp, tn, fn = default
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])

        # Macro and micro average.
        return ConfusionMatrix.to_table(data, self.labels + ["micro","macro","not-O"], ["label", "acc", "prec", "rec", "f1"])


def get_chunks(seq, default=LBLS.index(NONE)):
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        # End of a chunk + start of a chunk!
        elif tok != default:
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

def evaluate(self, sess, examples, examples_raw):
    token_cm = ConfusionMatrix(labels=LBLS)

    correct_preds, total_correct, total_preds = 0., 0., 0.
    for _, labels, labels_ in self.output(sess, examples_raw, examples):
        for l, l_ in zip(labels, labels_):
            token_cm.update(l, l_)
        gold = set(get_chunks(labels))
        pred = set(get_chunks(labels_))
        correct_preds += len(gold.intersection(pred))
        total_preds += len(pred)
        total_correct += len(gold)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return token_cm, (p, r, f1)