import time
from model import test_one_batch

from conll2003_batcher import DatasetConll2003
from collections import defaultdict, Counter
import numpy as np

class ConfusionMatrix(object):
    def __init__(self, labels, default_label):
        self.counts = defaultdict(Counter)
        self.labels = labels
        self.default_label = default_label

    @staticmethod
    def to_table(data, row_labels, column_labels):
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
        return ConfusionMatrix.to_table(data, self.labels, ["go\\pr"] + self.labels)

    def update(self, gold, guess):
        self.counts[gold][guess] += 1

    def summary(self):
        keys = range(len(self.labels))
        data = []
        macro = np.array([0., 0., 0., 0.])
        micro = np.array([0., 0., 0., 0.])
        default = np.array([0., 0., 0., 0.])
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
            micro += np.array([tp, fp, tn, fn])
            macro += np.array([acc, prec, rec, f1])
            if l != self.default_label: # Count count for everything that is not the default label!
                default += np.array([tp, fp, tn, fn])

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


class Evaluter(object):
    def __init__(self, dataset):
        self.token_cm = ConfusionMatrix(dataset.get_labels(), dataset.get_default_label())
        self.correct_preds, self.total_correct, self.total_preds = 0., 0., 0.
        self.right_tag = 0
        self.all_tag = 0
        self.dataset = dataset

    def batch_update(self, batch, pred_labelid):
        (_, s_lengths), y = batch.word, batch.ner
        golden_lists = y.data.numpy()
        predict_lists = pred_labelid.data.numpy()
        for i, s_len in enumerate(s_lengths):
            for j in range(s_len):
                self.token_cm.update(golden_lists[i][j], predict_lists[i][j])

            golden_list = self.dataset.label_ids2labels(golden_lists[i], s_len)
            predict_list = self.dataset.label_ids2labels(predict_lists[i], s_len)

            for idy in range(s_len):
                if golden_list[idy] == predict_list[idy]:
                        self.right_tag += 1
            self.all_tag += len(golden_list)

            if self.dataset.label_type == "BIOES":
                gold_matrix = DatasetConll2003.get_ner_BIOES(golden_list)
                pred_matrix = DatasetConll2003.get_ner_BIOES(predict_list)
            else:
                gold_matrix = DatasetConll2003.get_ner_BIO(golden_list)
                pred_matrix = DatasetConll2003.get_ner_BIO(predict_list)

            right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
            self.correct_preds += len(right_ner)
            self.total_preds += len(pred_matrix)
            self.total_correct += len(gold_matrix)

    def get_metric(self):
        precision = self.correct_preds / self.total_preds if self.correct_preds > 0 else 0
        recall = self.correct_preds / self.total_correct if self.correct_preds > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if self.correct_preds > 0 else 0

        accuracy = self.right_tag/self.all_tag
        print("gold_num = ", self.total_correct, " pred_num = ", self.total_preds, " right_num = ", self.correct_preds)
        return accuracy, precision, recall, f1

def evaluate(dataset, model, data_type, num_samples=None):
    tic = time.time()
    loss_per_batch = 0
    total_num_examples = 0

    batch_iterator = dataset.get_data_iterator(data_type)
    ev = Evaluter(dataset)

    for batch in batch_iterator:
        (s, s_lengths), y = batch.word, batch.ner
        logits, pred = test_one_batch(s, s_lengths, model)
        loss = model.get_loss(logits, y, s_lengths)

        curr_batch_size = batch.batch_size
        loss_per_batch += loss * curr_batch_size
        total_num_examples += curr_batch_size

        ev.batch_update(batch, pred)

        if num_samples and total_num_examples > num_samples:
            break

    toc = time.time()
    print("Computed inference over %i examples in %.2f seconds" % (total_num_examples, toc - tic))

    total_loss = loss_per_batch / float(total_num_examples)
    acc, p, r, f1 = ev.get_metric()
    print(ev.token_cm.as_table())
    return total_loss, acc, p, r, f1

if __name__ == '__main__':
    from config import config
    from conll2003_batcher import DatasetConll2003
    ds = DatasetConll2003(config)
    train_iter = ds.get_train_iterator()

    ev = Evaluter(ds)

    batch = next(iter(train_iter))
    (s, s_lengths), y = batch.word, batch.ner
    ev.batch_update(batch, y)

    batch = next(iter(train_iter))
    (s, s_lengths), y = batch.word, batch.ner
    ev.batch_update(batch, y)

    print(ev.get_metric())
    print(ev.token_cm.as_table())