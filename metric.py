from __future__ import division


from collections import defaultdict, Counter
import numpy as np

class ConfusionMatrix(object):
    def __init__(self, labels, default_label):
        self.counts = defaultdict(Counter)
        self.labels = labels
        self.default_label = default_label

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
        self.dataset = dataset

    def batch_update(self, batch, pred_labelid):
        (_, s_lengths), y = batch.word, batch.ner
        lables_true = y.data.numpy()
        lables_pred = pred_labelid.data.numpy()

        for i, s_len in enumerate(s_lengths):
            for j in range(s_len):
                self.token_cm.update(lables_true[i][j], lables_pred[i][j])

            labels = self.dataset.label_ids2labels(lables_true[i], s_len)
            labels_ = self.dataset.label_ids2labels(lables_pred[i], s_len)

            gold = set(self.dataset.get_chunks(labels))
            pred = set(self.dataset.get_chunks(labels_))
            self.correct_preds += len(gold.intersection(pred))
            self.total_preds += len(pred)
            self.total_correct += len(gold)

    def get_metric(self):
        p = self.correct_preds / self.total_preds if self.correct_preds > 0 else 0
        r = self.correct_preds / self.total_correct if self.correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if self.correct_preds > 0 else 0
        return p, r, f1

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
    print(ev.token_cm.summary())
