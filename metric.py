import time
from model import test_one_batch

from conll2003_batcher import DatasetConll2003

class Evaluter(object):
    def __init__(self, dataset):
        self.correct_preds, self.total_correct, self.total_preds = 0., 0., 0.
        self.right_tag = 0
        self.all_tag = 0
        self.dataset = dataset

    def batch_update(self, batch, pred_labelid):
        (_, s_lengths), y = batch.word, batch.ner
        golden_lists = y.data.numpy()
        predict_lists = pred_labelid.data.numpy()
        for i, s_len in enumerate(s_lengths):
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