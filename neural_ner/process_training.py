import sys
import time
import torch

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from config import config
from model import get_model
from conll2003_batcher import DatasetConll2003
from train_utils import setup_train_dir, save_model, write_summary, get_param_norm, get_grad_norm
from model import test_one_batch
from metric import Evaluter

import logging

logging.basicConfig(level=logging.INFO)

use_cuda = torch.cuda.is_available()
torch.manual_seed(1)

class Processor(object):
    def __init__(self, model_file_path):
        self.dataset = DatasetConll2003(config)
        self.model = get_model(self.dataset, config, use_cuda, model_file_path)

    def train_one_batch(self, batch, optimizer, params):
        self.model.train()
        optimizer.zero_grad()
        (s, s_lengths), y = batch.word, batch.ner
        logits = self.model(s, s_lengths)
        loss = self.model.get_loss(logits, y, s_lengths)

        loss.backward()

        param_norm = get_param_norm(params)
        grad_norm = get_grad_norm(params)

        clip_grad_norm_(params, config.max_grad_norm)
        optimizer.step()

        return loss.item(), param_norm, grad_norm

    def train(self):
        train_dir, model_dir, bestmodel_dir, summary_writer = setup_train_dir(config)

        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = Adam(params, lr=config.lr, amsgrad=True)

        num_params = sum(p.numel() for p in params)
        logging.info("Number of params: %d" % num_params)

        exp_loss, best_dev_f1 = None, None

        batch_iterator = self.dataset.get_train_iterator()

        logging.info("Beginning training loop...")
        epoch_tic = time.time()
        pre_epoch = 0
        for batch in batch_iterator:
            epoch, global_step = batch_iterator.epoch, batch_iterator.iterations

            if epoch >= config.num_epochs:
                break

            iter_tic = time.time()

            train_loss, param_norm, grad_norm = self.train_one_batch(batch, optimizer, params)
            write_summary(train_loss, "train/loss", summary_writer, global_step)

            iter_toc = time.time()
            iter_time = iter_toc - iter_tic

            exp_loss = 0.99 * exp_loss + 0.01 * train_loss if exp_loss else train_loss

            if global_step % config.print_every == 0:
                logging.info(
                    'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                    (epoch, global_step, train_loss, exp_loss, grad_norm, param_norm, iter_time))

            if pre_epoch < epoch:
                epoch_toc = time.time()
                logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc - epoch_tic))
                dev_f1 = self.evalute_test_dev(summary_writer, epoch, global_step, exp_loss)

                if best_dev_f1 is None or dev_f1 > best_dev_f1:
                    if best_dev_f1:
                        logging.info("Saving to %s..." % bestmodel_dir)
                        save_model(self.model, optimizer, exp_loss, global_step, epoch, bestmodel_dir)

                    best_dev_f1 = dev_f1

                sys.stdout.flush()
                pre_epoch = epoch

    def evalute_test_dev(self, summary_writer, epoch, global_step, exp_loss):
        logging.info("Calculating train loss...")
        _, train_acc, train_p, train_r, train_f1 = self.evaluate(DatasetConll2003.DATA_TYPE_TRAIN, num_samples=1000)
        logging.info("Epoch %d, Iter %d, Train loss: %f, accuracy, precision, recall, F1: %f, %f, %f, %f" % (
            epoch, global_step, exp_loss, train_acc, train_p, train_r, train_f1))
        write_summary(train_p, "train/P", summary_writer, global_step)
        write_summary(train_r, "train/r", summary_writer, global_step)
        write_summary(train_f1, "train/F1", summary_writer, global_step)

        logging.info("Calculating dev loss...")
        dev_loss, dev_acc, dev_p, dev_r, dev_f1 = self.evaluate(DatasetConll2003.DATA_TYPE_VAL)
        logging.info("Epoch %d, Iter %d, Dev loss: %f, accuracy, precision, recall, F1: %f, %f, %f, %f" % (
            epoch, global_step, dev_loss, train_acc, dev_p, dev_r, dev_f1))
        write_summary(dev_p, "dev/P", summary_writer, global_step)
        write_summary(dev_r, "dev/r", summary_writer, global_step)
        write_summary(dev_f1, "dev/F1", summary_writer, global_step)

        return dev_f1

    def inference(self, str):
        pass

    def evaluate_test(self):
        logging.info("Calculating test loss...")
        test_loss, test_acc, test_p, test_r, test_f1 = self.evaluate(DatasetConll2003.DATA_TYPE_TEST)
        logging.info("Test loss: %f, accuracy, precision, recall, F1: %f, %f, %f, %f" %
                     (test_loss, test_acc, test_p, test_r, test_f1))

    def evaluate(self, data_type, num_samples=None):
        tic = time.time()
        loss_per_batch = 0
        total_num_examples = 0

        batch_iterator = self.dataset.get_data_iterator(data_type)
        ev = Evaluter(self.dataset)

        for batch in batch_iterator:
            (s, s_lengths), y = batch.word, batch.ner
            logits, pred = test_one_batch(s, s_lengths, self.model)
            loss = self.model.neg_log_likelihood(logits, y, s_lengths)

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

if __name__ == "__main__":
    mode = sys.argv[1]
    model_file_path = None
    if len(sys.argv) > 2:
        model_file_path = sys.argv[2]

    processor = Processor(model_file_path)
    if mode == 'train':
        processor.train()
    elif mode == 'test':
        processor.evaluate_test()
    elif mode == 'inference':
        str = sys.argv[3]
        processor.inference(str)


