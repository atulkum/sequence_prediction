import logging
import sys
import os
import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD

from data_utils.batcher import DatasetConll2003
from data_utils.vocab import Vocab
from model import get_model, test_one_batch
from train_utils import setup_train_dir, save_model, write_summary, \
    get_param_norm, get_grad_norm, Evaluter

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1)

class Processor(object):
    def __init__(self, config, model_file_path):
        self.config = config
        self.vocab = Vocab(config)
        self.model = get_model(self.vocab, config, model_file_path)

    def train_one_batch(self, batch, optimizer, params):
        self.model.train()
        optimizer.zero_grad()
        s_lengths = batch['words_lens']
        y = batch['tags']
        logits = self.model(batch)
        loss = self.model.get_loss(logits, y, s_lengths)

        loss.backward()

        param_norm = get_param_norm(params)
        grad_norm = get_grad_norm(params)

        clip_grad_norm_(params, self.config.max_grad_norm)
        optimizer.step()

        return loss.item(), param_norm, grad_norm

    def train(self):
        train_dir, summary_writer = setup_train_dir(self.config)

        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = Adam(params, lr=0.001, amsgrad=True)

        num_params = sum(p.numel() for p in params)
        logging.info("Number of params: %d" % num_params)

        exp_loss, best_dev_f1 = None, None

        batch_iterator = DatasetConll2003(DatasetConll2003.DATA_TYPE_TRAIN, self.config, self.vocab, is_train=True)

        logging.info("Beginning training loop...")
        epoch_tic = time.time()
        pre_epoch = 0
        for batch in batch_iterator:
            epoch, global_step = batch_iterator.epoch, batch_iterator.iterations

            iter_tic = time.time()

            train_loss, param_norm, grad_norm = self.train_one_batch(batch, optimizer, params)
            write_summary(train_loss, "train/loss", summary_writer, global_step)

            iter_toc = time.time()
            iter_time = iter_toc - iter_tic

            exp_loss = 0.99 * exp_loss + 0.01 * train_loss if exp_loss else train_loss

            if global_step % self.config.print_every == 0:
                logging.info(
                    'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                    (epoch, global_step, train_loss, exp_loss, grad_norm, param_norm, iter_time))

            if pre_epoch < epoch:
                epoch_toc = time.time()
                logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc - epoch_tic))
                dev_f1 = self.evalute_test_dev(summary_writer, epoch, global_step, exp_loss)

                if best_dev_f1 is None or dev_f1 > best_dev_f1:
                    if best_dev_f1:
                        bestmodel_dir = os.path.join(train_dir, 'bestmodel')
                        logging.info("Saving to %s..." % bestmodel_dir)
                        save_model(self.model, optimizer, exp_loss, global_step, epoch, bestmodel_dir)

                    best_dev_f1 = dev_f1

                sys.stdout.flush()
                pre_epoch = epoch

            if epoch >= self.config.num_epochs:
                break

    def evalute_test_dev(self, summary_writer, epoch, global_step, exp_loss):
        _, train_f1 = self.evaluate(DatasetConll2003.DATA_TYPE_TRAIN, num_samples=1000)
        logging.info("Train: Epoch %d, Iter %d, loss: %f, F1: %f" % (epoch, global_step, exp_loss, train_f1))
        write_summary(train_f1, "train/F1", summary_writer, global_step)

        dev_loss, dev_f1 = self.evaluate(DatasetConll2003.DATA_TYPE_EVAL)
        logging.info("Dev: Epoch %d, Iter %d, loss: %f, F1: %f" % (epoch, global_step, dev_loss, dev_f1))
        write_summary(dev_f1, "dev/F1", summary_writer, global_step)

        self.evaluate_test()

        return dev_f1

    def inference(self, str):
        pass

    def evaluate_test(self):
        logging.info("Calculating test loss...")
        test_loss, test_f1 = self.evaluate(DatasetConll2003.DATA_TYPE_TEST)
        logging.info("Test: loss: %f,  F1: %f" %(test_loss, test_f1))

    def evaluate(self, data_type, num_samples=None):
        tic = time.time()
        loss_per_batch = 0
        total_num_examples = 0

        batch_iterator = DatasetConll2003(data_type, self.config, self.vocab, is_train=False)
        ev = Evaluter(self.vocab, self.config.label_type)

        for batch in batch_iterator:
            s_lengths = batch['words_lens']
            y = batch['tags']

            logits, pred = test_one_batch(batch, self.model)
            loss = self.model.get_loss(logits, y, s_lengths)

            curr_batch_size = len(batch['raw_sentence'])
            loss_per_batch += loss * curr_batch_size
            total_num_examples += curr_batch_size

            ev.batch_update(batch, pred)

            if num_samples and total_num_examples > num_samples:
                break

        toc = time.time()
        logging.info("Computed inference over %i examples in %.2f seconds" % (total_num_examples, toc - tic))

        total_loss = loss_per_batch / float(total_num_examples)
        eval_dir = os.path.join(self.config.log_root, 'eval')
        f1 = ev.get_metric(eval_dir)
        return total_loss, f1

if __name__ == "__main__":
    from config import config
    config.is_cuda = config.is_cuda and torch.cuda.is_available()
    mode = sys.argv[1]
    model_file_path = None
    if len(sys.argv) > 2:
        model_file_path = sys.argv[2]

    processor = Processor(config, model_file_path)
    if mode == 'train':
        processor.train()
    elif mode == 'test':
        processor.evaluate_test()
    elif mode == 'inference':
        str = sys.argv[3]
        processor.inference(str)


