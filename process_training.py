import io
import json
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf
import torch

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from config import config
from model import NCRF

logging.basicConfig(level=logging.INFO)

use_cuda = torch.cuda.is_available()

class Processor(object):
    def __init__(self):
       pass

    def train_one_batch(self, batch, model, optimizer, params):
        model.train()
        optimizer.zero_grad()
        q_seq, q_lens, d_seq, d_lens, span = self.get_data(batch)
        loss, _, _ = model(q_seq, q_lens, d_seq, d_lens, span)
	
        l2_reg = None
        for W in params:
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)
        loss = loss + config.reg_lambda * l2_reg

        loss.backward()

        param_norm = self.get_param_norm(params)
        grad_norm = self.get_grad_norm(params)

        clip_grad_norm_(params, config.max_grad_norm)
        optimizer.step()

        return loss.item(), param_norm, grad_norm

    def eval_one_batch(self, batch, model):
        model.eval()
        q_seq, q_lens, d_seq, d_lens, span = self.get_data(batch)
        loss, pred_start_pos, pred_end_pos = model(q_seq, q_lens, d_seq, d_lens, span)
        return loss.item(), pred_start_pos.data, pred_end_pos.data

    def test_one_batch(self, batch, model):
        model.eval()
        q_seq, q_lens, d_seq, d_lens = self.get_data(batch, is_train=False)
        pred_start_pos, pred_end_pos = model(q_seq, q_lens, d_seq, d_lens)
        return pred_start_pos.data, pred_end_pos.data

    def train(self, model_file_path):
        train_dir, model_dir, bestmodel_dir, summary_writer = setup_train_dir(config)

        model = get_model(model_file_path)
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(params, lr=config.lr, amsgrad=True)

        num_params = sum(p.numel() for p in params)
        logging.info("Number of params: %d" % num_params)

        exp_loss, best_dev_f1 = None, None

        epoch = 0
        global_step = 0

        logging.info("Beginning training loop...")
        while config.num_epochs == -1 or epoch < config.num_epochs:
            epoch += 1
            epoch_tic = time.time()
            for batch in sequence_batcher:
                global_step += 1
                iter_tic = time.time()

                loss, param_norm, grad_norm = self.train_one_batch(batch, model, optimizer, params)
                write_summary(loss, "train/loss", summary_writer, global_step)

                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                exp_loss = 0.99 * exp_loss + 0.01 * loss if exp_loss else loss

                if global_step % config.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))


                if global_step % config.save_every == 0:
                    logging.info("Saving to %s..." % model_dir)
                    save_model(model, optimizer, loss, global_step, epoch, model_dir)

                if global_step % config.eval_every == 0:
                    dev_loss = self.get_dev_loss(model)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)

                    train_f1, train_em = self.check_f1_em(model, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (
                        epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)

                    dev_f1, dev_em = self.check_f1_em(model, "dev", num_samples=0)
                    logging.info(
                        "Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)

                    if best_dev_f1 is None or dev_f1 > best_dev_f1:
                        best_dev_f1 = dev_f1

                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_dir)
                        self.save_model(model, optimizer, loss, global_step, epoch, bestmodel_dir)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc - epoch_tic))

        sys.stdout.flush()

    def check_f1(self, model, dataset, num_samples=100, print_to_screen=False):
        logging.info("Calculating F1 for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        if dataset == "train":
            context_path, qn_path, ans_path = self.train_context_path, self.train_qn_path, self.train_ans_path
        elif dataset == "dev":
            context_path, qn_path, ans_path = self.dev_context_path, self.dev_qn_path, self.dev_ans_path
        else:
            raise ('dataset is not defined')

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, config.batch_size,
                                         context_len=config.context_len, question_len=config.question_len,
                                         discard_long=False):

            pred_start_pos, pred_end_pos = self.test_one_batch(batch, model)

            pred_start_pos = pred_start_pos.tolist()
            pred_end_pos = pred_end_pos.tolist()

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) \
                    in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                true_answer = " ".join(true_ans_tokens)

                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx],
                                  batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start,
                                  pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total

    def get_dev_loss(self, model):
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []
        i = 0
        for batch in get_batch_generator(self.word2id, self.dev_context_path, self.dev_qn_path, self.dev_ans_path,
                                         config.batch_size, context_len=config.context_len,
                                         question_len=config.question_len, discard_long=True):

            loss, _, _ = self.eval_one_batch(batch, model)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)
            i += 1
            if i == 10:
                break
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss



if __name__ == "__main__":
    mode = sys.argv[1]
    processor = Processor()
    if mode == "train":
        model_file_path = None
        if len(sys.argv) > 2:
            model_file_path = sys.argv[2]
        processor.train(model_file_path)
    elif mode == "show_examples":
        model_file_path = sys.argv[2]
        model = processor.get_model(model_file_path)
        processor.check_f1_em(model, num_samples=10, print_to_screen=True)

    elif mode == "official_eval":
        model_file_path = sys.argv[2]
        json_in_path = ""
        json_out_path = ""
        qn_uuid_data, context_token_data, qn_token_data = get_json_data(json_in_path)
        model = processor.get_model(model_file_path)
        answers_dict = generate_answers(config, model, processor, qn_uuid_data, context_token_data, qn_token_data)
        print "Writing predictions to %s..." % json_out_path
        with io.open(json_out_path, 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
            print "Wrote predictions to %s" % json_out_path
