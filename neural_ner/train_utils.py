import torch
import time
import os
import tensorflow as tf
import json
import numpy as np
import codecs

from data_utils.tag_scheme_utils import iobes_iob

def setup_train_dir(config):
    eval_dir = os.path.join(config.log_root, 'eval')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    bestmodel_dir = os.path.join(train_dir, 'bestmodel')
    if not os.path.exists(bestmodel_dir):
        os.makedirs(bestmodel_dir)

    summary_writer = tf.summary.FileWriter(train_dir)

    with open(os.path.join(train_dir, "flags.json"), 'w') as fout:
        json.dump(vars(config), fout)

    return train_dir, summary_writer

def get_param_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def save_model(model, optimizer, loss, global_step, epoch, model_dir):
    model_state = model.state_dict()
    model_state = {k: v for k, v in model_state.items() if 'embedding' not in k}

    state = {
        'global_step': global_step,
        'epoch': epoch,
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'current_loss': loss
    }
    model_save_path = os.path.join(model_dir, 'model_%d_%d_%d' % (global_step, epoch, int(time.time())))
    torch.save(state, model_save_path)

def write_summary(value, tag, summary_writer, global_step):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)

class Evaluter(object):
    def __init__(self, vocab, label_type):
        self.vocab = vocab
        self.n_tags = len(self.vocab.id_to_tag)
        self.predictions = []
        self.count = np.zeros((self.n_tags, self.n_tags), dtype=np.int32)
        self.label_type = label_type

    def batch_update(self, batch, pred):
        s_lengths = batch['words_lens']
        y = batch['tags']
        raw_sentence = batch['raw_sentence']
        y_reals = y.cpu().data.numpy()
        y_preds = pred.cpu().data.numpy()
        for i, s_len in enumerate(s_lengths):
            r_tags = []
            p_tags = []
            for j in range(s_len):
                r_tags.append(self.vocab.id_to_tag[y_reals[i][j]])
                p_tags.append(self.vocab.id_to_tag[y_preds[i][j]])
                self.count[y_reals[i][j], y_preds[i][j]] += 1

            if self.label_type == 'IOBES':
                p_tags = iobes_iob(p_tags)
                r_tags = iobes_iob(r_tags)
            for j in range(s_len):
                new_line = " ".join([raw_sentence[i][j], r_tags[j], p_tags[j]])
                self.predictions.append(new_line)
            self.predictions.append("")

    def get_metric(self, log_dir, is_cf=False):
        # Write predictions to disk and run CoNLL script externally
        eval_id = np.random.randint(1000000, 2000000)
        output_path = os.path.join(log_dir, "eval.%i.output" % eval_id)
        scores_path = os.path.join(log_dir , "eval.%i.scores" % eval_id)
        with codecs.open(output_path, 'w', 'utf8') as f:
            f.write("\n".join(self.predictions))
        eval_script = '../conlleval.pl'
        os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

        # CoNLL evaluation results
        eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
        if is_cf:
            for line in eval_lines:
                print (line)

            # Confusion matrix with accuracy for each tag
            format_str = "{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * self.n_tags)
            values = [self.vocab.id_to_tag[i] for i in range(self.n_tags)]
            print(format_str.format("ID", "NE", "Total", *values, "Percent"))

            for i in range(self.n_tags):
                percent = "{:.3f}".format(self.count[i][i] * 100. / max(1, self.count[i].sum()))
                values = [self.count[i][j] for j in range(self.n_tags)]
                print (format_str.format(str(i), self.vocab.id_to_tag[i], str(self.count[i].sum()), *values, percent))

            # Global accuracy
            print ("{}/{} ({:.3f} %)" .format (
                self.count.trace(), self.count.sum(), 100. * self.count.trace() / max(1, self.count.sum())
            ))

        # F1 on all entities
        return float(eval_lines[1].strip().split()[-1])

