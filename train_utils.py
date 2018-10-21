import torch
import time
import os
import tensorflow as tf
import json

def setup_train_dir(config):
    train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    model_dir = os.path.join(train_dir, 'model')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    bestmodel_dir = os.path.join(train_dir, 'bestmodel')
    if not os.path.exists(bestmodel_dir):
        os.makedirs(bestmodel_dir)

    summary_writer = tf.summary.FileWriter(train_dir)

    with open(os.path.join(train_dir, "flags.json"), 'w') as fout:
        json.dump(vars(config), fout)

    return train_dir, model_dir, bestmodel_dir, summary_writer

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
