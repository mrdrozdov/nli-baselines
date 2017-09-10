#!/bin/python3

import os
import sys
import json
import random
import itertools
import time

import gflags
import numpy as np
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from blocks import get_l2_loss, save, pack_checkpoint, ckpt_path
from data import preprocess_data, default_vocab, PADDING_TOKEN
from utils import MainLogger


FLAGS = gflags.FLAGS


def args():

    default_experiment_name = "exp-{}".format(int(time.time()))

    gflags.DEFINE_string("data_path", os.path.expanduser("~/data/multinli_0.9/multinli_0.9_dev_matched.jsonl"), "Path to NLI data.")
    gflags.DEFINE_string("eval_data_path", os.path.expanduser("~/data/multinli_0.9/multinli_0.9_dev_matched.jsonl"), "Path to NLI data.")
    gflags.DEFINE_string("embedding_path", os.path.expanduser("~/data/glove.6B.50d.txt"), "Path to GloVe vectors.")
    gflags.DEFINE_string("save_path", ".", "Path to logs and checkpoints.")
    gflags.DEFINE_string("load_path", None, "Path to load checkpoint.")
    gflags.DEFINE_string("log_path", None, "Path to log.")
    gflags.DEFINE_string("experiment_name", default_experiment_name, "Experiment name.")
    gflags.DEFINE_float("l2", None, "Use l2 regularization.")
    gflags.DEFINE_boolean("extract", False, "Use pretrained model to calculate query and target vectors for input data.")
    gflags.DEFINE_integer("seed", 11, "Random seed.")

    FLAGS(sys.argv)

    if not FLAGS.load_path:
        FLAGS.load_path = FLAGS.save_path  # this way we use logs/ckpt for an experiment_name if it exists.

    if not FLAGS.log_path:
        FLAGS.log_path = os.path.join('.', FLAGS.experiment_name + '.log')

    logger = MainLogger().init(path=FLAGS.log_path)
    logger.Log(json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))


def run():
    model = GRU()
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    data_paths = [FLAGS.data_path, FLAGS.eval_data_path]

    datasets, embeddings = preprocess_data(data_paths, FLAGS.embedding_path)

    training_data = Dataset(datasets[0])
    eval_data = Dataset(datasets[1], shuffle=False, truncate_final_batch=True)

    if FLAGS.extract:
        raise NotImplementedError

    def eval_fn():
        return evaluate(model, eval_data, embeddings)

    def ckpt_fn(ckpt_args):
        step = ckpt_args['step']
        best_dev_error = ckpt_args['best_dev_error']
        best = ckpt_args.get('best', False)
        save_dict = pack_checkpoint(step, best_dev_error, model, optimizer)
        filename = ckpt_path(FLAGS.save_path, filename=FLAGS.experiment_name, best=best)

        print("Checkpointing...")
        save(save_dict, filename)

    train_args = dict(step=0, best_dev_error=1.0)

    train(model, optimizer, training_data, embeddings, eval_fn, ckpt_fn, train_args)


class Dataset(object):
    indexes = None
    batch_index = None

    def __init__(self, dataset, shuffle=True, truncate_final_batch=False):
        self.shuffle = shuffle
        self.dataset = dataset
        self.truncate_final_batch = truncate_final_batch

    def reset(self):
        self.indexes = list(range(len(self.dataset)))
        self.batch_index = 0
        if self.shuffle:
            random.shuffle(self.indexes)

    def next_batch(self, batch_size):
        if not self.indexes:
            self.reset()

        if self.batch_index + batch_size > len(self.dataset):
            if self.truncate_final_batch and self.batch_index < len(self.dataset):
                batch_size = len(self.dataset) - self.batch_index
            else:
                self.reset()

        batch_index = self.batch_index
        indexes = self.indexes[batch_index:batch_index+batch_size]
        examples = [self.dataset[i] for i in indexes]
        self.batch_index += batch_size

        tokens1 = list(map(lambda x: x.tokens1, examples))
        tokens2 = list(map(lambda x: x.tokens2, examples))
        tokens = tokens1 + tokens2
        max_length = max(map(len, tokens))

        # Pad
        for i, tt in enumerate(tokens):
            if len(tt) < max_length:
                tokens[i] = tt + [default_vocab[PADDING_TOKEN]] * (max_length - len(tt))

        labels = list(map(lambda x: x.label, examples))

        return tokens, labels


def evaluate(model, eval_data, embeddings):
    batch_size = 10

    n_correct = 0.
    n_total = 0
    training = False
    eval_data.reset()
    num_batches = len(eval_data.dataset) // batch_size
    model.eval()
    pbar = tqdm(total=num_batches, desc='eval')
    for _ in range(num_batches):
        tokens, label = eval_data.next_batch(batch_size)
        tokens_np = np.array(tokens, dtype=np.int32)
        input = embeddings.take(tokens_np, axis=0)

        x = Variable(torch.from_numpy(input).float(), volatile=not training)
        outp = model(x)

        pred = outp.data.max(1)[1]
        target = torch.LongTensor(label)
        correct = pred.eq(target).sum()

        n_correct += correct
        n_total += pred.size(0)

        pbar.update(1)
    pbar.close()
    avg_acc = n_correct / n_total

    print("eval avg_acc={} [{}/{}]".format(avg_acc, n_correct, n_total))

    return 1 - avg_acc


def train(model, optimizer, training_data, embeddings, eval_fn, ckpt_fn, train_args):
    batch_size = 10
    eval_every = 20
    log_every = 20
    ckpt_step = 10000

    step = train_args.get('step', 0)
    best_dev_error = train_args.get('best_dev_error', 1.)

    training = True
    pbar = None
    for _ in itertools.repeat(None):

        if not pbar:
            pbar = tqdm(total=log_every, desc='train')

        model.train()
        tokens, label = training_data.next_batch(batch_size)
        tokens_np = np.array(tokens, dtype=np.int32)
        input = embeddings.take(tokens_np, axis=0)

        x = Variable(torch.from_numpy(input).float(), volatile=not training)
        outp = model(x)

        optimizer.zero_grad()
        target = Variable(torch.LongTensor(label), volatile=not training)
        total_loss = 0
        xent_loss = nn.NLLLoss()(outp, target)
        l2_loss = get_l2_loss(model, l2_lambda=2e-5) if FLAGS.l2 else Variable(torch.FloatTensor([0]))
        total_loss = xent_loss + l2_loss
        total_loss.backward()
        optimizer.step()

        pbar.update(1)

        if step > 0 and step % log_every == 0:
            pbar.close()
            pbar = None
            print("step={} total_loss={} xent_loss={} l2_loss={}".format(
                step, total_loss.data[0], xent_loss.data[0], l2_loss.data[0]))

        if step > 0 and step % eval_every == 0:
            dev_error = eval_fn()
            if dev_error < best_dev_error and step >= ckpt_step:
                best_dev_error = dev_error
                ckpt_fn(dict(step=step, best_dev_error=best_dev_error, best=True))

        step += 1
    pbar.close()


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        self.input_dim = 50
        self.hidden_dim = 50
        self.num_layers = 1
        
        # self.transform = nn.Linear(50, 50)
        self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(2 * self.hidden_dim * self.num_layers, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 3)
        self.dropout_rate = 0.2

        for w in self.parameters():
            if len(w.size()) == 2:
                nn.init.kaiming_uniform(w.data, mode='fan_in')
            else:
                w.data.fill_(0.)

    def forward(self, x):
        batch_size = x.size(0) // 2
        seq_length = x.size(1)

        h0 = Variable(torch.zeros(batch_size, self.num_layers, self.hidden_dim))
        outp, hn = self.rnn(x, h0)

        hn = hn.transpose(0, 1).view(2 * batch_size, self.num_layers * self.hidden_dim)

        xq = hn[:batch_size]
        xt = hn[batch_size:]

        xh = torch.cat([xq, xt], 1)
        xh = F.dropout(xh, self.dropout_rate, training=self.training)
        xh = F.relu(self.fc1(xh))
        xh = F.dropout(xh, self.dropout_rate, training=self.training)
        xh = F.relu(self.fc2(xh))
        xh = F.dropout(xh, self.dropout_rate, training=self.training)
        xh = self.fc3(xh)
        y = F.log_softmax(xh)
        return y


if __name__ == '__main__':
    args()

    random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    run()
