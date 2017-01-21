from __future__ import division
from utilities import *
import argparse
import json
import gzip
from models.rnn import BasicRNN
import cPickle as pickle
from datetime import datetime
import os
import random
import numpy as np
import time
import sys
import math
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    # What's this?
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class BaseExperiment:
    ''' Implements a base experiment class for TensorFLow
    '''
    def __init__(self):
        self.dataset = 'Restaurants'
        self.mode = 'aspect'
        self.uuid = datetime.now().strftime("%d_%H:%M:%S")
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--mdl", dest="model_type", type=str, metavar='<str>', default='normal', help="Model type (reg|regp|breg|bregp) (default=regp)")
        self.parser.add_argument("--rnn_type", dest="rnn_type", type=str, metavar='<str>', default='LSTM', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
        self.parser.add_argument("--term_mdl", dest="term_model", type=str, metavar='<str>', default='mean', help="Model type for term sequences (default=mean)")
        self.parser.add_argument("--opt", dest="opt", type=str, metavar='<str>', default='Adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
        self.parser.add_argument("--emb_size", dest="embedding_size", type=int, metavar='<int>', default=300, help="Embeddings dimension (default=50)")
        self.parser.add_argument("--rnn_size", dest="rnn_size", type=int, metavar='<int>', default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
        self.parser.add_argument("--batch-size", dest="batch_size", type=int, metavar='<int>', default=128, help="Batch size (default=256)")
        self.parser.add_argument("--rnn_layers", dest="rnn_layers", type=int, metavar='<int>', default=1, help="Number of RNN layers")
        self.parser.add_argument("--rnn_direction", dest="rnn_direction", type=str, metavar='<str>', default='uni', help="Direction of RNN")
        self.parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='mot', help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
        self.parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability. To disable, give a negative number (default=0.5)")
        self.parser.add_argument("--pretrained", dest="pretrained", type=int, metavar='<int>', default=1, help="Whether to use pretrained or not")
        self.parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=50, help="Number of epochs (default=50)")
        self.parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
        self.parser.add_argument('--gpu', dest='gpu', type=int, metavar='<int>', default=0, help="Specify which GPU to use (default=0)")
        self.parser.add_argument("--hdim", dest='hidden_layer_size', type=int, metavar='<int>', default=300, help="Hidden layer size (default=50)")
        self.parser.add_argument("--lr", dest='learn_rate', type=float, metavar='<float>', default=0.001, help="Learning Rate")
        self.parser.add_argument("--clip_norm", dest='clip_norm', type=int, metavar='<int>', default=1, help="Clip Norm value")
        self.parser.add_argument("--trainable", dest='trainable', type=int, metavar='<int>', default=1, help="Trainable Word Embeddings (0|1)")
        self.parser.add_argument('--l2_reg', dest='l2_reg', type=float, metavar='<float>', default=0.0, help='L2 regularization, default=0')
        self.parser.add_argument('--eval', dest='eval', type=int, metavar='<int>', default=1, help='Epoch to evaluate results')
        self.parser.add_argument('--log', dest='log', type=int, metavar='<int>', default=1, help='1 to output to file and 0 otherwise')
        self.parser.add_argument('--dev', dest='dev', type=int, metavar='<int>', default=1, help='1 for development set 0 to train-all')
        self.parser.add_argument('--cuda', action='store_true', help='use CUDA')
        self.parser.add_argument('--seed', type=int, default=1111, help='random seed')
        self.parser.add_argument('--debug', type=str, default='', help='debug')
        self.args = self.parser.parse_args()
        # Set the random seed manually for reproducibility.
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(self.args.seed)

        # Load Data files for training
        with open('./store/{}_{}_{}.pkl'.format(self.dataset, self.mode, self.args.debug),'r') as f:
            self.env = pickle.load(f)

        self.train_set = self.env['train']
        self.test_set = self.env['test']
        self.dev_set = self.env['dev']

        if(self.args.dev==0):
            self.train_set = self.train_set + self.dev_set

        print("Loaded environment")
        print("Creating Model...")
        self.model_name = self.args.aggregation + '_' + self.args.model_type
        self.mdl = BasicRNN(self.args, len(self.env['word_index']))
        if(self.args.cuda):
            self.mdl.cuda()
        # self.mdl = AspectModel(len(self.env['word_index']), self.args, cat_index=self.env['cat_index'])
        # self.make_dir()

    def make_dir(self):
        if(self.args.log==1):
            self.out_dir = './logs/{}/{}/{}/{}/'.format(self.mode, self.dataset, self.model_name, self.uuid)
            self.mkdir_p(self.out_dir)
            self.mdl_path = self.out_dir + '/mdl.ckpt'  # What is the new file format?
            self.path = self.out_dir + '/logs.txt'
            self.print_args(self.args, path=self.path)

    def write_to_file(self, txt):
        if(self.args.log==1):
            with open(self.path,'a+') as f:
                f.write(txt + '\n')
        print(txt)

    def print_args(self, args, path=None):
        if path:
            output_file = open(path, 'w')
        args.command = ' '.join(sys.argv)
        items = vars(args)
        output_file.write('=============================================== \n')
        for key in sorted(items.keys(), key=lambda s: s.lower()):
            value = items[key]
            if not value:
                value = "None"
            if path is not None:
                output_file.write("  " + key + ": " + str(items[key]) + "\n")
        output_file.write('=============================================== \n')
        if path:
            output_file.close()
        del args.command
    
    def mkdir_p(self, path):
        if path == '':
            return
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

    def evaluate(self, batch, eval_type='test'):
        # TODO : Implement Evaluate Function
        pass

    def load_embeddings(self):
        # TODO: Implement Preload Glove Embeddings
        pass

    def make_batch(self, x, i):
        batch = x[int(i * self.args.batch_size):int(i * self.args.batch_size)+self.args.batch_size]
        if(len(batch)==0):
            return None,None
        sentence = torch.LongTensor(np.array([x[0] for x in batch]).tolist())
        targets = torch.LongTensor(np.array([x[3] for x in batch], dtype=np.int32).tolist())
        if(self.args.cuda):
            sentence = sentence.cuda()
            targets = targets.cuda()  
        sentence = Variable(sentence)
        targets = Variable(targets)      
        return sentence, targets

    def train(self):
        print("Starting training")
        self.criterion = nn.CrossEntropyLoss()
        print(self.args)
        total_loss = 0
        num_batches = int(len(self.train_set) / self.args.batch_size) + 1
        self.optimizer =  optim.SGD(self.mdl.parameters(), lr=self.args.learn_rate)
        for epoch in tqdm(range(1,self.args.epochs)):
            losses = []
            hidden = self.mdl.init_hidden(self.args.batch_size)
            for i in range(num_batches):
                sentence, targets = self.make_batch(self.train_set, i)
                if(sentence is None):
                    continue
                hidden = repackage_hidden(hidden)
                self.mdl.zero_grad()
                output, hidden = self.mdl(sentence, hidden)
                loss = self.criterion(output, targets)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.data[0])

                # # Need to figure out how to Clip Gradient with Optimizer 
                # # Waiting for feature? 
                # clipped_lr = self.args.learn_rate * clip_gradient(self.mdl, self.args.clip_norm)
                # for p in self.mdl.parameters():
                #     p.data.add_(-clipped_lr, p.grad.data)
            print("[Epoch {}] Loss={}".format(epoch, np.mean(losses)))


if __name__ == '__main__':
    exp = BaseExperiment()
    exp.train()



