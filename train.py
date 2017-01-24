from __future__ import division
from utilities import *
import argparse
from sklearn.metrics import accuracy_score
import json
import gzip
from models.rnn import BasicRNN
from models.td_rnn import TargetRNN
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
from keras.preprocessing import sequence
from collections import Counter
import time

def tensor_to_numpy(x):
    ''' Need to cast before calling numpy()
    '''
    return x.data.type(torch.DoubleTensor).numpy()

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
    ''' Implements a base experiment class for Aspect-Based Sentiment Analysis on SemEval 2014
    '''
    def __init__(self):
        self.uuid = datetime.now().strftime("%d_%H:%M:%S")
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--mode", dest="mode", type=str, metavar='<str>', default='term', help="Experiment Mode (term|aspect) (default=term)")
        self.parser.add_argument("--dataset", dest="dataset", type=str, metavar='<str>', default='Restaurants', help="Dataset (Laptop/Restaurants) (default=Restaurants)")
        self.parser.add_argument("--mdl", dest="model_type", type=str, metavar='<str>', default='RNN', help="(RNN|TD-RNN|ATT-RNN)")
        self.parser.add_argument("--rnn_type", dest="rnn_type", type=str, metavar='<str>', default='LSTM', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
        self.parser.add_argument("--term_mdl", dest="term_model", type=str, metavar='<str>', default='mean', help="Model type for term sequences (default=mean)")
        self.parser.add_argument("--opt", dest="opt", type=str, metavar='<str>', default='Adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
        self.parser.add_argument("--emb_size", dest="embedding_size", type=int, metavar='<int>', default=300, help="Embeddings dimension (default=50)")
        self.parser.add_argument("--rnn_size", dest="rnn_size", type=int, metavar='<int>', default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
        self.parser.add_argument("--batch-size", dest="batch_size", type=int, metavar='<int>', default=20, help="Batch size (default=256)")
        self.parser.add_argument("--rnn_layers", dest="rnn_layers", type=int, metavar='<int>', default=1, help="Number of RNN layers")
        self.parser.add_argument("--rnn_direction", dest="rnn_direction", type=str, metavar='<str>', default='uni', help="Direction of RNN")
        self.parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='mean', help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
        self.parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability. To disable, give a negative number (default=0.5)")
        self.parser.add_argument("--pretrained", dest="pretrained", type=int, metavar='<int>', default=1, help="Whether to use pretrained or not")
        self.parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=50, help="Number of epochs (default=50)")
        self.parser.add_argument("--attention_width", dest="attention_width", type=int, metavar='<int>', default=5, help="Width of attention (default=5)")
        self.parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
        self.parser.add_argument('--gpu', dest='gpu', type=int, metavar='<int>', default=0, help="Specify which GPU to use (default=0)")
        self.parser.add_argument("--hdim", dest='hidden_layer_size', type=int, metavar='<int>', default=300, help="Hidden layer size (default=50)")
        self.parser.add_argument("--lr", dest='learn_rate', type=float, metavar='<float>', default=0.001, help="Learning Rate")
        self.parser.add_argument("--clip_norm", dest='clip_norm', type=int, metavar='<int>', default=0, help="Clip Norm value")
        self.parser.add_argument("--trainable", dest='trainable', type=int, metavar='<int>', default=1, help="Trainable Word Embeddings (0|1)")
        self.parser.add_argument('--l2_reg', dest='l2_reg', type=float, metavar='<float>', default=0.0, help='L2 regularization, default=0')
        self.parser.add_argument('--eval', dest='eval', type=int, metavar='<int>', default=1, help='Epoch to evaluate results')
        self.parser.add_argument('--log', dest='log', type=int, metavar='<int>', default=1, help='1 to output to file and 0 otherwise')
        self.parser.add_argument('--dev', dest='dev', type=int, metavar='<int>', default=1, help='1 for development set 0 to train-all')
        self.parser.add_argument('--cuda', action='store_true', help='use CUDA')
        self.parser.add_argument('--seed', type=int, default=1111, help='random seed')
        self.parser.add_argument('--toy', action='store_true', help='Use toy dataset (for fast testing)')
        self.args = self.parser.parse_args()
        # Set the random seed manually for reproducibility.
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                print("There are {} CUDA devices".format(torch.cuda.device_count()))
                if(self.args.gpu > 0):
                    print("Setting torch GPU to {}".format(self.args.gpu))
                    torch.cuda.set_device(self.args.gpu)
                    print("Using device:{} ".format(torch.cuda.current_device()))
                torch.cuda.manual_seed(self.args.seed)

        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        # Load Data files for training
        if(self.args.toy):
            file_path = './store/{}_{}_{}.pkl'.format(self.args.dataset, self.args.mode, 'toy')
        else:
            file_path = './store/{}_{}.pkl'.format(self.args.dataset, self.args.mode)

        with open(file_path,'r') as f:
            self.env = pickle.load(f)

        print('Stored Environment:{}'.format(self.env.keys()))
        self.train_set = self.env['train']
        self.test_set = self.env['test']
        self.dev_set = self.env['dev']

        if(self.args.dev==0):
            self.train_set = self.train_set + self.dev_set

        print("Loaded environment")
        print("Creating Model...")
        self.model_name = self.args.aggregation + '_' + self.args.model_type
        if(self.args.model_type=='TD-RNN'):
            self.mdl = TargetRNN(self.args, len(self.env['word_index']),pretrained=self.env['glove'])
        elif(self.args.model_type in ['RNN','ATT-RNN']):
            self.mdl = BasicRNN(self.args, len(self.env['word_index']),pretrained=self.env['glove'])
        if(self.args.cuda):
            self.mdl.cuda()

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

    def evaluate(self, x, eval_type='test'):
        ''' Evaluates normal RNN model
        '''
        hidden = self.mdl.init_hidden(len(x))
        sentence, targets, actual_batch = self.make_batch(x, -1, evaluation=True)
        output, hidden = self.mdl(sentence, hidden)
        loss = self.criterion(output, targets).data
        print("Test loss={}".format(loss[0]))
        accuracy = self.get_accuracy(output, targets)

    def evaluate_target(self, x, eval_type='test'):
        ''' Evaluates Target-RNN model
        '''
        sentence, targets, actual_batch = self.make_target_batch(x, -1, evaluation=True)
        left_input, right_input = sentence[0], sentence[1]
        if(sentence is None):
            return None
        left_hidden = self.mdl.init_hidden(actual_batch)
        right_hidden = self.mdl.init_hidden(actual_batch)
        left_hidden = repackage_hidden(left_hidden)
        right_hidden = repackage_hidden(right_hidden)
        output = self.mdl(left_input, right_input, left_hidden, right_hidden)
        loss = self.criterion(output, targets).data[0]
        print("Test loss={}".format(loss))
        accuracy = self.get_accuracy(output, targets)
       
    def get_accuracy(self, output, targets):
        output = tensor_to_numpy(output)
        targets = tensor_to_numpy(targets)
        output = np.argmax(output, axis=1)
        dist = dict(Counter(output))
        print("Output Distribution={}".format(dist))
        acc = accuracy_score(targets, output)
        print("Accuracy={}".format(acc))
        return acc

    def pad_to_batch_max(self, x):
        lengths = [len(y) for y in x]
        max_len = np.max(lengths)
        padded_tokens = sequence.pad_sequences(x, maxlen=max_len)
        return torch.LongTensor(padded_tokens.tolist()).transpose(0,1)

    def make_target_batch(self, x, i, evaluation=False):
        ''' target dependent batches
        '''
        if(i>=0):
            batch = x[int(i * self.args.batch_size):int(i * self.args.batch_size)+self.args.batch_size]
        else:
            batch = x
        if(len(batch)==0):
            return None, None, self.args.batch_size
        left_tensor = self.pad_to_batch_max([x['left'] for x in batch])
        right_tensor = self.pad_to_batch_max([x['right'] for x in batch][::-1])
        targets = torch.LongTensor(np.array([x['polarity'] for x in batch], dtype=np.int32).tolist())
        assert(left_tensor.size(1)==right_tensor.size(1))
        actual_batch = left_tensor.size(1)
        if(self.args.cuda):
            left_tensor = left_tensor.cuda()
            right_tensor = right_tensor.cuda()
            targets = targets.cuda()  
        left_tensor = Variable(left_tensor)
        right_tensor = Variable(right_tensor)
        targets = Variable(targets, volatile=evaluation)      
        return [left_tensor, right_tensor], targets, actual_batch

    def make_batch(self, x, i, evaluation=False):
        ''' -1 to take all
        '''
        if(i>=0):
            batch = x[int(i * self.args.batch_size):int(i * self.args.batch_size)+self.args.batch_size]
        else:
            
            batch = x
        if(len(batch)==0):
            return None,None, self.args.batch_size

        sentence = self.pad_to_batch_max([x['tokenized_txt'] for x in batch])
        targets = torch.LongTensor(np.array([x['polarity'] for x in batch], dtype=np.int32).tolist())
        actual_batch = sentence.size(1)
        if(self.args.cuda):
            sentence = sentence.cuda()
            targets = targets.cuda()  
        sentence = Variable(sentence)
        targets = Variable(targets, volatile=evaluation)      
        return sentence, targets, actual_batch

    def select_optimizer(self):
        if(self.args.opt=='Adam'):
            self.optimizer =  optim.Adam(self.mdl.parameters(), lr=self.args.learn_rate)
        elif(self.args.opt=='RMS'):
            self.optimizer =  optim.RMSprop(self.mdl.parameters(), lr=self.args.learn_rate)
        elif(self.args.opt=='SGD'):
            self.optimizer =  optim.SGD(self.mdl.parameters(), lr=self.args.learn_rate)
        elif(self.args.opt=='Adagrad'):
            self.optimizer =  optim.Adagrad(self.mdl.parameters(), lr=self.args.learn_rate)
        elif(self.args.opt=='Adadelta'):
            self.optimizer =  optim.Adadelta(self.mdl.parameters(), lr=self.args.learn_rate)

    def train_target_batch(self, i):
        ''' Trains a regular Target-Dependent RNN model
        '''
        sentence, targets, actual_batch = self.make_target_batch(self.train_set, i)
        left_input, right_input = sentence[0], sentence[1]
        if(sentence is None):
            return None

        # Do I need to init both? Can I just pass in 0 vectors?
        left_hidden = self.mdl.init_hidden(actual_batch)
        right_hidden = self.mdl.init_hidden(actual_batch)

        left_hidden = repackage_hidden(left_hidden)
        right_hidden = repackage_hidden(right_hidden)
        self.mdl.zero_grad()
        output = self.mdl(left_input, right_input, left_hidden, right_hidden)
        loss = self.criterion(output, targets)
        loss.backward()
        if(self.args.clip_norm>0):
            coeff = clip_gradient(self.mdl, self.args.clip_norm)
            for p in self.mdl.parameters():
                p.grad.mul_(coeff)
        self.optimizer.step()
        return loss.data[0]

    def train_batch(self, i):
        ''' Trains a regular RNN model
        '''
        sentence, targets, actual_batch = self.make_batch(self.train_set, i)
        if(sentence is None):
            return None
        hidden = self.mdl.init_hidden(actual_batch)
        hidden = repackage_hidden(hidden)
        self.mdl.zero_grad()
        output, hidden = self.mdl(sentence, hidden)
        loss = self.criterion(output, targets)
        loss.backward()
        if(self.args.clip_norm>0):
            coeff = clip_gradient(self.mdl, self.args.clip_norm)
            for p in self.mdl.parameters():
                p.grad.mul_(coeff)
        self.optimizer.step()
        return loss.data[0]

    def train(self):
        print("Starting training")
        self.criterion = nn.CrossEntropyLoss()
        print(self.args)
        total_loss = 0
        num_batches = int(len(self.train_set) / self.args.batch_size) + 1
        self.select_optimizer()
        for epoch in range(1,self.args.epochs+1):
            t0 = time.clock()
            random.shuffle(self.train_set)
            print("========================================================================")
            losses = []
            actual_batch = self.args.batch_size
            for i in range(num_batches):
                if(self.args.model_type in ['TD-RNN']):
                    loss = self.train_target_batch(i)
                else:
                    loss = self.train_batch(i)
                if(loss is None):
                    continue    
                losses.append(loss)
            t1 = time.clock()
            print("[Epoch {}] Train Loss={} T={}s".format(epoch, np.mean(losses),t1-t0))
            if(epoch >0 and epoch % self.args.eval==0):
                if(self.args.model_type in ['TD-RNN']):
                    self.evaluate_target(self.test_set)
                else:
                    self.evaluate(self.test_set)
               
            


if __name__ == '__main__':
    exp = BaseExperiment()
    exp.train()



