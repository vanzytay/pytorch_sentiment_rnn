import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class TargetRNN(nn.Module):
    """Target-dependent Recurrent Neural Network
    Takes a left context and right context and merges them (via concat)
    Passes through a linear transform and a softmax activation for sentiment classification
    """

    def __init__(self, args, vocab_size, pretrained=None):
        super(TargetRNN, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.encoder = nn.Embedding(self.vocab_size, self.args.embedding_size)
        self.left_rnn = getattr(nn, self.args.rnn_type)(self.args.embedding_size, self.args.rnn_size, self.args.rnn_layers, 
                            bias=True)
        self.right_rnn = getattr(nn, self.args.rnn_type)(self.args.embedding_size, self.args.rnn_size, self.args.rnn_layers, 
                            bias=True)
        self.decoder = nn.Linear(self.args.rnn_size * 2, 3)
        self.softmax = nn.Softmax()
        self.init_weights(pretrained=pretrained)
        print("Initialized {} model".format(self.args.rnn_type))

    def init_weights(self, pretrained):
        initrange = 0.1
        if(pretrained is not None):
            print("Setting Pretrained Embeddings")
            pretrained = pretrained.astype(np.float32)
            pretrained = torch.from_numpy(pretrained)
            if(self.args.cuda):
                pretrained = pretrained.cuda()
            self.encoder.weight.data = pretrained
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, left_input, right_input, left_hidden, right_hidden):
        left_emb = self.encoder(left_input)
        right_emb = self.encoder(right_input)
  
        left_output, left_hidden = self.left_rnn(left_emb, left_hidden)
        right_output, right_hidden = self.right_rnn(right_emb, right_hidden)  

        left_last = Variable(torch.LongTensor([left_output.size()[0]-1]))
        right_last = Variable(torch.LongTensor([right_output.size()[0]-1]))

        if(self.args.cuda):
            left_last = left_last.cuda()
            right_last = right_last.cuda()

        left_output = torch.index_select(left_output,0,left_last)
        left_output = torch.squeeze(left_output)
        right_output = torch.index_select(right_output,0,right_last)
        right_output = torch.squeeze(right_output)
        output = torch.cat((left_output, right_output), 1)   # this is seq_len right?
        decoded = self.decoder(output)
        decoded = self.softmax(decoded)
 
        return decoded

    def init_hidden(self, bsz):
    
        weight = next(self.parameters()).data
        # print(weight)
        if (self.args.rnn_type == 'LSTM'):
            return (Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_()),
                    Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_()))
        else:
            return Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_())
