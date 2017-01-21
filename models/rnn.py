import torch
import torch.nn as nn
from torch.autograd import Variable

class BasicRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, args, vocab_size):
        super(BasicRNN, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.encoder = nn.Embedding(self.vocab_size, self.args.embedding_size)
        self.rnn = getattr(nn, self.args.rnn_type)(self.args.embedding_size, self.args.rnn_size, self.args.rnn_layers, 
                            bias=False, batch_first=True)
        self.decoder = nn.Linear(self.args.rnn_size, 3)
        self.softmax = nn.Softmax()
        self.init_weights()
        print("Initialized Basic RNN model")

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        # print("Forwarding...")
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        last = Variable(torch.LongTensor([output.size()[1]-1]))
        if(self.args.cuda):
            last = last.cuda()
        output = torch.index_select(output,1,last)
        output = torch.squeeze(output)
        decoded = self.decoder(output)
        decoded = self.softmax(decoded)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.args.rnn_type == 'LSTM':
            return (Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_()),
                    Variable(weight.new(self.args.rnn_layers, self.args.rnn_size).zero_()))
        else:
            return Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_())
