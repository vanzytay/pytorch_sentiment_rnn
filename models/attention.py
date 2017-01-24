import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

class AttentionLayer(nn.Module):
    """Implements an Attention Layer"""

    def __init__(self, args, layer_size):
        super(AttentionLayer, self).__init__()
        self.layer_size = layer_size
        self.args = args
        self.weight_W = nn.Parameter(torch.Tensor(layer_size,layer_size))
        self.bias = nn.Parameter(torch.Tensor(layer_size))
        self.weight_proj = nn.Parameter(torch.Tensor(layer_size, 1))
        self.softmax = nn.Softmax()
        self.weight_W.data.uniform_(-0.1, 0.1)
        self.weight_proj.data.uniform_(-0.1,0.1)

    def forward(self, inputs, attention_width=3):
        results = None
        for i in range(inputs.size(0)):
            if(i<attention_width):
                output = inputs[i]
                output = output.unsqueeze(0)
            else:
                lb = i - attention_width
                if(lb<0):
                    lb = 0
                selector = torch.from_numpy(np.array(np.arange(lb, i)))
                selector = Variable(selector)
                if(self.args.cuda):
                    selector = selector.cuda()
                vec = torch.index_select(inputs, 0, selector)
                u = batch_matmul(vec, self.weight_W, nonlinearity='tanh')
                a = batch_matmul(u, self.weight_proj)
                a = self.softmax(a)
                output = None
                for i in range(vec.size(0)):
                    h_i = vec[i]
                    a_i = a[i].unsqueeze(1).expand_as(h_i)
                    h_i = a_i * h_i
                    h_i = h_i.unsqueeze(0)
                    if(output is None):
                        output = h_i
                    else:
                        output = torch.cat((output,h_i),0)
               # print(output.size())
                output = torch.sum(output,0)
                # print(output.size())
            if(results is None):
                results = output
            else:
                results = torch.cat((results,output),0)
            # print(results.size())
        return results

