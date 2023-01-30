import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# ## Functions to accomplish attention
def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    bias_dim = bias.size()
    _s = torch.mm(seq, weight) 
    _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
    if(nonlinearity=='tanh'):
        _s_bias = torch.tanh(_s_bias)
    return _s_bias

def batch_matmul(seq, weight, nonlinearity=''):
    _s = torch.mm(seq, weight)
    if(nonlinearity=='tanh'):
        _s = torch.tanh(_s)
    return _s

def attention_mul(rnn_outputs, att_weights):
    h_i = rnn_outputs
    a_i = att_weights.expand_as(h_i)
    h_i = a_i * h_i
    return torch.sum(h_i, 0)


# ## attention model with bias
class AttentionRNN(nn.Module):
    def __init__(self, embed_size, gru_hidden, bidirectional= True, pooling_mode = 'attention_pooling', linear=False):
        
        super(AttentionRNN, self).__init__()
        
        self.embed_size = embed_size
        self.gru_hidden = gru_hidden
        self.bidirectional = bidirectional
        self.pooling_mode = pooling_mode
        self.linear = linear
        
        if linear:
            self.linear = nn.Linear(embed_size, self.gru_hidden)
            self.mapping = nn.Linear(self.gru_hidden, self.gru_hidden)
            self.weight_proj = nn.Linear(self.gru_hidden, 1, bias=False)
 
        elif bidirectional == True:
            self.gru = nn.GRU(embed_size, gru_hidden, bidirectional= True)
            self.mapping = nn.Linear(2*gru_hidden, 2*gru_hidden)
            self.weight_proj = nn.Linear(2*gru_hidden, 1, bias=False)
        else:
            self.gru = nn.GRU(embed_size, gru_hidden, bidirectional= False)
            self.weight_W = nn.Parameter(torch.Tensor(gru_hidden, gru_hidden))
            self.bias = nn.Parameter(torch.Tensor(gru_hidden,1))
            self.weight_proj = nn.Parameter(torch.Tensor(gru_hidden, 1))
            
        self.softmax = nn.Softmax(dim=2)

    def forward(self, embedded, d, sub2phn=False):
        # gru and 
        embedded = embedded.transpose(1,2)
        if self.linear:
            output = self.linear(embedded)
        else:
            output, _ = self.gru(embedded.transpose(0,1))
            output = output.transpose(0,1)
        output = self.pooling(output, d, sub2phn=sub2phn)
        #output = torch.zeros((d.shape[0],d.shape[1],self.gru_hidden)).to(d.device)
        return output.transpose(1,2)
 
    def pooling(self, hidden, mask, sub2phn=False):
        squish = torch.tanh(self.mapping(hidden))
        attn = self.weight_proj(squish)
        attn = attn.squeeze(2).unsqueeze(1).repeat(1,mask.size(1),1)
        
        #print(attn.shape, mask.shape)
        attn_norm = self.softmax(attn.masked_fill(mask,-np.inf))
        attn_norm = attn_norm.masked_fill(mask, 0.)
        attn_vectors = torch.bmm(attn_norm, hidden)

        return attn_vectors
 

if __name__ == "__main__":
    attn = AttentionRNN(embed_size=300, gru_hidden=100, bidirectional= True, pooling_mode='attention')
    x = torch.zeros(1, 20, 300)
    d = torch.LongTensor([[4,5,6,5]])
    print(x.shape, d.shape)
    _s= attn(x, d)
    print(_s.shape)
