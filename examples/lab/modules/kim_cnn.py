import collections
import torch
import torch.nn as nn
import argparse

import torch.nn.functional as F
from . import serialmodule


def add_args(parser):
    parser.add_argument("--kim_cnn_num_convs",type=int, default=100)
    parser.add_argument("--kim_cnn_dropout_rate",type=int, default=0.5)
    return parser

KimCNNConfig=collections.namedtuple("KimCNNConfig","num_convs, dropout_rate, out_dim")
#based on  https://github.com/castorini/Castor/tree/master/kim_cnn
class KimCNN(serialmodule.SerializableModule):
    @staticmethod
    def from_args(embedding,args):
        configure=KimCNNConfig(num_convs= args.kim_cnn_num_convs, dropout_rate=args.kim_cnn_dropout_rate, out_dim=2)
        return KimCNN(embedding, configure )
    
    def __init__(self,embedding, configure):
        super().__init__()
        embedding_dim=embedding.weight.shape[1]
        self.embedding=embedding
        self.conv1=nn.Conv2d(1, configure.num_convs,(3,embedding_dim), padding=(1,0))
        self.conv2=nn.Conv2d(1, configure.num_convs, (5,embedding_dim),padding=(2,0))
        
        self.dropout=nn.Dropout(configure.dropout_rate)

        self.lin=nn.Linear(2*configure.num_convs, configure.out_dim)

    def forward(self, x,pad_mat):
        '''
            Note: currentloy pad_mat is not used
        '''
        #x has dim batchsize by seq length
        out=self.embedding(x) #batchsize by sequence length by embedding_dim
        batchsize=out.shape[0]
        seq_length=out.shape[1]
        embedding_dim=out.shape[2]
        out=out.view(batchsize,1,seq_length, embedding_dim)
        out=[F.leaky_relu(self.conv1(out) ), F.leaky_relu(self.conv2(out)) ] #list of variables, each with dimension batchsize by num_convs by sequence_length by  1 (the convolutions fill the entire embedding dim) 
        out=[sec.squeeze(3) for sec in out] #remove the singleton dimension
        out=[F.max_pool1d(sec,sec.shape[2]).squeeze(2) for sec in out ] #maxout dimension #2 (sequence length)
        if len(out[0].shape)<2:
            import pdb; pdb.set_trace()
        out=torch.cat(out, dim=1) #join allong num_convs
        out=self.dropout(out)
        out=self.lin(out)
        return out
