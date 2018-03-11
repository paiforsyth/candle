import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import serialmodule


VERY_NEGATIVE=-1000000

MaxPoolLSTMConfigure=collections.namedtuple("MaxPoolLSTMConfigure","hidden_dim")


class MaxPoolLSTM(serialmodule.SerializableModule):
    '''
        input:
        x: dimensions batchsize bt max sequence length
        pad mat: dimensions batchsize by max sequence length by 1
    '''
    @staticmethod
    def from_configure(embedding, configure):
        embedding_dim=embedding.weight.shape[1]
        lstrim=nn.LSTM(embedding_dim,configure.hidden_dim,num_layers=1,bidirectional=True, batch_first=True )
        return MaxPoolLSTM(embedding, lstm)

    def __init__(self,embedding, lstm ):
        super().__init__()
        self.embedding=embedding
        self.lstm=lstm
    def forward(self,x,pad_mat):
        out=self.embedding(x) #batchsize by max sequence length by embedding dim
        out,_=self.lstm(out) #batchsize by max_sequence length by hidden dim
        #import pdb; pdb.set_trace()
        out=out+VERY_NEGATIVE*pad_mat #broadcast
        out, _=torch.max(out,dim=1) #batchsize by hidden_dim
        return out

        
MaxPoolLSTMFCConfigure=collections.namedtuple("MaxPoolLSTMFCConfigure","hidden_dim, dropout_rate, out_dim")
class MaxPoolLSTMFC(serialmodule.SerializableModule): 
    '''
        FC = fully connected
    '''
    def __init__(self,embedding, lstm, lin, dropout):
        super().__init__()
        self.max_pool_lstm=MaxPoolLSTM(embedding,lstm)
        self.lin=lin
        self.dropout=dropout

    
    def forward(self,x,pad_mat):
        out=self.max_pool_lstm(x,pad_mat)
        out=self.lin(out)
        out=F.leaky_relu(out)
        out=self.dropout(out)
        return out
    
    @staticmethod
    def from_configure(embedding, configure):
        embedding_dim=embedding.weight.shape[1]
        lstm=nn.LSTM(embedding_dim,configure.hidden_dim,1, bidirectional=True, batch_first=True)
        lin=nn.Linear(2*configure.hidden_dim, configure.out_dim, bias=False)
        dropout=nn.Dropout(configure.dropout_rate)
        return MaxPoolLSTMFC(embedding,lstm,lin,dropout) 

    @staticmethod
    def from_args(embedding, args):
        configure=MaxPoolLSTMFCConfigure(hidden_dim=args.lstm_hidden_dim, dropout_rate=args.maxlstm_dropout_rate, out_dim=2)
        return MaxPoolLSTMFC.from_configure(embedding, configure)
