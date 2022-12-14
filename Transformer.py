#=====================================#
#            {import Modules}         #
#=====================================#

import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt
import os
import random # random.seed
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2,1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    
    return torch.softmax(m , -1)


def attention(Q, K, V):
    a = a_norm(Q, K) #(batch_size, dim_attn, seq_length)
    
    return  torch.matmul(a,  V) #(batch_size, seq_length, seq_length)

class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
    
    def forward(self, x, kv = None):
        if(kv is None):
            #Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x))
        
        #Attention with x as Q, external vector kv as K an V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv))
    
class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))
        
        self.heads = nn.ModuleList(self.heads)
        
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias = True)
                      
        
    def forward(self, x, kv = None):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv))
            
        a = torch.stack(a, dim = -1) #combine heads
        a = a.flatten(start_dim = 2) #flatten all head outputs
        
        x = self.fc(a)
        
        return x
    
class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        
        self.fc1 = nn.Linear(dim_input, dim_val, bias = True)
        #self.fc2 = nn.Linear(5, dim_val)
    
    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        
        return x

class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = True)
        #self.fc2 = nn.Linear(5, dim_attn)
    
    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        
        return x

class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn
        
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = True)
        #self.fc2 = nn.Linear(5, dim_attn)
    
    def forward(self, x):
        
        x = self.fc1(x)
        #print(x.shape)
        #x = self.fc2(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]. squeeze(1)
        return x     
      
#=====================================#
#             {get_data}              #
#=====================================#

def get_data6(data,n_val, input_sequence_length, output_sequence_length):   
    inout_seq = []
    L = len(data)-input_sequence_length
    #n_val = int(n)
    for i in range(L):
        train_seq = data[i:i+input_sequence_length+output_sequence_length]
        inout_seq.append(train_seq)  
    s = torch.tensor(inout_seq)
    return s[:-n_val, :input_sequence_length].unsqueeze(-1),s[:-n_val,-output_sequence_length:],s[-n_val:, :input_sequence_length].unsqueeze(-1),s[-n_val:,-output_sequence_length:]    

  
#=====================================#
#           {Network file}            #
#=====================================#

class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn , n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
    
    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)
        
        a = self.fc1(F.relu(self.fc2(x)))
        x = self.norm2(x + a)
        
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)
        
    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)
        
        a = self.attn2(x, kv = enc)
        x = self.norm2(a + x)
        
        a = self.fc1(F.relu(self.fc2(x)))
        
        x = self.norm3(x + a)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len, out_seq_len, n_decoder_layers = 1, n_encoder_layers = 1, n_heads = 1):
        super(Transformer, self).__init__()
        self.dec_seq_len = dec_seq_len
        
        #Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads))
        
        self.decs = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads))
        
        self.pos = PositionalEncoding(dim_val)
        
        #Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)
    
    def forward(self, x):
        #encoder
        e = self.encs[0](self.pos(self.enc_input_fc(x)))
        for enc in self.encs[1:]:
            e = enc(e)
        
        #decoder
        d = self.decs[0](self.dec_input_fc(x[:,-self.dec_seq_len:]), e)
        for dec in self.decs[1:]:
            d = dec(d, e)
            
        #output
        x = self.out_fc(d.flatten(start_dim=1))
        
        return x

#=====================================#
#            {Hyperparams}            #
#=====================================#

def transformer(dataset, Enc_len, Dec_len,Out_len,Dim_val,Dim_attn,N_heads,N_d,N_e,h,val_spl,Epoch,Lr):
  enc_seq_len = int(Enc_len) 
  dec_seq_len = int(Dec_len) 
  output_sequence_length = int(Out_len) 

  dim_val = int(Dim_val) 
  dim_attn = int(Dim_attn) 
  lr = Lr
  epochs = int(Epoch)

  n_heads = int(N_heads) 
  n_val=int(len(dataset) * val_spl)
  n_decoder_layers = int(N_d) 
  n_encoder_layers = int(N_e) 

  batch_size = 10
  h = int(h)
  #=====================================#
  #    {init network and optimizer}     #
  #=====================================#
  t = Transformer(dim_val, dim_attn, 1,dec_seq_len, 
                output_sequence_length,
                n_decoder_layers, n_encoder_layers, n_heads)
                
  optimizer = torch.optim.Adam(t.parameters(), lr=lr)

  #=====================================#
  #         {keep track of loss}        #
  #=====================================#

  losses = []
  losses_val = []
  for e in range(epochs):
        optimizer.zero_grad()
        X_train, Y_train,X_val, Y_val = get_data6(dataset,n_val, enc_seq_len, output_sequence_length)
        #Forward pass and calculate loss
        net_out = t(X_train)
        #random.seed(1253)
        net_out_val = t(X_val)
        #print(net_out.shape,Y.shape)
        loss = torch.mean((net_out - Y_train) ** 2)
        loss_val = torch.mean((net_out_val - Y_val) ** 2)
        #backwards pass
        loss.backward()
        optimizer.step()

        losses.append(loss)
        losses_val.append(loss_val)

  #=====================================#
  #             {Prediction}            #
  #=====================================#

  l=len(dataset)-enc_seq_len
  x = dataset[l:(l+enc_seq_len)]

  for i in range(0, h, output_sequence_length):
    Y = [torch.tensor(x[i:]).unsqueeze(-1).numpy().tolist()]
    x = [torch.tensor(x).unsqueeze(-1).numpy().tolist()]
    q = torch.tensor(x).float()
    if(output_sequence_length == 1):
        x[0].append([t(q).detach().squeeze().numpy().tolist()])
    else:
        for a in t(q).detach().squeeze().numpy():
            x[0].append([a])
    x=[i[0] for i in x[0]]

  new_data = dataset[0:l] + x
  return new_data,losses,losses_val
