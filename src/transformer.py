"""

@author: Jinal Shah

This class will be for the transformer model


"""
import torch
import torch.nn as nn
import math
import sys

sys.path.append("../")

# Getting the vocab
vocab = torch.load("vocab.pt")

# Encodings and Embeddings for Transformer
class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size,emb_size,padding_idx=vocab['<pad>'])
        self.embed_size = emb_size

    def forward(self,tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.embed_size)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self,emb_size:int, dropout:float, maxlen:int = 500):
        super(PositionalEncoding,self).__init__()
        den = torch.exp(-torch.arange(0,emb_size,2)*math.log(10000) / emb_size)
        pos = torch.arange(0,maxlen).reshape(maxlen,1)
        pos_embedding = torch.zeros((maxlen,emb_size))
        pos_embedding[:,0::2] = torch.sin(pos * den)
        pos_embedding[:,1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)

        # Saving the positional encoding in the model state dict, but making sure PyTorch doesn't "train"
        # these parameters because they don't need to be trained
        self.register_buffer('pos_embedding',pos_embedding)

    def forward(self,token_embedding):
        return self.dropout(token_embedding + self.pos_embedding)

# Transformer Model
class Model(nn.Module):
    def __init__(self,d_model:int,nheads:int,dim_feedforward:int,dropout:float,num_layers:int):
        super().__init__()
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model,nheads,dim_feedforward,dropout,batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer,num_layers)
        self.fc1 = nn.Linear(d_model,1)
    
    # Forward Function
    def forward(self,X,src_key_padding_mask):
        output = self.transformer(X,src_key_padding_mask=src_key_padding_mask)
        output = torch.mean(output,dim=1)
        return nn.functional.sigmoid(self.fc1(output))     