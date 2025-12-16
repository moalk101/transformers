import torch
import torch.nn as nn

from src.modelling.layers.attention import MultiHeadAttention
from src.modelling.layers.feedforward import PostionalFeedForward


class TransformerEncoder(nn.Module):
    
    def __init__(self,d_model, n_heads, feature_dim, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model,n_heads)
        self.ff = PostionalFeedForward(d_model,feature_dim,dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask = None):
        attn_out = self.attention(x,x,x,mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        ff_out = self.ff(x)
        x = x +self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x