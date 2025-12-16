import torch
import torch.nn as nn

from src.modelling.layers.attention import MultiHeadAttention
from src.modelling.layers.feedforward import PostionalFeedForward


class BaseTransformerLayer(nn.Module):
    
    def __init__(self,input_dim, num_heads, feature_dim, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(input_dim,num_heads)
        self.feature_transformation = PostionalFeedForward(input_dim,feature_dim,dropout)
        
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask = None):
        attn_out = self.self_attention(x,x,x,mask)
        x = x + self.dropout1(attn_out)
        x = self.layer_norm_1(x)
        
        ff_out = self.feature_transformation(x)
        x = x +self.dropout2(ff_out)
        x = self.layer_norm_2(x)
        
        return x