import torch
import torch.nn as nn

from modelling.layers.attention import MultiHeadAttention
from modelling.layers.feedforward import PostionalFeedForward

class TransformerDecoderLayer(nn.Module):
    
    def __init__(self,input_dim,num_heads, feature_dim,dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(
            input_dim,
            num_heads,
            mask_future=True
        )
        
        self.encoder_attention = MultiHeadAttention(input_dim,num_heads,mask_future=False)
        self.feature_transformation = PostionalFeedForward(
            input_dim,feature_dim,dropout
        )
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        
    def forward(self, target, encoder_output, src_mask=None, target_mask=None):
        
        attn_out = self.self_attention(target,target,target,mask=target_mask)
        target = target + self.dropout1(attn_out)
        target = self.layer_norm_1(target)
        
        attn_out = self.encoder_attention(q=target,k=encoder_output,v=encoder_output,mask=src_mask)
        target = target + self.dropout1(attn_out)
        target = self.layer_norm_2(target)
        
        ff_out = self.feature_transformation(target)
        target = target + self.dropout1(ff_out)
        target = self.layer_norm_3(target)
        
        return target