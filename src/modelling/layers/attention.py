import torch.nn as nn
import torch
import math


class MultiHeadAttention(nn.Module):
    
    def __init__(self,d_model, n_heads, mask_future=False):
        super().__init__()
        
        assert d_model % n_heads == 0, "number of heads must be divisible by number of heads!"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.future_mask = mask_future
        self.d_k = d_model // n_heads
        
        
        self.query_transform  = nn.Linear(d_model, d_model, bias=False)
        self.key_transform    = nn.Linear(d_model, d_model, bias=False)
        self.value_transform  = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)
        
    def split(self,x:torch.Tensor):
        """
        Splits the input into n Splits.
        Input shape (batch_size,seq_len,d_model)
        The input is reshaped to (batch_size,n_heads,seq_len,d_k)

        Args:
            x (torch.Tensor): The matrix, that should be splitted
        """
        
        b,sq_len, d_model = x.size()
        
        x = x.view(b,sq_len,self.n_heads,self.d_k)
        x = torch.permute(x,(0,2,1,3)).contiguous()
        return x
    
    def combine(self,x:torch.Tensor):
        """
        Reshape the input into the original shape.
        Input shape (batch_size,n_heads,seq_len,d_k) -> (batch_size,seq_len,d_model)

        Args:
            x (torch.Tensor)
        """
        
        b, n_heads, sq_len, d_k = x.size()
        
        x = torch.permute(x,(0,2,1,3)).contiguous()
        x = x.view(b,sq_len,self.d_model)
        return x
    
    def forward(self,q,k,v,mask=None):
        
        Q = self.split(self.query_transform(q))
        K = self.split(self.key_transform(k))
        V = self.split(self.value_transform(v))
        qk = torch.matmul(Q,K.transpose(-2,-1))
        scaled_qk = qk / math.sqrt(self.d_k)
        
        if self.future_mask == True:
            seq_len = scaled_qk.size(-1)
            future_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            while future_mask.dim() < scaled_qk.dim():
                future_mask = future_mask.unsqueeze(0)
            scaled_qk = scaled_qk.masked_fill(future_mask,float("-inf"))
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scaled_qk = scaled_qk.masked_fill(mask == 0, float("-inf"))
            
        attn = torch.softmax(scaled_qk,dim=-1)
        output = torch.matmul(attn,V)
        output = self.combine(output)
        
        output = self.output_transform(output)
        
        return output
        