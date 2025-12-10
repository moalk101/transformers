import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    
    def __init__(self,d_model, n_heads, future_mask=False):
        super().__init__()
        
        assert d_model % n_heads == 0, "number of heads must be divisible by number of heads!"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.future_mask = future_mask
        self.d_k = d_model // n_heads
        
        
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        
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
        x = torch.permute(x,(0,2,1,3))
        return x
    
    def combine(self,x:torch.Tensor):
        """
        Reshape the input into the original shape.
        Input shape (batch_size,n_heads,seq_len,d_k) -> (batch_size,seq_len,d_model)

        Args:
            x (torch.Tensor)
        """
        
        b,sq_len,self.n_heads,self.d_k = x.size()
        
        x = torch.permute(x,(0,2,1,3)).contiguous()
        x = x.view(b,sq_len,self.d_model)
        return x
    
    def forward(self,q,k,v):
        
        Q = self.split(self.Wq(q))
        K = self.split(self.Wq(k))
        V = self.split(self.Wq(v))
        
        qk = torch.matmul(Q,K.transpose(-2,-1))
        scaled_qk = torch.divide(qk,torch.sqrt(self.d_k))
        
        if self.future_mask == True:
            mask = torch.triu(torch.ones(scaled_qk.size(-2),scaled_qk.size(-1)),diagonal=1).bool()
            mask = mask.expand(scaled_qk.size())
            scaled_qk = scaled_qk.masked_fill(mask,float("-inf"))
            
        attn = torch.softmax(scaled_qk,dim=-1)
        output = torch.matmul(attn,V)
        output = self.combine(output)
        
        output = self.Wo(output)
        
        return output
            
        
        
        