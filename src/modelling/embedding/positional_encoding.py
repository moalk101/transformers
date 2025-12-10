import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len):
        
        super().__init__()
        
        pe = torch.zeros(max_len,d_model)
        positions = torch.arange(0,max_len).unsqueeze(1)
        
        steps = torch.arange(0,d_model,2)
        
        pe[:,0::2] = torch.sin(positions / (10000) ** (steps / d_model))
        pe[:,1::2] = torch.cos(positions / (10000) ** (steps / d_model))

        pe = pe.unsqueeze(0)
        
        self.register_buffer("pe",pe)    
        
    def forward(self,x):
        
        length = x.size(1)
        
        x = x + self.pe[:,:length,:]    
        
        return x