import torch.nn as nn


class PostionalFeedForward(nn.Module):
    
    def __init__(self,d_model,d_hl,dropout=0.1):
        
        super().__init__()
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_hl),  
            nn.ReLU(),                 
            nn.Linear(d_hl, d_model),
            nn.Dropout(dropout)     
        )
        
    def forward(self,x):
        x = self.ff(x)
        return x