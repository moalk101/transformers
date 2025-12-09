import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
    
    
    def forward(self,x):
        return self.embedding(x)