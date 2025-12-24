import torch.nn as nn

from src.modelling.embedding.positional_encoding import PositionalEncoding
from src.modelling.embedding.embedding import Embedding


class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()

        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)

    def forward(self, token):
        device = token.device
        token_emb = self.token_emb(token).to(device)
        pos_emb = self.pos_emb(token_emb).to(device)

        return pos_emb
