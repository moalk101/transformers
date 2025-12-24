import torch
import torch.nn as nn

from src.modelling.functional import BaseTransformerLayer


class Encoder(nn.Module):
    def __init__(
        self, d_model,  n_heads, dim_feedforward, dropout, num_layers
    ):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                BaseTransformerLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src_emb, src_mask):
        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_mask)
        return src_emb
