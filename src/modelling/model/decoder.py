import torch
import torch.nn as nn

from src.modelling.functional import TransformerDecoderLayer

class Decoder(nn.Module):
    def __init__(
        self, d_model,  n_heads, dim_feedforward, dropout, num_layers
    ):
        super().__init__()
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, tgt_emb, src_emb, src_mask, tgt_mask):
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, src_emb, src_mask, tgt_mask)
        return tgt_emb
