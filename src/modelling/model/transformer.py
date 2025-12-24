import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from src.modelling.model.model_embedding import TransformerEmbedding
from src.modelling.model.encoder import Encoder
from src.modelling.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()

        self.d_model = d_model

        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
        )

        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        self.output_projection.weight = self.embedding.token_emb.embedding.weight

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ):

        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)

        encoder_output = self.encoder(
            src_emb,
            src_mask=src_mask,
        )

        output = self.decoder(
            tgt_emb,
            encoder_output,
            tgt_mask=tgt_mask,
            src_mask=src_mask

        )

        logits = self.output_projection(output)

        return logits
