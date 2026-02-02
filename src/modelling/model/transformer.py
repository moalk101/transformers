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

    def generate(self, src, max_length=50):
        device = next(self.parameters()).device
        src = src.to(device)
        PAD = 0
        batch_size = src.size(0)
    
        all_outputs = []
    
        for i in range(batch_size):
            seq = src[i]
            length = (seq != PAD).sum().item()
            trimmed_seq = seq[:length].unsqueeze(0) 
    

            src_emb = self.embedding(trimmed_seq)
            memory = self.encoder(src_emb, None)
    

            tgt = torch.tensor([[1]], device=device, dtype=torch.long)
    
            for _ in range(max_length):
                tgt_emb = self.embedding(tgt)
                output = self.decoder(tgt_emb, memory, None, None)
                logits = self.output_projection(output)
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
                
                if next_token.item() == 2:  
                    break
                
            all_outputs.append(tgt.squeeze(0))  
    
        return all_outputs


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

        masked_encoder_output = encoder_output * src_mask.unsqueeze(-1)

        output = self.decoder(
            tgt_emb,
            masked_encoder_output,
            tgt_mask=tgt_mask,
            src_mask=src_mask

        )

        logits = self.output_projection(output)

        return logits
