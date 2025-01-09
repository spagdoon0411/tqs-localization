import torch
from torch import nn

PE_NORM = 10000.0


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_chain_len=128):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size

        self.pe = torch.zeros(max_chain_len, embed_size)
        position = torch.arange(0, max_chain_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2)
            * -(torch.log(torch.tensor(PE_NORM)) / embed_size)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, potentials: torch.Tensor, configs: torch.Tensor):
        # potentials: (seq, batch, embedding)
        # configs: (seq, batch, site_configs)
        seq_len, batch_size, _ = potentials.shape
        pe = (
            self.pe[:seq_len, :]
            .unsqueeze(1)
            .expand(seq_len, batch_size, self.embed_size)
        )
        return potentials + pe, configs + pe


class Embedding(nn.Module):
    def __init__(self, site_configs=2, embed_size=32, max_chain_len=128):
        super(Embedding, self).__init__()
        self.site_configs = site_configs
        self.embed_size = embed_size

        self.config_to_embed = nn.Linear(site_configs, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)

        # Just a buffer that we grow as needed?
        self.potential_basis = nn.Parameter(embed_size, max_chain_len)

    # Forward pass takes a sequence of site potentials and a
    # sequence of site configurations
    def forward(self, potentials: torch.Tensor, configs: torch.Tensor):
        # potentials: (batch)
        #   -> (batch, embedding)
        # via plain scaling of the potential basis vectors

        # Scale the columns of the potential basis by the potentials
        potentials = potentials.unsqueeze(2) * self.potential_basis

        # configs: (seq, batch)
        #   ->(seq, batch, site_configs)
        #   ->(seq, batch, embedding)

        pass
