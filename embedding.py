import torch
from torch import nn

PE_NORM = 10000.0


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_chain_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.max_chain_len = max_chain_len
        self.pe = torch.zeros(max_chain_len, embed_dim)
        position = torch.arange(0, max_chain_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2)
            * -(torch.log(torch.tensor(PE_NORM)) / embed_dim)
        )
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, tokens: torch.Tensor):
        seq_len, batch_size, _ = tokens.shape
        pe = (
            self.pe[:seq_len, :]
            .unsqueeze(1)
            .expand(seq_len, batch_size, self.embed_dim)
        )
        return tokens + pe


class PotentialSpinEncoding(nn.Module):
    def __init__(self, embed_dim, max_chain_len, test_init_base=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Trainable scaling for a potential
        self.base_potential_vect = (
            nn.Parameter(torch.randn(embed_dim))
            if test_init_base is None
            else nn.Parameter(test_init_base)
        )

        # Trainable scaling for a particle
        self.base_spin_vect = (
            nn.Parameter(torch.randn(embed_dim))
            if test_init_base is None
            else nn.Parameter(test_init_base)
        )

        # Additive positional encoding function
        self.pos_encoding = PositionalEncoding(embed_dim, max_chain_len)

        # TODO: perhaps perform a linear map from R^1 to R ^ m for each potential and/or spin?
        # Would provide more dimensions for the physical interpretation of a spin or a potential?

    def forward(self, potentials: torch.Tensor, spins: torch.Tensor):
        """
        potentials: (seq, batch)
        spins: (seq, batch)
        """

        a = self.base_potential_vect  # hidden .unsqueeze(0).unsqueeze(0)
        b = potentials.unsqueeze(
            2
        )  # unsqueeze(2) is necessary to create a dim aligned with the embedding vect
        potentials = self.pos_encoding(a * b)

        a = self.base_spin_vect.unsqueeze(0).unsqueeze(0)
        b = spins.unsqueeze(2)
        spins = self.pos_encoding(a * b)

        seq = torch.cat((potentials, spins), dim=0)
        return seq
