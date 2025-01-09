# This module is meant to implement a solution to take in a variable-
# length list of entries in a spatial potential function and return
# a fixed-embedding size tensor.

import torch
from torch import nn


class PotentialEmbedding(nn.Module):
    """
    This class takes a sequence of potentials of length m
    and maps it to a tensor of corresponding embeddings of dimension
    (m, embed_dim).

    Architecture overview:
    - Input sequences tensor: (m, batch)
    - Pre-RNN embedding: (m, batch, embed1)
    - Post-RNN embedding: (m, batch, embed2)
    """

    def __init__(self, embed1: int, embed2: int):
        super(PotentialEmbedding, self).__init__()
        self.embed1 = embed1
        self.embed2 = embed2

        # Embedding applied to a SINGLE token
        self.embedding = nn.Linear(1, embed1)
        self.lstm = nn.LSTM(embed1, embed2, batch_first=True, bidirectional=True)
        self.final = nn.Linear(embed2 * 2, embed2)

    def forward(self, potentials: torch.Tensor):
        """
        Forward pass takes a sequence of potentials and returns a
        sequence of embeddings.

        Args:
            potentials (torch.Tensor): (m, batch) tensor of potentials

        Returns:
            torch.Tensor: (m, batch, embed2) tensor of embeddings
        """
        # potentials: (m, batch) -> (m, batch, embed1)

        return potentials
