import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):

    def __init__(self, max_length, embedding_size):
        super().__init__()

        positions = torch.arange(max_length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embedding_size,
                2
            ) * (-math.log(10000.0) / embedding_size)
        )

        positional = torch.zeros(
            max_length,
            embedding_size
        )

        positional[:, 0::2] = torch.sin(
            positions * div_term
        )

        positional[:, 1::2] = torch.cos(
            positions * div_term
        )

        self.register_buffer(
            "positional",
            positional.unsqueeze(0)
        )

    def forward(self, x):
        length = x.size(1)

        return x + self.positional[:, :length, :]
