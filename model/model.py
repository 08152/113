import torch
import torch.nn as nn

from .embeddings import TokenEmbedding, PositionalEmbedding
from .attention import SelfAttention


class FeedForward(nn.Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_size,
                hidden_size
            ),

            nn.GELU(),

            nn.Linear(
                hidden_size,
                embedding_size
            )
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_size,
        number_of_heads,
        hidden_size
    ):
        super().__init__()

        self.attention = SelfAttention(
            embedding_size,
            number_of_heads
        )

        self.feed_forward = FeedForward(
            embedding_size,
            hidden_size
        )

        self.norm1 = nn.LayerNorm(
            embedding_size
        )

        self.norm2 = nn.LayerNorm(
            embedding_size
        )

    def forward(self, x):

        attention_output = self.attention(
            self.norm1(x)
        )

        x = x + attention_output

        feed_forward_output = self.feed_forward(
            self.norm2(x)
        )

        x = x + feed_forward_output

        return x


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size=128,
        number_of_heads=4,
        hidden_size=512,
        number_of_layers=4,
        max_length=256
    ):
        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size,
            embedding_size
        )

        self.position = PositionalEmbedding(
            max_length,
            embedding_size
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                embedding_size,
                number_of_heads,
                hidden_size
            )
            for _ in range(number_of_layers)
        ])

        self.final_norm = nn.LayerNorm(
            embedding_size
        )

        self.output = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, token_ids):

        x = self.embedding(
            token_ids
        )

        x = self.position(
            x
        )

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(
            x
        )

        logits = self.output(
            x
        )

        return logits
