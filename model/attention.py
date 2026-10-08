import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):

    def __init__(self, embedding_size, number_of_heads):
        super().__init__()

        if embedding_size % number_of_heads != 0:
            raise ValueError(
                "embedding_size muss durch number_of_heads teilbar sein."
            )

        self.embedding_size = embedding_size
        self.number_of_heads = number_of_heads
        self.head_size = embedding_size // number_of_heads

        self.query = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.key = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.value = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.output = nn.Linear(
            embedding_size,
            embedding_size
        )

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_size
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_size
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_size
        ).transpose(1, 2)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_size)

        # Verhindert, dass ein Token zukünftige Tokens sieht.
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            ),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        result = torch.matmul(
            attention,
            v
        )

        result = result.transpose(
            1,
            2
        ).contiguous()

        result = result.view(
            batch_size,
            sequence_length,
            self.embedding_size
        )

        return self.output(result)
