"""Residual MLP with entity embeddings and two heads (poverty logit, log-consumption)."""
import torch
import torch.nn as nn


def embedding_dim(cardinality, cap=16):
    return int(min(cap, round(1.6 * cardinality ** 0.56)))


class ResidualBlock(nn.Module):
    def __init__(self, width=128, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )

    def forward(self, x):
        return x + self.net(x)


class PovertyNet(nn.Module):
    def __init__(self, n_numeric, cardinalities, width=128, hidden=256, n_blocks=3, dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, embedding_dim(c)) for c in cardinalities])
        in_dim = n_numeric + sum(e.embedding_dim for e in self.embeddings)
        self.inp = nn.Sequential(nn.Linear(in_dim, width), nn.LayerNorm(width), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(width, hidden, dropout) for _ in range(n_blocks)])
        self.poverty_head = nn.Linear(width, 1)
        self.consumption_head = nn.Linear(width, 1)

    def forward(self, x_num, x_cat):
        parts = [x_num] + [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        h = self.blocks(self.inp(torch.cat(parts, dim=1)))
        return self.poverty_head(h).squeeze(-1), self.consumption_head(h).squeeze(-1)
