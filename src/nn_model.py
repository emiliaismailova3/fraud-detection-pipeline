"""A plain PyTorch neural network for tabular data.

Architecture (a standard "MLP with entity embeddings"):

    categorical codes ──> one nn.Embedding per column ──┐
                                                         ├─ concat ─> [Linear → BatchNorm → ReLU → Dropout] x2 ─> Linear → 1 logit
    scaled numeric features ─────────────────────────────┘

An embedding turns a category code (e.g. email domain #17) into a small
learned vector, so similar categories can end up close to each other. This is
the neural-net counterpart of how trees split on categories.
"""
import torch
from torch import nn
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    """Wraps numpy arrays so a DataLoader can serve shuffled mini-batches."""

    def __init__(self, x_num, x_cat, y):
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


def embedding_dim(cardinality: int) -> int:
    # Common rule of thumb: grows with the number of categories, capped at 16.
    return min(16, (cardinality + 1) // 2)


class TabularMLP(nn.Module):
    def __init__(self, n_numeric: int, cardinalities: list[int],
                 hidden: tuple[int, ...] = (256, 128), dropout: float = 0.3):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(card, embedding_dim(card)) for card in cardinalities
        )
        in_dim = n_numeric + sum(e.embedding_dim for e in self.embeddings)

        layers: list[nn.Module] = []
        for width in hidden:
            layers += [nn.Linear(in_dim, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat[:, i] holds the codes of categorical column i for the whole batch
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num, *embedded], dim=1)
        # Return raw logits; the sigmoid lives inside the loss for numerical stability
        return self.mlp(x).squeeze(1)
