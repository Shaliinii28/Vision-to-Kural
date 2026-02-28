# app/model.py
"""
ProjectionHead definition.

This file is the single source of truth for the MLP architecture.
Both the training script (scripts/03_train_projection.py) and the
inference pipeline (app/inference.py) import from here.

If you change the architecture, update ONLY this file.
Then re-run steps 03 and 04 to regenerate projection.pt and
kural_index.faiss.
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    2-layer MLP that projects any feature dimension to a shared
    512-dim normalised embedding space.

    Used for:
      - Image side:  CLIP ViT-L/14 features (768-dim) → 512-dim
      - Text side:   Sarvam-2B features (2048-dim) → 512-dim

    The L2 normalisation in forward() means that after projection,
    cosine similarity = dot product, which is what FAISS IndexFlatIP
    computes.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 512,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.in_dim     = in_dim
        self.out_dim    = out_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_dim) float tensor
        Returns:
            (batch_size, out_dim) L2-normalised float tensor
        """
        return nn.functional.normalize(self.net(x), dim=-1)

    def __repr__(self) -> str:
        return (
            f"ProjectionHead("
            f"in_dim={self.in_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"out_dim={self.out_dim})"
        )
