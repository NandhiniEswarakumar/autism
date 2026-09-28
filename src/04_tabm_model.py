import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
TabM model components: PLE encoder, BatchEnsemble layers and TabM blocks.

This module provides the building blocks used by the TabM architecture in
the project: piecewise-linear encoding (PLE) for numerical features, an
efficient BatchEnsemble linear layer, and the TabM MLP blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# PIECEWISE-LINEAR ENCODING  (PLE)
# -------------------------------------------------------------
class PLEEncoding(nn.Module):
    """
    Piecewise-Linear Encoding for a single numerical feature.

    Each scalar x is mapped to a B-dimensional vector:
      t_b = clip((x - edge_b) / (edge_{b+1} - edge_b), 0, 1)

    After quantile-bin construction, the encoder is differentiable
    and expressive enough to capture non-linear relationships.

    Args:
        n_bins  (int)  : Number of bins B (each feature -> B dims)
        edges   (Tensor, shape B+1): Bin edges (quantile-based)
    """

    def __init__(self, n_bins: int, edges: torch.Tensor):
        super().__init__()
        # Register as buffer so it moves with .to(device)
        self.register_buffer("edges", edges)          # (B+1,)
        self.n_bins = n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch,)  -- raw scalar values for one feature
        Returns:
            out : (batch, n_bins)
        """
        # Expand dims for broadcasting
        x_exp   = x.unsqueeze(-1)                      # (batch, 1)
        left    = self.edges[:-1]                      # (B,)
        right   = self.edges[1:]                       # (B,)
        width   = (right - left).clamp(min=1e-8)
        t       = (x_exp - left) / width              # (batch, B)
        return t.clamp(0.0, 1.0)


# -------------------------------------------------------------
# PLE ENCODER WRAPPER  (all numerical features)
# -------------------------------------------------------------
class PLEEmbedder(nn.Module):
    """
    Wraps one PLEEncoding per numerical feature and concatenates.

    Binary/categorical features are passed through unchanged.

    Args:
        n_num       (int)           : number of numerical columns
        n_bins      (int)           : bins per feature
        edges_list  (list[Tensor])  : one edge Tensor per num. feature
        n_passthru  (int)           : number of binary/pass-through features
    """

    def __init__(self, n_num, n_bins, edges_list, n_passthru=0):
        super().__init__()
        self.n_num     = n_num
        self.n_bins    = n_bins
        self.n_passthru= n_passthru
        self.encoders  = nn.ModuleList(
            [PLEEncoding(n_bins, edges_list[i]) for i in range(n_num)]
        )
        # Output dimension
        self.out_dim = n_num * n_bins + n_passthru

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, n_num + n_passthru)
        Returns:
            out : (batch, out_dim)
        """
        parts = []
        for i, enc in enumerate(self.encoders):
            parts.append(enc(x[:, i]))                 # (batch, n_bins)

        if self.n_passthru > 0:
            parts.append(x[:, self.n_num:])            # pass-through cols
        return torch.cat(parts, dim=-1)                 # (batch, out_dim)

class BatchEnsembleLinear(nn.Module):
    """
    Efficient Batch Ensemble for a Linear layer.

    Instead of K separate copies of W (expensive), we learn:
      - Shared W  in R^{in x out}
      - Per-head scale vectors:  r_k in R^in,  s_k in R^out

    Forward for head k:
      h_k(x) = (x x r_k) @ W x s_k + b

    This gives K diverse predictions at the cost of ~2K extra
    parameters per layer instead of K times the full weight matrix.

    Reference: Wen et al. "BatchEnsemble" ICLR 2020
    """

    def __init__(self, in_features: int, out_features: int, k: int, bias: bool = True):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.r = nn.Parameter(torch.ones(k, in_features))   # input scale
        self.s = nn.Parameter(torch.ones(k, out_features))  # output scale
        nn.init.normal_(self.r, mean=1.0, std=0.1)
        nn.init.normal_(self.s, mean=1.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch * k, in_features)
                The batch is replicated k times before calling this.
        Returns:
            out : (batch * k, out_features)
        """

        out = self.linear(x * self.r.repeat_interleave(
            x.shape[0] // self.k, dim=0))                   # (batch*k, out)
        out = out * self.s.repeat_interleave(
            x.shape[0] // self.k, dim=0)
        return out

class TabMBlock(nn.Module):
    """
    One MLP block used inside TabM:
      Linear (BatchEnsemble) -> BatchNorm1d -> GELU -> Dropout

    Args:
        in_dim   : input dimension
        out_dim  : output dimension
        k        : number of ensemble heads
        dropout  : dropout probability
    """

    def __init__(self, in_dim: int, out_dim: int, k: int, dropout: float = 0.1):
        super().__init__()
        self.linear = BatchEnsembleLinear(in_dim, out_dim, k)
        self.bn     = nn.BatchNorm1d(out_dim)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.linear(x))))

class TabM(nn.Module):
    """
    TabM -- Tabular Mini-batch Ensemble Model

    Architecture:
      1. [Optional] PLE embedding for numerical features
      2. Input projection linear layer
      3. n_layers x TabMBlock (BatchEnsemble MLP)
      4. Output head -- one logit per ensemble member
      5. Mean aggregation -> final logit

    Args:
        in_dim      (int)   : input feature dimension (after PLE if used)
        hidden_dim  (int)   : hidden layer width
        n_layers    (int)   : number of MLP blocks
        k           (int)   : ensemble size (number of heads)
        dropout     (float) : dropout probability
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128,
                 n_layers: int = 3, k: int = 32, dropout: float = 0.1):
        super().__init__()
        self.k       = k
        self.n_layers= n_layers

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_bn   = nn.BatchNorm1d(hidden_dim)

        # MLP blocks
        self.blocks = nn.ModuleList([
            TabMBlock(hidden_dim, hidden_dim, k, dropout)
            for _ in range(n_layers)
        ])

        # Output head -- produces one scalar per ensemble head
        self.head = BatchEnsembleLinear(hidden_dim, 1, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, in_dim)
        Returns:
            logits : (batch,)   -- averaged over ensemble heads
        """
        batch = x.shape[0]

        # --- Replicate input for each ensemble head ---
        # x : (batch, d) -> (batch*k, d)
        x_rep = x.repeat_interleave(self.k, dim=0)

        # --- Input projection ---
        h = self.input_bn(self.input_proj(x_rep))   # (batch*k, hidden)

        # --- MLP blocks ---
        for block in self.blocks:
            h = block(h)                              # (batch*k, hidden)

        # --- Output head ---
        logits_flat = self.head(h).squeeze(-1)        # (batch*k,)

        # --- Reshape and average over heads ---
        logits = logits_flat.view(batch, self.k).mean(dim=-1)  # (batch,)
        return logits

    @staticmethod
    def build(n_features: int, hidden_dim: int = 128, n_layers: int = 3,
              k: int = 32, dropout: float = 0.1) -> "TabM":
        """
        Convenience factory method.

        Example:
            model = TabM.build(n_features=7, hidden_dim=128, k=32)
        """
        return TabM(in_dim=n_features, hidden_dim=hidden_dim,
                    n_layers=n_layers, k=k, dropout=dropout)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -------------------------------------------------------------
# QUICK SANITY CHECK
# -------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    n_features  = 7
    batch_size  = 32
    hidden_dim  = 128
    n_layers    = 3
    k           = 16
    dropout     = 0.2

    model = TabM.build(n_features, hidden_dim=hidden_dim,
                       n_layers=n_layers, k=k, dropout=dropout)

    dummy = torch.randn(batch_size, n_features)
    out   = model(dummy)                         # forward pass

    print("=" * 55)
    print("  TabM Architecture Sanity Check")
    print("=" * 55)
    print(model)
    print(f"\n  Input  shape : {dummy.shape}")
    print(f"  Output shape : {out.shape}   (batch logits)")
    print(f"  Parameters   : {model.count_parameters():,}")
    print("=" * 55)
    assert out.shape == (batch_size,), f"Unexpected output shape: {out.shape}"
    print("\n  [OK]  All checks passed!")


