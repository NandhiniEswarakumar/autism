import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
================================================================
STEP 04b -- TabM WITH PLE ENCODING (Upgraded Architecture)
================================================================
Enhancements over basic TabM:
  1. Piecewise Linear Encoding (PLE) for numerical features
     - Converts each numerical value into n_bins learned features
     - Captures non-linear patterns without manual feature engineering
     - Reference: Gorishniy et al. 2022 (NeurIPS)

  2. Larger hidden dimensions (256 units)
  3. More ensemble heads (k=16)
  4. More layers (3 hidden layers)
  5. Feature-type aware architecture
     - Numerical features -> PLE -> Linear
     - Binary features   -> Direct Linear embedding

Architecture:
  Input (numerical+binary)
       |
  [PLE Encoding for num features]    [Direct embed for binary]
       |___________________________________|
                    |
            [Concatenate]
                    |
        [BatchEnsembleLinear x k] x 3 layers
                    |
        [Average k heads -> 1 output]
                    |
                [Sigmoid]
================================================================
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# PLE ENCODING  (Piecewise Linear Encoding)
# ─────────────────────────────────────────────────────────────
class PLEEncoding(nn.Module):
    """
    Converts numerical features into piecewise-linear representations.
    Each numerical feature x becomes a n_bins-dimensional vector.
    Uses quantile-based boundaries fitted on training data.
    """
    def __init__(self, n_num_features: int, n_bins: int = 50):
        super().__init__()
        self.n_num  = n_num_features
        self.n_bins = n_bins
        # Boundaries will be registered as buffer (non-trainable)
        self.register_buffer(
            "boundaries",
            torch.zeros(n_num_features, n_bins + 1)
        )
        self._fitted = False

    def fit(self, X_num: np.ndarray):
        """Compute quantile boundaries from training data."""
        boundaries = np.zeros((self.n_num, self.n_bins + 1), dtype=np.float32)
        for i in range(self.n_num):
            col = X_num[:, i]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                boundaries[i] = np.linspace(0, 1, self.n_bins + 1)
            else:
                q = np.percentile(col, np.linspace(0, 100, self.n_bins + 1))
                # Ensure strict monotonicity
                q = np.maximum.accumulate(q)
                if q[-1] == q[0]:
                    q = np.linspace(q[0] - 1e-6, q[0] + 1e-6, self.n_bins + 1)
                boundaries[i] = q
        self.boundaries = torch.from_numpy(boundaries)
        self._fitted = True
        return self

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        x_num : (batch, n_num_features)
        output: (batch, n_num_features * n_bins)
        """
        batch = x_num.shape[0]
        out   = torch.zeros(batch, self.n_num * self.n_bins,
                            device=x_num.device, dtype=x_num.dtype)
        for i in range(self.n_num):
            xi = x_num[:, i].unsqueeze(1)              # (B,1)
            b  = self.boundaries[i]                    # (n_bins+1,)
            lo = b[:-1].unsqueeze(0)                   # (1, n_bins)
            hi = b[1:].unsqueeze(0)                    # (1, n_bins)
            span = (hi - lo).clamp(min=1e-8)
            enc  = ((xi - lo) / span).clamp(0.0, 1.0) # (B, n_bins)
            out[:, i * self.n_bins:(i + 1) * self.n_bins] = enc
        return out

    def output_dim(self) -> int:
        return self.n_num * self.n_bins


# ─────────────────────────────────────────────────────────────
# BATCH ENSEMBLE LINEAR  (same as before)
# ─────────────────────────────────────────────────────────────
class BatchEnsembleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, k: int):
        super().__init__()
        self.k   = k
        self.W   = nn.Linear(in_features, out_features, bias=True)
        self.r   = nn.Parameter(torch.empty(k, in_features))
        self.s   = nn.Parameter(torch.empty(k, out_features))
        nn.init.normal_(self.r, mean=1.0, std=0.1)
        nn.init.normal_(self.s, mean=1.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, k, in) OR (batch, in)
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.k, -1)
        x_scaled  = x * self.r.unsqueeze(0)              # (B, k, in)
        out       = self.W(x_scaled)                      # (B, k, out)
        out_scaled= out * self.s.unsqueeze(0)             # (B, k, out)
        return out_scaled


# ─────────────────────────────────────────────────────────────
# TABM-PLE MODEL
# ─────────────────────────────────────────────────────────────
class TabMPLE(nn.Module):
    """
    TabM with Piecewise Linear Encoding for numerical features.

    Args:
        n_num_features  : number of numerical (continuous) features
        n_bin_features  : number of binary / categorical features
        n_bins          : PLE bins per numerical feature
        hidden_dim      : hidden layer width
        n_layers        : number of BatchEnsemble hidden layers
        k               : number of ensemble heads
        dropout         : dropout rate
    """
    def __init__(self,
                 n_num_features: int,
                 n_bin_features: int,
                 n_bins:         int = 50,
                 hidden_dim:     int = 256,
                 n_layers:       int = 3,
                 k:              int = 16,
                 dropout:        float = 0.2):
        super().__init__()
        self.k             = k
        self.n_num         = n_num_features
        self.n_bin         = n_bin_features
        self.n_bins        = n_bins

        # PLE encoder for numerical features
        self.ple = PLEEncoding(n_num_features, n_bins) if n_num_features > 0 else None

        # Input dimension after PLE + binary
        ple_dim   = n_num_features * n_bins if n_num_features > 0 else 0
        input_dim = ple_dim + n_bin_features

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # BatchEnsemble hidden layers
        self.hidden_layers = nn.ModuleList([
            BatchEnsembleLinear(hidden_dim, hidden_dim, k)
            for _ in range(n_layers)
        ])
        self.norms   = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

        # Output head (k separate heads)
        self.output_head = BatchEnsembleLinear(hidden_dim, 1, k)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_num: torch.Tensor, x_bin: torch.Tensor) -> torch.Tensor:
        """
        x_num : (B, n_num_features)  numerical features
        x_bin : (B, n_bin_features)  binary/categorical features
        Returns: (B,) logits
        """
        parts = []
        if self.ple is not None and x_num.shape[1] > 0:
            parts.append(self.ple(x_num))
        if x_bin.shape[1] > 0:
            parts.append(x_bin)
        x = torch.cat(parts, dim=-1)                        # (B, input_dim)

        x = self.input_proj(x)                              # (B, hidden)

        # Expand for k heads
        x = x.unsqueeze(1).expand(-1, self.k, -1)          # (B, k, hidden)

        # Hidden layers with residual connections
        for layer, norm in zip(self.hidden_layers, self.norms):
            residual = x
            x = layer(x)                                    # (B, k, hidden)
            x = self.act(x)
            x = self.dropout(x)
            # LayerNorm on last dim
            x = norm(x + residual)                          # residual connection

        # Output
        logits = self.output_head(x)                        # (B, k, 1)
        logits = logits.squeeze(-1)                         # (B, k)
        logits = logits.mean(dim=1)                         # (B,) averaged
        return logits

    @classmethod
    def build(cls, n_num: int, n_bin: int, **kwargs):
        return cls(n_num_features=n_num, n_bin_features=n_bin, **kwargs)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.set_num_threads(2)

    n_num, n_bin = 9, 58
    B = 32

    model = TabMPLE.build(n_num=n_num, n_bin=n_bin,
                          n_bins=50, hidden_dim=256,
                          n_layers=3, k=16, dropout=0.2)
    print(f"TabM-PLE parameters: {model.count_params():,}")

    # Fit PLE on dummy data
    dummy_num = np.random.randn(1000, n_num).astype(np.float32)
    model.ple.fit(dummy_num)

    # Forward pass
    x_num = torch.randn(B, n_num)
    x_bin = torch.randint(0, 2, (B, n_bin)).float()
    out = model(x_num, x_bin)
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print("TabM-PLE architecture test PASSED")
