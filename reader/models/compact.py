"""Clean re-implementation of the archived compact decoder ``tcnseq_ceilmean_maxnorm_narrow``.

Executed graph read from ``cba_online/model_tcn_sequence.py`` (CBAPolyphaseTCNSequence
with polyphase_components=1) and ``cba_online/model.py`` (CausalBandAttention):

  x[B,C,T] -> 3 causal temporal convs (16/32/64 taps, 16 carriers each, no bias)
           -> BatchNorm2d(48)
           -> depthwise spatial conv (48 -> 96, (C,1), groups=48, max-norm 1, no bias)
           -> BatchNorm2d(96) -> LeakyReLU(.2)                          [B,96,T]
           -> right-pad, mean-pool every `pool` samples, rescale the partial last window
           -> Dropout(.3) -> Linear(96,64) -> + learned positions        [B,N,64]
           -> causal dilated depthwise TCN (3 blocks, kernel 6, dilation 1/2/4,
              LeakyReLU .2, dropout .3, second conv zero-initialised)
           -> Linear(64,classes) with per-row max-norm .25              [B,N,classes]

Parity with the archived build is asserted in ``reader/tests/test_compact_parity.py``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaxNormConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.data = torch.renorm(self.weight.data, 2, 0, self.max_norm)
        return super().forward(x)


class CausalTCNBlock(nn.Module):
    def __init__(self, channels, kernel=6, dilation=1, dropout=0.3):
        super().__init__()
        self.kernel, self.dilation = kernel, dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation, groups=channels, bias=False)
        self.norm1 = nn.BatchNorm1d(channels)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation, groups=channels, bias=False)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act2 = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv2.weight)

    def _pad(self, x):
        return F.pad(x, ((self.kernel - 1) * self.dilation, 0))

    def forward(self, x):
        h = self.drop(self.act1(self.norm1(self.conv1(self._pad(x)))))
        h = self.drop(self.act2(self.norm2(self.conv2(self._pad(h)))))
        return x + h


class CausalTCN(nn.Module):
    def __init__(self, channels, depth=3, kernel=6, dropout=0.3):
        super().__init__()
        self.blocks = nn.ModuleList([CausalTCNBlock(channels, kernel, 2 ** i, dropout) for i in range(depth)])

    def forward(self, tokens):  # [B,N,D] -> [B,N,D]
        h = tokens.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        return h.transpose(1, 2)


def ceil_mean_pool(h, pool):
    """Mean over non-overlapping windows; the partial final window is rescaled by pool/observed."""
    samples = h.shape[-1]
    padding = (-samples) % pool
    if padding:
        h = F.pad(h, (0, padding))
    mean = F.avg_pool1d(h, pool, pool)
    if padding:
        mean = mean.clone()
        mean[..., -1] = mean[..., -1] * (pool / (pool - padding))
    return mean


class CompactDecoder(nn.Module):
    def __init__(self, n_ch, n_cls, pool=32, kernels=(16, 32, 64), f_per=16, spatial_multiplier=2,
                 embedding=64, dropout=0.3, head_max_norm=0.25, pos_tokens=None, tcn_depth=3):
        super().__init__()
        self.kernels, self.pool, self.head_max_norm = tuple(kernels), pool, head_max_norm
        carriers = f_per * len(kernels)
        features = carriers * spatial_multiplier
        self.temporal = nn.ModuleList([nn.Conv2d(1, f_per, (1, k), bias=False) for k in kernels])
        self.tbn = nn.BatchNorm2d(carriers)
        self.spatial = MaxNormConv2d(carriers, features, (n_ch, 1), groups=carriers, bias=False, max_norm=1.0)
        self.sbn = nn.BatchNorm2d(features)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(features, embedding)
        if pos_tokens is None:
            pos_tokens = max(64, (1024 + pool - 1) // pool)
        self.pos = nn.Parameter(torch.randn(1, pos_tokens, embedding) * 0.02)
        self.tcn = CausalTCN(embedding, depth=tcn_depth, dropout=dropout)
        self.head = nn.Linear(embedding, n_cls)

    # -- stages exposed separately so diagnostics and fused variants can reuse them
    def carriers(self, x):
        raw = x.unsqueeze(1)
        temporal = [conv(F.pad(raw, (k - 1, 0))) for k, conv in zip(self.kernels, self.temporal)]
        return self.tbn(torch.cat(temporal, 1))                               # [B,48,C,T]

    def spatial_features(self, carriers):
        return self.act(self.sbn(self.spatial(carriers))).squeeze(2)         # [B,96,T]

    def tokens(self, features):
        pooled = ceil_mean_pool(features, self.pool)                          # [B,96,N]
        return self.proj(self.drop(pooled.transpose(1, 2)))                  # [B,N,64]

    def read(self, tokens):
        return self.tcn(tokens + self.pos[:, :tokens.shape[1]])

    def classify(self, hidden):
        if self.head_max_norm is not None:
            with torch.no_grad():
                self.head.weight.copy_(torch.renorm(self.head.weight, p=2, dim=0, maxnorm=self.head_max_norm))
        return self.head(hidden)

    def forward(self, x):
        sequence = self.classify(self.read(self.tokens(self.spatial_features(self.carriers(x)))))
        return {"logits": sequence[:, -1], "sequence": sequence}

