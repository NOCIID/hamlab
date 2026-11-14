"""
A SplitJEPA-style split of a learned memory h (memory.py) into a part that only depends on the period
and a part that depends on the economy.

SplitJEPA (Hua, Liu, Zhao, Zheng, 2026, arXiv:2610.12349) splits a latent state into an invariant
block, shared by matched observations, and a variant block. Here the matched observations are the
simulated economies at the same period: the invariant block says "where we are in the episode", the
same for every economy at that period; the variant block is economy-specific (its shock history,
whether its prices run high or low). If a memory mixes the two, economies whose prices differ from the
training economies move h to where the policy GP has no data.

This module implements SplitJEPA's post-hoc spectral split, applied to an already trained GRU: whiten
h; then the directions in which economies at the same period differ least form the invariant block.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitBasis:
    """Whitening of h, and the directions of whitened h sorted from most to least invariant."""

    mean: np.ndarray  # (d,)
    whiten: np.ndarray  # (d, d) ZCA whitening matrix
    within_share: np.ndarray  # (d,) ascending: share of each direction's variance that differs across
    #                           economies at the same period (0: a function of the period only)
    directions: np.ndarray  # (d, d) columns: the matching directions in whitened coordinates


def within_period_split(H, eps: float = 1e-8) -> SplitBasis:
    """
    H: memory states (T, N, d), several economies per period. In whitened coordinates every
    direction has unit total variance, so an eigenvalue of the within-period covariance (the
    covariance of each economy's deviation from that period's mean) is the share of that
    direction's variance that is economy-specific; 1 minus it is the share explained by the period.
    This is SplitJEPA's spectral step: half the covariance of matched-pair differences.
    """
    T, N, d = H.shape
    X = H.reshape(-1, d).astype(np.float64)
    mean = X.mean(0)
    vals, vecs = np.linalg.eigh(np.cov(X, rowvar=False, bias=True))
    whiten = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 0.0) + eps)) @ vecs.T
    Hw = (H.astype(np.float64) - mean) @ whiten
    dev = Hw - Hw.mean(axis=1, keepdims=True)
    within = np.einsum("tni,tnj->ij", dev, dev) / (T * N)
    share, directions = np.linalg.eigh(within)
    return SplitBasis(mean, whiten, np.clip(share, 0.0, 1.0), directions)


def project(basis: SplitBasis, H, n_invariant: int):
    """The n_invariant most invariant coordinates of h: (..., d) -> (..., n_invariant)."""
    return ((np.asarray(H, np.float64) - basis.mean) @ basis.whiten) @ basis.directions[:, :n_invariant]
