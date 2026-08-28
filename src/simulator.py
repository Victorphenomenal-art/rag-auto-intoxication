"""
src/simulator.py
=================
Two simulation modes, chosen for cost/fidelity trade-offs:

1. CountSimulator (default, used for ALL N0 including 1e6):
   Tracks only two floats (S_t, N_t) or a single alpha value per step.
   No document objects are created. This is what makes N0=1,000,000
   feasible on a laptop or free Colab CPU runtime -- the "40 hour FAISS
   rebuild" problem some drafts warn about only applies if you insist on
   materializing every document, which the closed-form math never
   requires.

2. DocumentLevelSimulator (small N0 only, e.g. <= ~5,000):
   Actually instantiates Document objects and applies a real
   src.eviction.EvictionPolicy each iteration. Use this to check how much
   a "real" policy (e.g. ProvenanceEviction) deviates from the idealized
   random-chemostat assumption baked into Eq (8)-(10). This is O(N0) per
   iteration and is NOT meant to be run at 100k/1M -- the CountSimulator
   is the one to use for the scale-up figures.

Both support an optional `detection_delay` for the corrected/eviction case.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from .eviction import Document, EvictionPolicy, RandomEviction


@dataclass
class CountSimulator:
    """Scalable, count-only simulator. Safe at any N0."""
    N0: float
    mean_q: float
    mu: float = 0.0
    alpha0: float = 0.0
    detection_delay: int = 0  # only meaningful for the discrete Euler-step mode
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def run(self, max_iter: int) -> np.ndarray:
        """Returns alpha history, shape (max_iter+1,)."""
        history = np.empty(max_iter + 1)
        if self.mu <= 0:
            N = float(self.N0)
            S = self.alpha0 * self.N0
            history[0] = S / N
            for t in range(1, max_iter + 1):
                q_t = self.rng.poisson(self.mean_q)
                S += q_t
                N += q_t
                history[t] = S / N
        else:
            alpha = self.alpha0
            # simple FIFO queue of pending synthetic mass to model detection_delay
            pending = [0.0] * max(self.detection_delay, 1)
            history[0] = alpha
            for t in range(1, max_iter + 1):
                q_t = self.rng.poisson(self.mean_q)
                if self.detection_delay > 0:
                    pending.append(q_t)
                    matured_q = pending.pop(0)  # influx from `detection_delay` steps ago
                else:
                    matured_q = q_t
                # growth uses this step's actual influx; eviction acts only on matured mass
                dalpha_growth = (matured_q / self.N0) * (1 - alpha)  # use matured_q, not q_t
                dalpha_evict = self.mu * alpha * (1 - alpha)          # unconditional

                dalpha = dalpha_growth - dalpha_evict
                alpha = float(np.clip(alpha + dalpha, 0.0, 1.0))
                history[t] = alpha
        return history


def run_multi_seed(N0, mean_q, mu, alpha0, max_iter, seeds, detection_delay=0):
    """Vectorized-over-seeds convenience wrapper. Returns (n_seeds, max_iter+1)."""
    out = np.empty((len(seeds), max_iter + 1))
    for i, s in enumerate(seeds):
        sim = CountSimulator(N0=N0, mean_q=mean_q, mu=mu, alpha0=alpha0,
                              detection_delay=detection_delay, seed=s)
        out[i] = sim.run(max_iter)
    return out


@dataclass
class DocumentLevelSimulator:
    """
    Small-scale (N0 <~ 5,000), document-object simulator that applies a real
    EvictionPolicy each iteration. Intended for validating how far a
    realistic policy (e.g. ProvenanceEviction) diverges from the idealized
    random-chemostat assumption behind Eq (8). O(N0) memory and per-step
    cost -- do not use for 100k/1M runs.
    """
    N0: int
    mean_q: float
    policy: Optional[EvictionPolicy] = None
    gamma: float = 0.5           # provenance decay param, Eq (11), for doc generation gen assignment
    alpha0: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        n_synth0 = int(round(self.alpha0 * self.N0))
        self.docs: List[Document] = (
            [Document(is_synthetic=False) for _ in range(self.N0 - n_synth0)]
            + [Document(is_synthetic=True, generation=1) for _ in range(n_synth0)]
        )
        if self.policy is None:
            self.policy = RandomEviction(rate=0.0, rng=self.rng)

    def alpha(self) -> float:
        if not self.docs:
            return 0.0
        n_synth = sum(1 for d in self.docs if d.is_synthetic)
        return n_synth / len(self.docs)

    def run(self, max_iter: int, detection_delay: int = 0) -> np.ndarray:
        history = np.empty(max_iter + 1)
        history[0] = self.alpha()
        for t in range(1, max_iter + 1):
            q_t = self.rng.poisson(self.mean_q)
            for _ in range(q_t):
                self.docs.append(Document(is_synthetic=True, generation=1, created_at=t))
            evict_idx = set(self.policy.select_for_eviction(self.docs, t, detection_delay))
            if evict_idx:
                self.docs = [d for i, d in enumerate(self.docs) if i not in evict_idx]
            history[t] = self.alpha()
        return history
