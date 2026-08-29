"""
src/eviction.py
================
Operationalizes the abstract 'mu' knob (item B from the critique) as an
actual eviction policy that can be applied to a population of documents,
each carrying a provenance / generational-distance value g (Eq. 11).

Two policies are provided:
  - RandomEviction:      the chemostat baseline (uniform random purge at
                          a fixed rate mu). This is what all simulations
                          in this repo default to -- it's the policy whose
                          dynamics match Eq (8)-(10) exactly.
  - ProvenanceEviction:   purges documents whose provenance weight w(g)
                          (Eq. 11) falls below a threshold, i.e. deep
                          generational echoes are evicted preferentially.
                          This does NOT reduce exactly to Eq (8) -- it's
                          provided so you can empirically compare "real"
                          provenance-based mitigation against the
                          idealized random-chemostat model the closed-form
                          math assumes. Do not expect it to match
                          alpha_corrected_exact() precisely; that's the
                          point of running the comparison.

Supports an optional `detection_delay` (item G): a newly-added synthetic
document only becomes eligible for eviction `detection_delay` iterations
after it was added, modelling the realistic lag between generation and
detection.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class Document:
    """Minimal document record used by the count/flag-level simulator.

    is_synthetic : whether this document is synthetic (vs. original human)
    generation   : provenance depth g (Eq. 11); 0 for human docs
    created_at   : iteration at which the doc was added (for detection delay)
    """
    is_synthetic: bool
    generation: int = 0
    created_at: int = 0


class EvictionPolicy:
    """Base class. Subclasses implement `select_for_eviction`."""

    def select_for_eviction(self, docs: List[Document], current_iter: int,
                             detection_delay: int = 0) -> List[int]:
        raise NotImplementedError

    def _eligible_indices(self, docs, current_iter, detection_delay):
        return [i for i, d in enumerate(docs)
                if d.is_synthetic and (current_iter - d.created_at) >= detection_delay]


class RandomEviction(EvictionPolicy):
    """Chemostat baseline: purge a uniform-random `rate` fraction of
    eligible synthetic documents each iteration. This is the policy whose
    aggregate behaviour matches the closed-form Eq (8)/(9)/(10) exactly,
    since mu in the math IS this random purge rate.
    """

    def __init__(self, rate: float, rng: np.random.Generator = None):
        self.rate = rate
        self.rng = rng or np.random.default_rng()

    def select_for_eviction(self, docs, current_iter, detection_delay=0):
        eligible = self._eligible_indices(docs, current_iter, detection_delay)
        if not eligible:
            return []
        n_evict = int(round(self.rate * len(eligible)))
        if n_evict <= 0:
            return []
        return list(self.rng.choice(eligible, size=min(n_evict, len(eligible)), replace=False))


class ProvenanceEviction(EvictionPolicy):
    """Evicts synthetic documents whose provenance weight w(g) = 1/(1+gamma*g)
    (Eq. 11) falls below `weight_threshold`, i.e. deep-generation echoes are
    targeted first. `max_rate` caps the fraction of eligible docs evicted
    per iteration (models limited eviction throughput/budget).
    """

    def __init__(self, gamma: float, weight_threshold: float, max_rate: float = 1.0):
        self.gamma = gamma
        self.weight_threshold = weight_threshold
        self.max_rate = max_rate

    def _weight(self, g: int) -> float:
        return 1.0 / (1.0 + self.gamma * g)

    def select_for_eviction(self, docs, current_iter, detection_delay=0):
        eligible = self._eligible_indices(docs, current_iter, detection_delay)
        flagged = sorted(
            (i for i in eligible if self._weight(docs[i].generation) < self.weight_threshold),
            key=lambda i: self._weight(docs[i].generation)
        )
        if not flagged:
            return []
        n_cap = int(round(self.max_rate * len(eligible)))
        return flagged[:n_cap] if n_cap > 0 else []
