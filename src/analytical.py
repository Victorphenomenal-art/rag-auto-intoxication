"""
src/analytical.py
==================
Exact analytical solutions for Eqs (1)-(12) of the RAG epistemic-collapse
framework. No document text, embeddings, or index is needed here -- this
module is pure math and is safe/cheap to run at any N0, including 1e6+.

All formulas are verified against numerical integration (scipy.integrate
.odeint) in tests/test_analytical.py -- run `pytest tests/` before trusting
any figure generated downstream of this module.
"""

import numpy as np


def alpha_uncorrected(t, q, N0, alpha0=0.0):
    """
    Eq (3): exact transient solution for the uncorrected (growing) corpus.
        alpha(t) = 1 - (1-alpha0) * N0 / (N0 + q*t)
    """
    return 1 - (1 - alpha0) * N0 / (N0 + q * t)


def alpha_corrected_exact(t, q, N0, mu, alpha0=0.0):
    """
    Exact closed-form solution of Eq (8):
        dalpha/dt = (q/N0)(1-alpha) - mu*alpha*(1-alpha)

    Derived by factoring the RHS as mu*(alpha-1)*(alpha - lam/mu), lam=q/N0,
    then separating variables and solving via partial fractions.

    NOTE 1: an earlier draft of this formula had mu and lam swapped in the
    denominator, which made the solution diverge to alpha->1 even when mu
    was safely above the critical threshold.

    NOTE 2: evaluated naively, exp((mu-lam)*t) overflows for large t when
    mu > lam (the very regime we care about, since it's the stable one) --
    the ratio itself converges to lam/mu, but computing numerator and
    denominator separately as huge floats first causes inf/inf = nan. We
    avoid this by re-expressing the ratio in terms of exp(-(mu-lam)*t)
    (which -> 0, not inf) whenever mu > lam, and only using the direct
    form when mu <= lam (where the direct exponent is <= 0 and safe).

    Verified against odeint to < 1e-6 for finite t, and confirmed to
    converge to alpha* = q/(mu*N0) as t -> inf without overflow, in
    tests/test_analytical.py.
    """
    t = np.asarray(t, dtype=float)
    lam = q / N0
    diff = mu - lam

    if np.isclose(mu, lam):
        # Degenerate case mu == lam: dalpha/dt = lam*(1-alpha)^2
        return 1 - (1 - alpha0) / (1 + lam * (1 - alpha0) * t)

    b = lam / mu
    if alpha0 == 0.0:
        K = 1.0 / b  # = (0-1)/(0-b)
    else:
        K = (alpha0 - 1) / (alpha0 - b)

    if diff > 0:
        # Stable/corrected regime (mu > lam): use exp(-diff*t) -> 0, no overflow.
        neg = np.exp(-diff * t)
        return (neg - K * b) / (neg - K)
    else:
        # diff <= 0: direct exponent is <= 0, bounded in (0, 1], safe as-is.
        exp_term = np.exp(diff * t)
        num = 1 - K * b * exp_term
        den = 1 - K * exp_term
        return num / den


def alpha_corrected_ode_rhs(alpha, t, q, N0, mu):
    """RHS of Eq (8), for use with scipy.integrate.odeint (validation only)."""
    return (q / N0) * (1 - alpha) - mu * alpha * (1 - alpha)


def mu_bifurcation(q, N0):
    """The actual dynamical bifurcation point (q/N0). Below this, the only
    fixed point in [0,1] is alpha=1. mu_safe (2q/N0) is where the
    stabilized alpha* crosses the 0.5 operational threshold, not where
    the qualitative dynamics change."""
    return q / N0


def alpha_star(q, N0, mu):
    """Eq (9): steady-state synthetic fraction under correction."""
    return q / (mu * N0)


def half_life_exact(q, N0, alpha0=0.0):
    """Eq (6): exact number of iterations to reach alpha = 0.5."""
    return (N0 / q) * (1 - 2 * alpha0)
