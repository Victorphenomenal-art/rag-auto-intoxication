"""
tests/test_analytical.py
=========================
Item F from the critique: validate the closed-form solution of Eq (8)
against numerical integration. This is the single test that proves the
analytical shortcut used throughout the repo (and what makes N0=1e6
tractable without simulation) is mathematically correct.

Run with: pytest tests/ -v
"""

import numpy as np
from scipy.integrate import odeint
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytical import (
    alpha_uncorrected, alpha_corrected_exact, alpha_corrected_ode_rhs,
    mu_critical, alpha_star, half_life_exact,
)


@pytest.mark.parametrize("q,N0,mu", [
    (7.0, 23.0, 0.65),
    (7.0, 100.0, 0.16),
    (7.0, 1000.0, 0.017),
    (70.0, 10000.0, 0.0161),
])
def test_corrected_exact_matches_odeint(q, N0, mu):
    t_vals = np.linspace(0, 200, 1000)
    analytic = alpha_corrected_exact(t_vals, q, N0, mu, alpha0=0.0)
    numeric = odeint(alpha_corrected_ode_rhs, 0.0, t_vals, args=(q, N0, mu)).flatten()
    max_err = np.max(np.abs(analytic - numeric))
    assert max_err < 1e-6, f"closed-form vs odeint mismatch: {max_err}"


def test_corrected_converges_to_steady_state():
    q, N0, mu = 7.0, 1000.0, 0.02
    t_far = 1e5
    alpha_far = alpha_corrected_exact(t_far, q, N0, mu, alpha0=0.0)
    theory = alpha_star(q, N0, mu)
    assert abs(alpha_far - theory) < 1e-6


def _rhs_uncorrected(alpha, t, q, N0, alpha0):
    """Eq (4): the TRUE governing ODE for the growing/uncorrected pool.
    NOTE: this is NOT the same as Eq (8) at mu=0 -- Eq (8) assumes a fixed
    pool size N0 (chemostat), whereas Eq (3)/(4) assume the pool itself
    grows by q each step. Comparing Eq (3) against Eq (8)'s mu=0 limit is
    an invalid test (apples-to-oranges); this is the correct comparison.
    """
    return (q / (N0 * (1 - alpha0))) * (1 - alpha) ** 2


def test_uncorrected_matches_its_own_governing_ode():
    q, N0 = 7.0, 100.0
    t_vals = np.linspace(0, 50, 200)
    exact3 = alpha_uncorrected(t_vals, q, N0, alpha0=0.0)
    numeric = odeint(_rhs_uncorrected, 0.0, t_vals, args=(q, N0, 0.0)).flatten()
    assert np.max(np.abs(exact3 - numeric)) < 1e-6


def test_half_life_matches_definition():
    q, N0 = 7.0, 1000.0
    thalf = half_life_exact(q, N0, alpha0=0.0)
    alpha_at_thalf = alpha_uncorrected(thalf, q, N0, alpha0=0.0)
    assert abs(alpha_at_thalf - 0.5) < 1e-9


def test_mu_critical_boundary_behavior():
    # Just below mu_critical: alpha_star computed at mu slightly ABOVE mu_c
    # should still be < 0.5; at mu == mu_c, alpha_star should equal 0.5.
    q, N0 = 7.0, 1000.0
    mu_c = mu_critical(q, N0)
    assert abs(alpha_star(q, N0, mu_c) - 0.5) < 1e-9
    assert alpha_star(q, N0, mu_c * 1.5) < 0.5
    assert alpha_star(q, N0, mu_c * 0.5) > 0.5


def test_scale_invariance_under_fixed_lambda():
    # Core claim used in the scale-up experiments: holding lambda = q/N0
    # fixed, alpha_star and half-life depend ONLY on lambda, not on N0.
    lam = 0.007
    mu_mult = 1.15
    vals = []
    for N0 in [1e4, 1e5, 1e6]:
        q = lam * N0
        mu = mu_mult * mu_critical(q, N0)
        vals.append((half_life_exact(q, N0), alpha_star(q, N0, mu)))
    thalfs = [v[0] for v in vals]
    astars = [v[1] for v in vals]
    assert max(thalfs) - min(thalfs) < 1e-6
    assert max(astars) - min(astars) < 1e-9
