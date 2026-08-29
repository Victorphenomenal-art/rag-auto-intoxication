#!/usr/bin/env python3
"""
scripts/run_phase_transition.py
================================
Reproduces the N0 = 23 / 100 / 1000 phase-transition figure using
configs/phase_transition.yaml. CPU-only, runs in well under a minute.

Usage:
    python scripts/run_phase_transition.py [--config configs/phase_transition.yaml]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from scipy.stats import ttest_1samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytical import (
    alpha_uncorrected, alpha_corrected_exact, mu_bifurcation, mu_safe, alpha_star, half_life_exact,
)
from src.simulator import run_multi_seed


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    N0_list = cfg["N0_list"]
    q = cfg["q"]
    alpha0 = cfg["alpha0"]
    seeds = list(range(cfg["seeds"]))
    mu_multipliers = cfg["mu_multipliers"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    t90_largest = 9 * max(N0_list) / q
    max_iter = int(np.ceil(t90_largest * cfg.get("max_iter_margin", 1.1)))
    print(f"max_iter = {max_iter} (N0={max(N0_list)} reaches alpha=0.9 at t~{t90_largest:.1f})")

    iters = np.arange(max_iter + 1)
    summary_rows = []
    fig, axes = plt.subplots(1, len(N0_list), figsize=(19, 5.5), sharey=True)

    for idx, N0 in enumerate(N0_list):
        mu_b = mu_bifurcation(q, N0)  # dynamical bifurcation point (q/N0)
        mu_s = mu_safe(q, N0)         # operational safety threshold (2q/N0)
        ax = axes[idx]

        hist_base = run_multi_seed(N0, q, mu=0.0, alpha0=alpha0, max_iter=max_iter, seeds=seeds)
        mean_base, std_base = hist_base.mean(axis=0), hist_base.std(axis=0)
        exact_base = alpha_uncorrected(iters, q, N0, alpha0)

        ax.plot(iters, mean_base, color="crimson", lw=2, label=f"Baseline (stoch. mean, n={len(seeds)})")
        ax.fill_between(iters, mean_base - std_base, mean_base + std_base, color="crimson", alpha=0.15)
        ax.plot(iters, exact_base, "k--", lw=1.2, label="Baseline exact (Eq.3)")

        colors = ["#f4a300", "#2a9d8f", "#264653"]
        for mu_mult, c in zip(mu_multipliers, colors):
            mu_val = mu_mult * mu_s
            hist_corr = run_multi_seed(N0, q, mu=mu_val, alpha0=alpha0, max_iter=max_iter, seeds=seeds)
            mean_corr, std_corr = hist_corr.mean(axis=0), hist_corr.std(axis=0)
            exact_corr = alpha_corrected_exact(iters.astype(float), q, N0, mu_val, alpha0)
            a_star = alpha_star(q, N0, mu_val)
            stabilizes = mu_val > mu_b  # dynamically stable (finite alpha*), regardless of safety

            ax.plot(iters, mean_corr, color=c, lw=1.8, label=f"μ={mu_mult:.2f}×μ_safe (stoch. mean)")
            ax.fill_between(iters, mean_corr - std_corr, mean_corr + std_corr, color=c, alpha=0.12)
            ax.plot(iters, exact_corr, color=c, ls="--", lw=0.9)

            final_corr = hist_corr[:, -1]
            t_stat, p_val = ttest_1samp(final_corr, 0.5, alternative="less")
            summary_rows.append({
                "N0": N0, "q": q, "mu_multiplier": mu_mult, "mu": mu_val,
                "mu_bifurcation": mu_b, "mu_safe": mu_s,
                "dynamically_stable": stabilizes,
                "alpha_star_theory": a_star,
                "final_alpha_stoch_mean": mean_corr[-1],
                "final_alpha_stoch_std": std_corr[-1],
                "final_alpha_exact": exact_corr[-1],
                "p_value_below_0.5": p_val,
            })

        ax.axhline(0.5, color="gray", ls=":", lw=1, label="Failure threshold")
        ax.set_title(f"N0={N0}   μ_bif={mu_b:.4f}   μ_safe={mu_s:.4f}   T½={half_life_exact(q, N0):.1f}")
        ax.set_xlabel("Iteration")
        if idx == 0:
            ax.set_ylabel("Synthetic fraction α")
        ax.legend(fontsize=6.5, loc="center right")
        ax.set_ylim(-0.02, 1.05)

    plt.suptitle(f"Phase Transition: N0={N0_list}, q={q}, {len(seeds)} seeds, {max_iter} iterations")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "phase_transition.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(summary_df.to_string(index=False))
    print(f"\nSaved figure to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase_transition.yaml")
    args = parser.parse_args()
    main(args.config)
