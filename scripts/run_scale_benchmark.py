#!/usr/bin/env python3
"""
scripts/run_scale_benchmark.py
===============================
Reproduces the N0 = 10k / 100k / 1M scale-invariance figure using
configs/scale_benchmark.yaml. CPU-only, runs in under a minute.

Usage:
    python scripts/run_scale_benchmark.py [--config configs/scale_benchmark.yaml]
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
from src.analytical import alpha_uncorrected, alpha_corrected_exact, mu_bifurcation, mu_safe, alpha_star, half_life_exact
from src.simulator import run_multi_seed


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    lam = cfg["lambda_fixed"]
    N0_list = cfg["N0_list"]
    seed_counts = {int(k): v for k, v in cfg["seed_counts"].items()}
    alpha0 = cfg["alpha0"]
    mu_mult = cfg["mu_multiplier"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    max_iter = int(np.ceil(9 / lam)) + 100
    print(f"lambda={lam} fixed across all N0 -> max_iter={max_iter}\n")

    iters = np.arange(max_iter + 1)
    summary_rows = []
    results = {}

    for N0 in N0_list:
        q = lam * N0
        mu_b = mu_bifurcation(q, N0)
        mu_s = mu_safe(q, N0)
        mu_used = mu_mult * mu_s
        n_seeds = seed_counts[N0]

        alpha_base_exact = alpha_uncorrected(iters, q, N0, alpha0)
        alpha_corr_exact = alpha_corrected_exact(iters, q, N0, mu_used, alpha0)
        thalf = half_life_exact(q, N0, alpha0)
        astar = alpha_star(q, N0, mu_used)

        hist_base = run_multi_seed(N0, q, mu=0.0, alpha0=alpha0, max_iter=max_iter,
                                    seeds=range(n_seeds))
        hist_corr = run_multi_seed(N0, q, mu=mu_used, alpha0=alpha0, max_iter=max_iter,
                                    seeds=range(1000, 1000 + n_seeds))
        mean_base, std_base = hist_base.mean(axis=0), hist_base.std(axis=0)
        mean_corr, std_corr = hist_corr.mean(axis=0), hist_corr.std(axis=0)

        final_corr = hist_corr[:, -1]
        if n_seeds > 1:
            t_stat, p_val = ttest_1samp(final_corr, 0.5, alternative="less")
        else:
            t_stat, p_val = np.nan, np.nan

        results[N0] = dict(q=q, mu_bifurcation=mu_b, mu_safe=mu_s, mu_used=mu_used, n_seeds=n_seeds,
                            alpha_base_exact=alpha_base_exact, alpha_corr_exact=alpha_corr_exact,
                            mean_base=mean_base, std_base=std_base,
                            mean_corr=mean_corr, std_corr=std_corr)

        summary_rows.append({
            "N0": N0, "lambda": lam, "q": q, "n_seeds": n_seeds,
            "mu_bifurcation": mu_b, "mu_safe": mu_s, "mu_used": mu_used,
            "dynamically_stable": mu_used > mu_b,
            "half_life_iters": thalf, "alpha_star_theory": astar,
            "alpha_corrected_stoch_mean_final": mean_corr[-1],
            "alpha_corrected_stoch_std_final": std_corr[-1],
            "relative_poisson_noise_1_over_sqrtq": 1 / np.sqrt(q),
            "p_value_below_0.5": p_val,
        })
        print(f"N0={N0:>9,}  q={q:>7.0f}  mu_used={mu_used:.5f}  T1/2={thalf:.1f}  "
              f"alpha*={astar:.4f}  (n_seeds={n_seeds})")

    # Overlay figure: proves scale invariance under fixed lambda
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = dict(zip(N0_list, ["#e76f51", "#2a9d8f", "#264653"]))
    for N0 in N0_list:
        r = results[N0]
        ax.plot(iters, r["mean_base"], color=colors[N0], lw=1.6, label=f"Baseline, N0={N0:,}")
        ax.fill_between(iters, r["mean_base"] - r["std_base"], r["mean_base"] + r["std_base"],
                         color=colors[N0], alpha=0.10)
        ax.plot(iters, r["mean_corr"], color=colors[N0], lw=1.6, ls="--", label=f"Corrected, N0={N0:,}")
        ax.fill_between(iters, r["mean_corr"] - r["std_corr"], r["mean_corr"] + r["std_corr"],
                         color=colors[N0], alpha=0.10)
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Failure threshold")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Synthetic fraction α")
    ax.set_title(f"Scale invariance: N0={N0_list} collapse onto the SAME curve (λ={lam} fixed)")
    ax.legend(fontsize=8); ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "scale_invariance_overlay.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("\n" + summary_df.to_string(index=False))
    print(f"\nSaved figure to {fig1_path}")

    if cfg.get("qa_validation", {}).get("enabled", False):
        print("\nqa_validation.enabled=true in config, but this script only runs the "
              "alpha(t) math. Run scripts/run_qa_validation.py separately (needs "
              "network + GPU) -- see README.md for feasibility per scale on Colab.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scale_benchmark.yaml")
    args = parser.parse_args()
    main(args.config)
