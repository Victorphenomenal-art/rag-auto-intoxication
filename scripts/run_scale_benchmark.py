#!/usr/bin/env python3
"""
scripts/run_scale_benchmark.py
===============================
Reproduces the N0 = 10k / 100k / 1M scale-invariance figure using
configs/scale_benchmark.yaml. This is the alpha(t)-only part -- CPU-only,
runs in well under a minute total, including N0=1,000,000, because no
document text/embeddings/index are needed to compute alpha(t) (see
src/analytical.py and src/simulator.py).

The QA/FAISS validation (real retrieval + generation) is a SEPARATE,
optional script -- run_qa_validation.py -- gated behind
qa_validation.enabled in the config, since it needs network + ideally GPU.
This script does not call it.

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

try:
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # __file__ isn't defined when this code runs from a notebook cell
    # (pasted directly, or via exec()) rather than as an actual script
    # file -- fall back to the current working directory, which is the
    # repo root if you've already `%cd`'d into it (as Cell 1 does).
    _repo_root = os.getcwd()
sys.path.insert(0, _repo_root)
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

    # Overlay figure: proves scale invariance under fixed lambda.
    #
    # The N0 curves are numerically identical to <1e-6 by design (that IS
    # the scale-invariance result) -- so distinct colors/linestyles alone
    # aren't enough for a reader to tell all three N0 are actually
    # plotted: whichever series draws last simply occludes the other two,
    # since they land on the exact same pixels. Adding marker shapes
    # doesn't fix this either -- if the underlying data coincides, the
    # markers coincide too. Two things actually address it:
    #   1. Report the real, quantitative agreement (computed BEFORE any
    #      visual adjustment below, so it reflects the actual data).
    #   2. A small, explicitly-labeled vertical display offset per N0 --
    #      standard practice for visually separating numerically-identical
    #      curves, as long as the offset is disclosed and the true
    #      deviation is stated separately (which it is, both in the
    #      caption and in summary.csv).
    base_arrays = np.array([results[N0]["mean_base"] for N0 in N0_list])
    corr_arrays = np.array([results[N0]["mean_corr"] for N0 in N0_list])
    max_dev_base = float(np.max(base_arrays.max(axis=0) - base_arrays.min(axis=0)))
    max_dev_corr = float(np.max(corr_arrays.max(axis=0) - corr_arrays.min(axis=0)))
    print(f"Max cross-N0 deviation -- baseline: {max_dev_base:.2e}  corrected: {max_dev_corr:.2e}")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = dict(zip(N0_list, ["#e76f51", "#2a9d8f", "#264653"]))
    marker_styles = ['o', 's', '^']
    DISPLAY_OFFSET = 0.012  # purely visual; see caption. Real curves overlap to <1e-6.
    markevery = max(1, max_iter // 12)
    for idx, N0 in enumerate(N0_list):
        r = results[N0]
        offset = idx * DISPLAY_OFFSET
        offset_note = f" (+{offset:.3f} display offset)" if offset else ""
        ax.plot(iters, r["mean_base"] + offset, color=colors[N0], lw=1.6,
                marker=marker_styles[idx], markevery=markevery, markersize=5,
                label=f"Baseline, N0={N0:,}{offset_note}")
        ax.fill_between(iters, r["mean_base"] + offset - r["std_base"],
                         r["mean_base"] + offset + r["std_base"], color=colors[N0], alpha=0.10)
        ax.plot(iters, r["mean_corr"] + offset, color=colors[N0], lw=1.6, ls="--",
                marker=marker_styles[idx], markevery=markevery, markersize=5, fillstyle="none",
                label=f"Corrected, N0={N0:,}{offset_note}")
        ax.fill_between(iters, r["mean_corr"] + offset - r["std_corr"],
                         r["mean_corr"] + offset + r["std_corr"], color=colors[N0], alpha=0.10)
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Failure threshold")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Synthetic fraction \u03b1 (offset for display, see caption)")
    ax.set_title(f"Scale invariance: N0={N0_list} collapse onto the SAME curve (\u03bb={lam} fixed)")
    ax.text(0.02, 0.02,
            f"Curves offset vertically by up to {(len(N0_list)-1)*DISPLAY_OFFSET:.3f} for visual "
            f"separation only.\nActual max deviation across N0 -- baseline: {max_dev_base:.1e}, "
            f"corrected: {max_dev_corr:.1e}.",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
    ax.legend(fontsize=7); ax.set_ylim(-0.02, 1.05 + (len(N0_list)-1)*DISPLAY_OFFSET)
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
    # parse_known_args, not parse_args: when this code runs inside a
    # Colab/Jupyter kernel directly (pasted into a cell, or exec()'d)
    # rather than via `!python script.py`, sys.argv contains the
    # kernel's own `-f <connection-file>.json` argument. Plain
    # parse_args() treats that as an unrecognized argument and exits
    # with SystemExit(2); parse_known_args() ignores what it doesn't
    # recognize instead of failing on it.
    args, _unrecognized = parser.parse_known_args()
    main(args.config)
