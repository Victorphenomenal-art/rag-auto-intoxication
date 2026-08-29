#!/usr/bin/env python3
"""
scripts/run_operational_validation.py
=======================================

Run on CPU (N0=100, small). No GPU needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.simulator import DocumentLevelSimulator
from src.eviction import RandomEviction, ProvenanceEviction
from src.analytical import mu_safe, alpha_star, alpha_corrected_exact

# --- Configuration ---
N0 = 100
q = 7
alpha0 = 0.0
max_iter = 200
seeds = range(10)  # 10 seeds for statistical variance

# Safety threshold
mu_s = mu_safe(q, N0)          # 0.14
mu_used = 1.15 * mu_s           # 0.161 (safe regime)

# Detection delay to test (0 = ideal, 5 = realistic lag)
detection_delay = 5

# --- Run Simulations ---
results = {}

for policy_name, policy_class in [("Random", RandomEviction), ("Provenance", ProvenanceEviction)]:
    for delay in [0, detection_delay]:
        trajectories = []
        for seed in seeds:
            # Initialize policy
            if policy_name == "Random":
                policy = RandomEviction(rate=mu_used, rng=np.random.default_rng(seed))
            else:  # Provenance
                # Gamma=0.5, threshold=0.5 (evicts docs with provenance weight < 0.5)
                policy = ProvenanceEviction(gamma=0.5, weight_threshold=0.5, max_rate=mu_used)
            
            # Run document-level simulator (small N0, exact eviction)
            sim = DocumentLevelSimulator(N0=N0, mean_q=q, policy=policy, alpha0=alpha0, seed=seed)
            hist = sim.run(max_iter, detection_delay=delay)
            trajectories.append(hist)
        
        # Average across seeds
        mean_alpha = np.mean(trajectories, axis=0)
        std_alpha = np.std(trajectories, axis=0)
        
        key = f"{policy_name}_delay{delay}"
        results[key] = {"mean": mean_alpha, "std": std_alpha}
        print(f"{key}: Final alpha = {mean_alpha[-1]:.4f} ± {std_alpha[-1]:.4f}")

# --- Plot Comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Effect of Policy (at delay=0)
ax1.plot(results["Random_delay0"]["mean"], label="RandomEviction (Ideal)", color="blue", lw=2)
ax1.fill_between(range(max_iter+1), 
                 results["Random_delay0"]["mean"] - results["Random_delay0"]["std"],
                 results["Random_delay0"]["mean"] + results["Random_delay0"]["std"], 
                 color="blue", alpha=0.2)
ax1.plot(results["Provenance_delay0"]["mean"], label="ProvenanceEviction (Realistic)", color="orange", lw=2)
ax1.fill_between(range(max_iter+1), 
                 results["Provenance_delay0"]["mean"] - results["Provenance_delay0"]["std"],
                 results["Provenance_delay0"]["mean"] + results["Provenance_delay0"]["std"], 
                 color="orange", alpha=0.2)
ax1.axhline(0.5, color="gray", ls="--", label="Failure threshold")
ax1.axhline(alpha_star(q, N0, mu_used), color="green", ls=":", label=f"Theory α* = {alpha_star(q, N0, mu_used):.3f}")
ax1.set_title("Effect of Eviction Policy (No Detection Delay)")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Synthetic fraction α")
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Effect of Detection Delay (using ProvenanceEviction)
for delay in [0, detection_delay]:
    key = f"Provenance_delay{delay}"
    label = f"Delay = {delay} iters"
    ax2.plot(results[key]["mean"], label=label, lw=2)
    ax2.fill_between(range(max_iter+1), 
                     results[key]["mean"] - results[key]["std"],
                     results[key]["mean"] + results[key]["std"], 
                     alpha=0.2)
ax2.axhline(0.5, color="gray", ls="--", label="Failure threshold")
ax2.axhline(alpha_star(q, N0, mu_used), color="green", ls=":", label=f"Theory α* = {alpha_star(q, N0, mu_used):.3f}")
ax2.set_title("Effect of Detection Delay (ProvenanceEviction)")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Synthetic fraction α")
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle(f"Operational Validation: N0={N0}, q={q}, μ={mu_used:.3f} (1.15×μ_safe)")
plt.tight_layout()

# Save figure
import os
os.makedirs("results/operational_validation", exist_ok=True)
plt.savefig("results/operational_validation/operational_validation.png", dpi=200)
print("\n✅ Figure saved to results/operational_validation/operational_validation.png")
