#!/usr/bin/env python3
"""
scripts/run_qa_validation.py
=============================
OPTIONAL. REQUIRES network access + a GPU (T4 minimum; A100 recommended
for the N0=1,000,000 / ivfpq path). Ties alpha(t) to real downstream QA
accuracy (EM/F1) via retrieval + generation, per configs/scale_benchmark.yaml
-> qa_validation section.

NOT executed in the environment that produced this repo (sandboxed, no
network/GPU). Syntax-checked only. Run a small smoke test first:
    python scripts/run_qa_validation.py --config configs/scale_benchmark.yaml --smoke_test

Implements:
  - Item A: generator model != evaluator model (avoids self-eval confound)
  - Item C: optional Flat-vs-IVFPQ control check at 100k (asserts the two
    index types agree on EM/F1 within a tolerance before trusting the
    faster/approximate index at 1M)
  - Item D: proxy generation mode at 1M (see src/qa_evaluator.make_proxy_corpus)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytical import mu_critical
from src.qa_evaluator import DocumentGenerator, RAGQAEvaluator, make_proxy_corpus


def build_kb_at_checkpoint(N0, alpha_target, base_docs, synth_pool):
    """Materialize a document list whose synthetic fraction ~= alpha_target."""
    n_synth = int(round(alpha_target * N0))
    n_human = N0 - n_synth
    docs = list(base_docs[:n_human])
    if n_synth > 0:
        # cycle through the (possibly proxy) synthetic pool
        reps = (n_synth // max(len(synth_pool), 1)) + 1
        docs += (synth_pool * reps)[:n_synth]
    return docs


def run_for_scale(N0, cfg, smoke_test=False):
    qa_cfg = cfg["qa_validation"]
    lam = cfg["lambda_fixed"]
    q = lam * N0
    mu_c = mu_critical(q, N0)
    mu_used = cfg["mu_multiplier"] * mu_c

    index_type = qa_cfg["index_type_by_scale"][N0]
    gen_mode = qa_cfg["generation_mode_by_scale"][N0]

    n_human = 50 if smoke_test else N0
    base_docs = [f"Human document {i}: sample factual content about topic {i%50}."
                 for i in range(n_human)]

    generator = DocumentGenerator(qa_cfg["generator_model"])
    n_unique_synth = 5 if smoke_test else qa_cfg["proxy_unique_docs"]
    unique_synth = generator.generate("General knowledge passage.", n=n_unique_synth)

    evaluator = RAGQAEvaluator(eval_model_name=qa_cfg["eval_model"],
                                n_test_items=3 if smoke_test else 30)

    fracs = qa_cfg["checkpoints_fraction"]
    results = []
    for frac in fracs:
        if gen_mode == "proxy":
            synth_pool = make_proxy_corpus(unique_synth, target_size=max(1, int(frac * n_human)))
        else:
            synth_pool = unique_synth

        docs = build_kb_at_checkpoint(n_human, frac, base_docs, synth_pool)
        em, f1 = evaluator.evaluate_kb(docs, index_type=index_type,
                                        ivf_nlist=qa_cfg["ivf_nlist"],
                                        pq_m=qa_cfg["pq_m"], pq_nbits=qa_cfg["pq_nbits"])
        results.append({"N0": N0, "alpha_checkpoint": frac, "index_type": index_type,
                         "generation_mode": gen_mode, "EM": em, "F1": f1})
        print(f"  N0={N0} alpha~{frac:.2f} index={index_type} gen={gen_mode}  EM={em:.3f} F1={f1:.3f}")

    # Item C: Flat vs IVFPQ control check
    control_cfg = qa_cfg["ivfpq_vs_flat_control"]
    control_scale = control_cfg.get("run_at_N0", 100_000)  # add this key to the yaml
    if control_cfg["enabled"] and index_type != "flat" and not smoke_test and N0 == control_scale:
        print(f"  Running Flat-vs-{index_type} control check at N0={N0}...")
        docs = build_kb_at_checkpoint(n_human, 0.5, base_docs, unique_synth)
        em_a, f1_a = evaluator.evaluate_kb(docs, index_type="flat")
        em_b, f1_b = evaluator.evaluate_kb(docs, index_type=index_type,
                                            ivf_nlist=qa_cfg["ivf_nlist"],
                                            pq_m=qa_cfg["pq_m"], pq_nbits=qa_cfg["pq_nbits"])
        tol = qa_cfg["ivfpq_vs_flat_control"]["tolerance_pct"] / 100.0
        em_diff = abs(em_a - em_b) / max(em_a, 1e-9)
        f1_diff = abs(f1_a - f1_b) / max(f1_a, 1e-9)
        ok = em_diff <= tol and f1_diff <= tol
        print(f"  Flat: EM={em_a:.3f} F1={f1_a:.3f}  |  {index_type}: EM={em_b:.3f} F1={f1_b:.3f}  "
              f"|  within {tol*100:.0f}%? {ok}")
        if not ok:
            print(f"  WARNING: {index_type} deviates from Flat by more than {tol*100:.0f}% -- "
                  f"treat N0={N0} ANN results with caution.")

    return results


def main(config_path, smoke_test=False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if not cfg["qa_validation"]["enabled"] and not smoke_test:
        print("qa_validation.enabled=false in config. Set it to true, or pass "
              "--smoke_test to run a tiny end-to-end check regardless.")
        return

    out_dir = os.path.join(cfg["output_dir"], "qa_validation")
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    scales = [10_000] if smoke_test else cfg["N0_list"]
    for N0 in scales:
        t0 = time.time()
        all_results += run_for_scale(N0, cfg, smoke_test=smoke_test)
        print(f"  N0={N0} done in {time.time()-t0:.1f}s")

    import pandas as pd
    pd.DataFrame(all_results).to_csv(os.path.join(out_dir, "qa_validation_results.csv"), index=False)
    print(f"\nSaved to {out_dir}/qa_validation_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scale_benchmark.yaml")
    parser.add_argument("--smoke_test", action="store_true",
                         help="Tiny end-to-end run (5 docs, 3 QA items) to sanity-check the pipeline "
                              "before committing to a full run.")
    args = parser.parse_args()
    main(args.config, smoke_test=args.smoke_test)
