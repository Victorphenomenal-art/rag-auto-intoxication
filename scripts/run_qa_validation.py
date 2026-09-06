
qa_script_content = '''#!/usr/bin/env python3
"""
scripts/run_qa_validation.py
Updated version with --n0, --all_scales, and n_test_items=500 for the full run.
"""
import argparse
import os
import sys
import time
import numpy as np
import yaml

sys.path.insert(0, os.getcwd())
from src.analytical import mu_safe
from src.qa_evaluator import DocumentGenerator, RAGQAEvaluator, make_proxy_corpus

def build_kb_at_checkpoint(N0, alpha_target, base_docs, synth_pool):
    n_synth = int(round(alpha_target * N0))
    n_human = N0 - n_synth
    docs = list(base_docs[:n_human])
    if n_synth > 0:
        reps = (n_synth // max(len(synth_pool), 1)) + 1
        docs += (synth_pool * reps)[:n_synth]
    return docs

def build_base_docs_from_squad(test_set, total_docs, seed=0, extra_split="validation"):
    gold_contexts = []
    seen = set()
    for item in test_set:
        c = item["context"]
        if c not in seen:
            seen.add(c)
            gold_contexts.append(c)
    n_gold = len(gold_contexts)
    if n_gold > total_docs:
        print(f"  WARNING: total_docs={total_docs} < n_gold={n_gold} -- truncating gold contexts.")
        return gold_contexts[:total_docs], total_docs
    n_distractors_needed = total_docs - n_gold
    rng = np.random.default_rng(seed)
    from datasets import load_dataset
    full_split = load_dataset("rajpurkar/squad", split=extra_split)
    all_contexts = list(dict.fromkeys(full_split["context"]))
    distractor_pool = [c for c in all_contexts if c not in seen]
    if len(distractor_pool) >= n_distractors_needed:
        idx = rng.choice(len(distractor_pool), size=n_distractors_needed, replace=False)
        distractors = [distractor_pool[i] for i in idx]
    else:
        print(f"  WARNING: only {len(distractor_pool)} unique distractor contexts available, repeating...")
        reps = (n_distractors_needed // max(len(distractor_pool), 1)) + 1
        distractors = (distractor_pool * reps)[:n_distractors_needed]
    base_docs = gold_contexts + distractors
    return base_docs, n_gold

def run_for_scale(N0, cfg, smoke_test=False):
    qa_cfg = cfg["qa_validation"]
    lam = cfg["lambda_fixed"]
    q = lam * N0
    mu_s = mu_safe(q, N0)
    mu_used = cfg["mu_multiplier"] * mu_s
    index_type = qa_cfg["index_type_by_scale"][N0]
    gen_mode = qa_cfg["generation_mode_by_scale"][N0]
    total_docs = 50 if smoke_test else N0
    n_test_items = 3 if smoke_test else 500
    evaluator = RAGQAEvaluator(eval_model_name=qa_cfg["eval_model"], n_test_items=n_test_items)

    base_docs, n_gold = build_base_docs_from_squad(evaluator.test_set, total_docs, seed=42)
    print(f"  Base corpus: {n_gold} gold contexts + {len(base_docs)-n_gold} distractors (target {total_docs}).")

    generator = DocumentGenerator(qa_cfg["generator_model"])
    n_unique_synth = 5 if smoke_test else qa_cfg["proxy_unique_docs"]
    unique_synth = generator.generate("General knowledge passage.", n=n_unique_synth)

    fracs = qa_cfg["checkpoints_fraction"]
    results = []
    for frac in fracs:
        if gen_mode == "proxy":
            synth_pool = make_proxy_corpus(unique_synth, target_size=max(1, int(frac * total_docs)))
        else:
            synth_pool = unique_synth
        docs = build_kb_at_checkpoint(total_docs, frac, base_docs, synth_pool)
        em, f1 = evaluator.evaluate_kb(docs, index_type=index_type,
                                        ivf_nlist=qa_cfg["ivf_nlist"],
                                        pq_m=qa_cfg["pq_m"], pq_nbits=qa_cfg["pq_nbits"])
        results.append({"N0": N0, "alpha_checkpoint": frac, "index_type": index_type,
                         "generation_mode": gen_mode, "EM": em, "F1": f1})
        print(f"  N0={N0} alpha~{frac:.2f} index={index_type} gen={gen_mode}  EM={em:.3f} F1={f1:.3f}")

    # Defensive config handling for ivfpq_vs_flat_control
    control_cfg = qa_cfg.get("ivfpq_vs_flat_control") or {}
    control_scale = control_cfg.get("run_at_N0", 100_000)
    if (not smoke_test and control_cfg.get("enabled", False)
            and index_type != "flat" and N0 == control_scale):
        print(f"  Running Flat-vs-{index_type} control check at N0={N0}...")
        docs = build_kb_at_checkpoint(total_docs, 0.5, base_docs, unique_synth)
        em_a, f1_a = evaluator.evaluate_kb(docs, index_type="flat")
        em_b, f1_b = evaluator.evaluate_kb(docs, index_type=index_type,
                                            ivf_nlist=qa_cfg["ivf_nlist"],
                                            pq_m=qa_cfg["pq_m"], pq_nbits=qa_cfg["pq_nbits"])
        tol = control_cfg.get("tolerance_pct", 5.0) / 100.0
        em_diff = abs(em_a - em_b) / max(em_a, 1e-9)
        f1_diff = abs(f1_a - f1_b) / max(f1_a, 1e-9)
        ok = em_diff <= tol and f1_diff <= tol
        print(f"  Flat: EM={em_a:.3f} F1={f1_a:.3f}  |  {index_type}: EM={em_b:.3f} F1={f1_b:.3f}  "
              f"|  within {tol*100:.0f}%? {ok}")
        if not ok:
            print(f"  WARNING: {index_type} deviates from Flat by more than {tol*100:.0f}% -- "
                  f"treat N0={N0} ANN results with caution.")
    return results

def main(config_path, smoke_test=False, n0=10000, all_scales=False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not cfg["qa_validation"]["enabled"] and not smoke_test:
        print("qa_validation.enabled=false. Set to true or pass --smoke_test.")
        return
    out_dir = os.path.join(cfg["output_dir"], "qa_validation")
    os.makedirs(out_dir, exist_ok=True)

    if smoke_test:
        scales = [10_000]
    elif all_scales:
        scales = cfg["N0_list"]
    else:
        qa_cfg = cfg["qa_validation"]
        has_index = n0 in qa_cfg.get("index_type_by_scale", {})
        has_gen = n0 in qa_cfg.get("generation_mode_by_scale", {})
        if not (has_index and has_gen):
            print(f"ERROR: N0={n0} not configured.")
            return
        scales = [n0]

    print(f"Running scale(s): {scales}")
    all_results = []
    for N0 in scales:
        t0 = time.time()
        all_results += run_for_scale(N0, cfg, smoke_test=smoke_test)
        print(f"  N0={N0} done in {time.time()-t0:.1f}s")

    import pandas as pd
    pd.DataFrame(all_results).to_csv(os.path.join(out_dir, "qa_validation_results.csv"), index=False)
    print(f"\\nSaved to {out_dir}/qa_validation_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scale_benchmark.yaml")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--n0", type=int, default=10000)
    parser.add_argument("--all_scales", action="store_true")
    args, unknown = parser.parse_known_args()
    main(args.config, smoke_test=args.smoke_test, n0=args.n0, all_scales=args.all_scales)
'''

with open("scripts/run_qa_validation.py", "w") as f:
    f.write(qa_script_content)
print("✅ Updated run_qa_validation.py with --n0 support and 500 test items.")

# -------------------- 5. FIX THE YAML CONFIG USING PYTHON (SAFE) --------------------
print("Fixing configs/scale_benchmark.yaml using Python...")
with open("configs/scale_benchmark.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Ensure the missing block exists
if "ivfpq_vs_flat_control" not in cfg.get("qa_validation", {}):
    cfg["qa_validation"]["ivfpq_vs_flat_control"] = {
        "enabled": False,
        "run_at_N0": 100000,
        "tolerance_pct": 5.0
    }

# Enable QA and set N0_list to only 10000
cfg["qa_validation"]["enabled"] = True
cfg["N0_list"] = [10000]

# Write back with proper formatting
with open("configs/scale_benchmark.yaml", "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, indent=2)
print("✅ Config updated correctly.")

# -------------------- 6. RUN SMOKE TEST (Optional) --------------------
print("\n" + "="*60)
print("RUNNING QA VALIDATION SMOKE TEST (5 docs, 3 questions)")
print("="*60)
!python scripts/run_qa_validation.py --smoke_test

# -------------------- 7. RUN FULL QA VALIDATION (500 test items, N0=10,000) --------------------
print("\n" + "="*60)
print("RUNNING FULL QA VALIDATION (N0=10,000, 500 test items)")
print("="*60)
!python scripts/run_qa_validation.py --n0 10000

# -------------------- 8. DOWNLOAD RESULTS --------------------
from google.colab import files
csv_path = "results/scale_benchmark/qa_validation/qa_validation_results.csv"
if os.path.exists(csv_path):
    files.download(csv_path)
    print("\n✅ Results downloaded: qa_validation_results.csv")
else:
    print("\n⚠️ CSV not found. Check the script output above for errors.")

print("\n✅ Praise's QA validation complete!")
print("📊 The CSV contains EM/F1 for 500 test questions across α checkpoints.")
