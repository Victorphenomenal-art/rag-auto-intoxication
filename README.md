# RAG Auto-Intoxication: Scaling Laws for Epistemic Collapse

Research-grade implementation of the closed-form dynamics governing
synthetic-content contamination in a growing / actively-purged retrieval
corpus (Eqs. 1-12).

## What's proven here

- **Eq (3)** exact transient solution for the uncorrected (growing) corpus.
- **Eq (8)** exact closed-form solution for the corrected (chemostat) system
  — verified against `scipy.integrate.odeint` to <1e-6 in `tests/`.
- **Eq (10)** phase transition at `mu = 2q/N0`: below it, alpha drifts to 1;
  above it, alpha stabilizes at `alpha* = q/(mu*N0)` (Eq 9).
- **Scale invariance**: holding `lambda = q/N0` fixed, the dynamics for
  N0 = 10k / 100k / 1M are *identical* — this repo demonstrates that
  without needing to store a single document string at those scales.

## Quickstart (works on free Colab, no Pro account needed)

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
pip install -r requirements.txt
pytest tests/ -v                                   # ~5 seconds
python scripts/run_phase_transition.py              # N0=23/100/1000, ~10s
python scripts/run_scale_benchmark.py                # N0=10k/100k/1M, ~10s
```

All four commands above are **CPU-only** and run comfortably on a free
Colab runtime (or your laptop) — no GPU, no network beyond the initial
`pip install` and `git clone`. This is the part that gets you the
publication-grade phase-transition and scale-invariance figures.

## Honest feasibility table (free Colab, no Pro)

The alpha(t) math above is cheap at *any* N0 because it's either a
closed-form solution or a two-float running count — no documents,
embeddings, or index are ever materialized just to plot alpha(t).

Scale only matters once you additionally want **real retrieval + QA
accuracy** (`scripts/run_qa_validation.py`, optional, disabled by default):

| N0 | alpha(t) math | QA validation (real retrieval+generation) |
|---|---|---|
| 23 – 10,000 | seconds, CPU | Fine on free T4. Flat FAISS index, `real` generation. |
| 100,000 | seconds, CPU | Works on free T4 with `IndexIVFFlat`, but budget for session checkpointing (free Colab disconnects after ~90 min idle). |
| 1,000,000 | seconds, CPU | **Not feasible with real document generation** on free Colab — this would be a multi-day job even on paid A100s. Use `generation_mode: proxy` (duplicate/perturb a smaller unique pool — see `src/qa_evaluator.make_proxy_corpus`), `IndexIVFPQ`, and be explicit in any writeup that 1M results measure *retrieval-scale behaviour*, not linguistic diversity at that scale. |

**Practical Colab tips:**
- Mount Google Drive and save `results/` there periodically — free-tier
  sessions disconnect and you lose local disk state.
- `faiss-cpu`, not `faiss-gpu` — the pip `faiss-gpu` wheel is unofficial
  and frequently breaks on Colab's CUDA version. `faiss-cpu` with IVFPQ
  is plenty fast for the checkpoint-only evaluation pattern used here
  (you only rebuild the index at ~5 checkpoints, not every iteration).
- Run `python scripts/run_qa_validation.py --smoke_test` first — a tiny
  end-to-end pass (5 documents, 3 QA items) that catches config/API
  errors before you commit a session to a full run.

## Repo structure

```
src/
  analytical.py    # Eq 3, 6, 8-exact, 9, 10 — pure math, verified vs odeint
  simulator.py      # CountSimulator (scalable, any N0) +
                     # DocumentLevelSimulator (small N0, real eviction policy)
  eviction.py       # RandomEviction (chemostat baseline) +
                     # ProvenanceEviction (Eq 11-based, operationalized mu)
  qa_evaluator.py    # OPTIONAL. Generator/evaluator split, IVF/IVFPQ support,
                     # proxy-corpus generation. Needs torch/transformers/
                     # faiss/network. Not exercised by the CPU scripts.
scripts/
  run_phase_transition.py   # N0=23/100/1000, CPU-only
  run_scale_benchmark.py    # N0=10k/100k/1M, CPU-only
  run_qa_validation.py       # OPTIONAL, needs GPU/network, disabled by
                             # default (configs/scale_benchmark.yaml ->
                             # qa_validation.enabled: false)
configs/
  phase_transition.yaml
  scale_benchmark.yaml
tests/
  test_analytical.py         # the ODE-vs-closed-form validation (run this
                             # before trusting any figure in this repo)
```

## Design notes and known limitations

- **`src/simulator.py`'s `detection_delay`** in `CountSimulator` is a
  simplified proxy (it gates the eviction term on/off based on a lagged
  queue value, applied to the *aggregate* alpha) rather than tracking
  per-document eligibility. For a rigorous per-document detection delay,
  use `DocumentLevelSimulator`, which tracks `created_at` per document —
  but that's O(N0) per iteration, so only use it at small N0 (~<5,000).
- **`ProvenanceEviction` does not reduce exactly to Eq (8)/(9)/(10).**
  Those closed-form equations assume the idealized random-chemostat
  policy (`RandomEviction`); `ProvenanceEviction` is provided so you can
  empirically measure how far a *realistic* policy diverges from that
  idealization (see the repo's own sanity check: under matched nominal
  parameters, `ProvenanceEviction` settled at a substantially higher
  alpha than the theoretical alpha*, i.e. it's meaningfully weaker than
  the idealized random purge at the same rate — worth discussing in
  a paper, not hiding).
- **`src/qa_evaluator.py` and `scripts/run_qa_validation.py` are
  syntax-checked only** (`python -m py_compile`), not functionally
  tested, because the environment that produced this repo has no
  network/GPU access. Run the `--smoke_test` flag yourself before a
  full run.
- **Query traffic is uniform**, not Zipfian/temporal (item E from the
  original critique). Not implemented — flagged as a real gap if you
  want to claim realistic retrieval-traffic modelling, but it doesn't
  block the core phase-transition / scale-invariance results.

## Validating the math yourself

```bash
pytest tests/ -v
```

This checks: the closed-form Eq (8) solution against `odeint` across four
`(q, N0, mu)` configurations, correct convergence to the steady state
(including at extreme `t` without numerical overflow — this bit an
earlier draft of the formula), Eq (3) against its own governing ODE
(Eq 4 — note this is *not* the same equation as Eq 8 at `mu=0`, an easy
mistake to make since both look like "the uncorrected case"), the
half-life definition, the `mu_critical` boundary, and the scale-invariance
claim itself.
