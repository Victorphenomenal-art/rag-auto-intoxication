"""
src/qa_evaluator.py
====================
REQUIRES: torch, transformers, sentence-transformers, datasets, faiss,
network access (model + dataset downloads), and ideally a GPU.

NOT executed or tested in the environment that produced this repo (no
network/GPU there) -- syntax-checked with py_compile only. Test this
module yourself with a small smoke run (e.g. 5 documents, 3 QA items)
before trusting it in a full experiment.

Item A from the critique: splits GENERATION and EVALUATION across two
different models to avoid the self-referential confound where a single
model both writes the synthetic corpus and grades answers derived from
it (which can inflate scores because the evaluator is unusually good at
parsing its own generation style/patterns).

  - generator_model : produces synthetic documents (cheap, e.g. flan-t5-base)
  - eval_model       : answers QA questions using retrieved context
                        (should be a DIFFERENT model/family, e.g.
                        flan-t5-large or, if you have the VRAM budget,
                        mistralai/Mistral-7B-Instruct-v0.3 in 4-bit)

A note on feasibility: Mistral-7B-Instruct in 4-bit needs ~4-5GB VRAM and
fits on a free-tier T4 (16GB), but downloading + loading it repeatedly
across Colab session resets is slow and disk-quota-sensitive. flan-t5-large
(~800M params) is the more Colab-friendly default; swap in Mistral only if
you have a stable, longer-running session (Colab Pro or local GPU).
"""

import re
from collections import Counter
from typing import List, Tuple

import numpy as np


def normalize_answer(s) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-zA-Z0-9 ]", "", s)
    return " ".join(s.split())


def compute_em(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    g, p = normalize_answer(a_gold).split(), normalize_answer(a_pred).split()
    common = Counter(g) & Counter(p)
    num_same = sum(common.values())
    if num_same == 0 or len(p) == 0 or len(g) == 0:
        return 0.0
    prec, rec = num_same / len(p), num_same / len(g)
    return 2 * prec * rec / (prec + rec)


class DocumentGenerator:
    """Generates synthetic documents. Cheap model by design (item A)."""

    def __init__(self, model_name: str = "google/flan-t5-base", device: str = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def generate(self, seed_text: str, n: int = 1, max_new_tokens: int = 60) -> List[str]:
        import torch
        prompt = f"Continue or paraphrase this passage:\n{seed_text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=256).to(self.device)
        outs = []
        for _ in range(n):
            with torch.no_grad():
                gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                           do_sample=True, temperature=0.9)
            outs.append(self.tokenizer.decode(gen[0], skip_special_tokens=True))
        return outs


class RAGQAEvaluator:
    """
    Evaluates a knowledge base (list of document strings) on a SQuAD sample.
    Uses a SEPARATE model from whatever generated the synthetic documents
    (see DocumentGenerator) -- pass a different `eval_model_name`.
    """

    def __init__(self, eval_model_name: str = "google/flan-t5-large",
                 embed_model_name: str = "all-MiniLM-L6-v2",
                 n_test_items: int = 30, seed: int = 42, device: str = None):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from datasets import load_dataset

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(embed_model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(eval_model_name).to(self.device)
        self.test_set = (load_dataset("stanfordnlp/squad", split="validation")
                          .shuffle(seed=seed).select(range(n_test_items)))

    def answer(self, context: str, question: str) -> str:
        import torch
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer concisely:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=512).to(self.device)
        with torch.no_grad():
            out = self.generator.generate(**inputs, max_new_tokens=30, temperature=0.1)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def evaluate_kb(self, documents: List[str], index_type: str = "flat",
                     ivf_nlist: int = 4096, pq_m: int = 64, pq_nbits: int = 8,
                     top_k: int = 3) -> Tuple[float, float]:
        """
        index_type: 'flat' | 'ivfflat' | 'ivfpq'
        Returns (mean EM, mean F1) over the held-out QA test set.
        """
        import faiss

        if len(documents) == 0:
            return 0.0, 0.0
        embs = self.embedder.encode(documents, convert_to_numpy=True,
                                     batch_size=128).astype(np.float32)
        faiss.normalize_L2(embs)
        dim = embs.shape[1]

        if index_type == "ivfpq":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, ivf_nlist, pq_m, pq_nbits)
            index.train(embs)
        elif index_type == "ivfflat":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, ivf_nlist)
            index.train(embs)
        elif index_type == "flat":
            index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"unknown index_type: {index_type}")
        index.add(embs)

        ems, f1s = [], []
        for item in self.test_set:
            q_emb = self.embedder.encode([item["question"]], convert_to_numpy=True).astype(np.float32)
            faiss.normalize_L2(q_emb)
            _, idxs = index.search(q_emb, top_k)
            context = " ".join(documents[i] for i in idxs[0] if 0 <= i < len(documents))
            pred = self.answer(context, item["question"])
            gold = item["answers"]["text"][0]
            ems.append(compute_em(gold, pred))
            f1s.append(compute_f1(gold, pred))
        return float(np.mean(ems)), float(np.mean(f1s))


def make_proxy_corpus(unique_docs: List[str], target_size: int, noise_std: float = 0.0,
                       seed: int = 0) -> List[str]:
    """
    Item D: cheap stand-in for generating `target_size` unique documents at
    scale (e.g. 1M). Duplicates/paraphrase-shuffles `unique_docs` up to
    `target_size` rather than running the generator that many times.

    IMPORTANT: this only approximates RETRIEVAL-scale behaviour (index
    size, query latency, alpha bookkeeping). It does NOT approximate
    linguistic diversity -- label any results built on this as "scaling of
    retrieval behaviour", not "scaling of generation quality", per item D.
    """
    rng = np.random.default_rng(seed)
    if not unique_docs:
        return []
    out = []
    while len(out) < target_size:
        doc = unique_docs[rng.integers(0, len(unique_docs))]
        words = doc.split()
        if len(words) > 4 and rng.random() < 0.5:
            # cheap "paraphrase": shuffle a short span
            i = rng.integers(0, len(words) - 3)
            span = words[i:i + 3]
            rng.shuffle(span)
            words[i:i + 3] = span
            doc = " ".join(words)
        out.append(doc)
    return out[:target_size]
