#!/usr/bin/env python3
"""Compare RLM vs RAG vs ReAct+BM25 on a long PDF.

Fair comparison: all three methods use the SAME base model and provider, the
SAME PDF text, and the SAME questions. Scores are SQuAD-style EM + token F1.

Quick start
-----------
    # 1. Set a key (OpenAI, or any OpenAI-compatible endpoint via OPENAI_BASE_URL)
    export OPENAI_API_KEY=sk-...
    # optional: export OPENAI_BASE_URL=https://openrouter.ai/api/v1

    # 2. Dry-run (no API calls) -- verifies PDF parsing + retrieval:
    cd comparison
    python compare.py --pdf /path/to/big.pdf --dry-run

    # 3. Full comparison against your questions file:
    python compare.py --pdf /path/to/big.pdf --questions questions.example.jsonl \
        --model gpt-5-mini --out results.csv

Questions file format (JSONL, one per line):
    {"question": "...", "answer": "gold answer"}          # single gold
    {"question": "...", "answer": ["gold1", "gold2"]}      # multiple acceptable
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import retrievers
import methods
import scoring

ALL_METHODS = ["rag", "react_bm25", "rlm"]


def load_questions(path: str):
    qs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def run_one(method_name, question, full_text, retriever, bm25, cfg):
    if method_name == "rag":
        return methods.rag_answer(question, retriever, cfg.model, k=cfg.top_k)
    if method_name == "react_bm25":
        return methods.react_bm25_answer(question, bm25, cfg.model, max_steps=cfg.react_steps)
    if method_name == "rlm":
        return methods.rlm_answer(
            question, full_text, cfg.model, env=cfg.rlm_env,
            max_depth=cfg.rlm_depth, max_budget=cfg.rlm_budget,
            max_iterations=cfg.rlm_iters, log_dir=cfg.log_dir,
        )
    raise ValueError(method_name)


def main():
    ap = argparse.ArgumentParser(description="RLM vs RAG vs ReAct+BM25 on a long PDF.")
    ap.add_argument("--pdf", "--corpus", dest="pdf", required=True,
                    help="Path to a PDF, a .txt/.md file, or a DIRECTORY of text/PDF "
                         "files (e.g. the BrowseComp-Plus corpus) to evaluate on.")
    ap.add_argument("--questions", default="questions.example.jsonl",
                    help="JSONL of {question, answer}.")
    ap.add_argument("--model", default="gpt-5-mini",
                    help="Base model for ALL three methods (fair comparison).")
    ap.add_argument("--methods", default=",".join(ALL_METHODS),
                    help="Comma-separated subset: rag,react_bm25,rlm")
    ap.add_argument("--rag-retriever", default="dense",
                    choices=["dense", "bm25"],
                    help="Retriever for the RAG baseline (dense=OpenAI emb/TF-IDF, bm25).")
    ap.add_argument("--embed-model", default="text-embedding-3-small",
                    help="OpenAI embedding model (only if --rag-retriever=dense and key set).")
    ap.add_argument("--top-k", type=int, default=12, help="Chunks retrieved per query.")
    ap.add_argument("--chunk-size", type=int, default=1500)
    ap.add_argument("--chunk-overlap", type=int, default=300)
    ap.add_argument("--react-steps", type=int, default=6)
    ap.add_argument("--rlm-env", default="local", choices=["local", "docker", "e2b", "modal"])
    ap.add_argument("--rlm-depth", type=int, default=2, help="Max recursive sub-LLM depth.")
    ap.add_argument("--rlm-budget", type=float, default=2.0, help="RLM USD budget cap.")
    ap.add_argument("--rlm-iters", type=int, default=40)
    ap.add_argument("--log-dir", default="./logs", help="Where RLM writes JSONL trajectories.")
    ap.add_argument("--out", default="results.csv", help="CSV results path.")
    ap.add_argument("--limit", type=int, default=0, help="Only run first N questions (0=all).")
    ap.add_argument("--dry-run", action="store_true",
                    help="No API calls. Load PDF, chunk, build retrievers, "
                         "print stats + a sample retrieval.")
    cfg = ap.parse_args()

    # ---- load & chunk PDF -------------------------------------------------- #
    print(f"[load] reading corpus: {cfg.pdf}")
    t0 = time.time()
    full_text = retrievers.load_text(cfg.pdf)
    chunks = retrievers.chunk_text(full_text, cfg.chunk_size, cfg.chunk_overlap)
    n_chars = len(full_text)
    est_tokens = n_chars // 4
    print(f"[load] {n_chars:,} chars (~{est_tokens:,} tokens est) | "
          f"{len(chunks):,} chunks | {time.time()-t0:.1f}s")

    if not chunks:
        sys.exit("No text extracted from corpus.")

    # ---- build retrievers -------------------------------------------------- #
    print(f"[retrievers] fitting BM25 ...")
    bm25 = retrievers.BM25Retriever.fit(chunks)
    retriever = bm25
    if cfg.rag_retriever == "dense":
        key = os.environ.get("OPENAI_API_KEY")
        base = os.environ.get("OPENAI_BASE_URL")
        print(f"[retrievers] fitting dense ({'OpenAI/compatible' if key else 'TF-IDF fallback'}) ...")
        retriever = retrievers.DenseRetriever.fit(chunks, key, cfg.embed_model, base)

    # ---- dry run ----------------------------------------------------------- #
    if cfg.dry_run:
        sample_q = "What is this document about?"
        print("\n[dry-run] sample retrieval (BM25):")
        for i, c in enumerate(bm25.retrieve(sample_q, k=3), 1):
            print(f"  [{i}] {c[:160].replace(chr(10),' ')}…")
        print("\n[dry-run] sample retrieval (dense / %s):" % retriever.kind
              if hasattr(retriever, "kind") else "")
        if cfg.rag_retriever == "dense":
            for i, c in enumerate(retriever.retrieve(sample_q, k=3), 1):
                print(f"  [{i}] {c[:160].replace(chr(10),' ')}…")
        print("\n[dry-run] OK. Set OPENAI_API_KEY and drop --dry-run to run the methods.")
        return

    # ---- run methods ------------------------------------------------------- #
    method_names = [m.strip() for m in cfg.methods.split(",") if m.strip() in ALL_METHODS]
    if not method_names:
        sys.exit(f"--methods must include some of {ALL_METHODS}")

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (needed for all three methods).")

    questions = load_questions(cfg.questions)
    if cfg.limit:
        questions = questions[: cfg.limit]
    print(f"[run] {len(questions)} questions × {len(method_names)} methods "
          f"({method_names}) | model={cfg.model}")

    rows = []
    agg = {m: {"em": [], "f1": [], "ptok": 0, "ctok": 0, "lat": 0.0, "calls": 0}
           for m in method_names}

    for qi, q in enumerate(questions, 1):
        question, gold = q["question"], q.get("answer", "")
        print(f"\n─ Q{qi}: {question[:90]}")
        for m in method_names:
            try:
                t0 = time.time()
                r = run_one(m, question, full_text, retriever, bm25, cfg)
                sc = scoring.score(r["answer"], gold)
                status = "✓" if sc["em"] else ("~" if sc["f1"] > 0 else "✗")
                print(f"   {m:11s} {status} EM={sc['em']:.0f} F1={sc['f1']:.2f} "
                      f"P={r['prompt_tokens']} C={r['completion_tokens']} "
                      f"{r['latency_s']:.1f}s")
                row = {"question": question, "method": m,
                       "answer": r["answer"][:500], "em": sc["em"], "f1": sc["f1"],
                       "prompt_tokens": r["prompt_tokens"],
                       "completion_tokens": r["completion_tokens"],
                       "total_tokens": r["prompt_tokens"] + r["completion_tokens"],
                       "latency_s": round(r["latency_s"], 2),
                       "n_calls": r.get("n_calls"), "gold": gold}
                rows.append(row)
                agg[m]["em"].append(sc["em"]); agg[m]["f1"].append(sc["f1"])
                agg[m]["ptok"] += r["prompt_tokens"]; agg[m]["ctok"] += r["completion_tokens"]
                agg[m]["lat"] += r["latency_s"]
                agg[m]["calls"] += r.get("n_calls") or 0
            except Exception as e:
                print(f"   {m:11s} ERROR: {e}")
                rows.append({"question": question, "method": m, "answer": f"ERROR: {e}",
                             "em": 0, "f1": 0, "prompt_tokens": 0, "completion_tokens": 0,
                             "total_tokens": 0, "latency_s": 0, "n_calls": 0, "gold": gold})

    # ---- write CSV --------------------------------------------------------- #
    with open(cfg.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "method", "answer", "em", "f1",
                                          "prompt_tokens", "completion_tokens", "total_tokens",
                                          "latency_s", "n_calls", "gold"])
        w.writeheader(); w.writerows(rows)
    print(f"\n[wrote] {cfg.out}")

    # ---- summary ----------------------------------------------------------- #
    print("\n===== SUMMARY =====")
    print(f"{'method':12s} {'EM':>6s} {'F1':>6s} {'P-tok':>9s} {'C-tok':>9s} "
          f"{'tot-tok':>9s} {'latency':>8s}")
    nq = len(questions) or 1
    for m in method_names:
        a = agg[m]
        em = sum(a["em"]) / nq
        f1 = sum(a["f1"]) / nq
        print(f"{m:12s} {em:6.2f} {f1:6.2f} {a['ptok']:9,} {a['ctok']:9,} "
              f"{a['ptok']+a['ctok']:9,} {a['lat']:7.1f}s")
    print(f"\nRLM trajectories: {cfg.log_dir}/  (visualize with the rlm visualizer)")


if __name__ == "__main__":
    main()