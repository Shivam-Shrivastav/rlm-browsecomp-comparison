#!/usr/bin/env python3
"""Re-run only the RLM cloud-timeout misses (Q791, Q794) from Set 2.
Writes results to bc_results_rerun_timeouts.csv so the main CSVs stay intact.
"""
from __future__ import annotations

import csv, json, os, time
import multiprocessing as mp

import retrievers, methods, scoring

os.environ.setdefault("OPENAI_API_KEY", "ollama")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11434/v1")

MODEL = "kimi-k2.7-code:cloud"
TARGET = {"791", "794"}
DEC = "browsecomp_plus_decrypted.jsonl"
OUT = "bc_results_rerun_timeouts.csv"


def _run(task):
    qi, rec = task
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11434/v1")
    q, gold = rec["query"], rec["answer"]
    cdir = f"./query_corpus/q{rec['query_id']}"
    ndocs = 0
    # reuse corpus dir if already built by the prior run
    if not os.path.isdir(cdir):
        from run_browsecomp import build_corpus_dir
        ndocs = build_corpus_dir(rec, cdir)
    full_text = retrievers.load_text(cdir)
    chunks = retrievers.chunk_text(full_text, size=1500, overlap=300)
    print(f"[rerun] Q{rec['query_id']} gold={gold} | {len(full_text)//4:,} tok", flush=True)
    t0 = time.time()
    r = methods.rlm_answer(
        q, full_text, MODEL, env="local", max_depth=2, max_budget=3.0,
        max_iterations=60, log_dir="./logs_bc", max_tokens=None,
        per_call_max_tokens=8192)
    sc = scoring.score(r["answer"], gold)
    dt = time.time() - t0
    print(f"[rerun] Q{rec['query_id']} -> correct={sc['correct']} EM={sc['em']} "
          f"F1={sc['f1']:.2f} calls={r['n_calls']} {dt:.0f}s ans={r['answer'][:80]}", flush=True)
    return {"query_id": rec["query_id"], "method": "rlm", "question": q[:300],
            "answer": r["answer"][:500], "gold": gold,
            "em": sc["em"], "f1": sc["f1"], "containment": sc["containment"],
            "correct": sc["correct"], "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"], "latency_s": round(dt, 2),
            "n_calls": r.get("n_calls")}


def main():
    recs = [json.loads(l) for l in open(DEC) if l.strip()]
    tasks = [(i, r) for i, r in enumerate(recs, 1) if str(r["query_id"]) in TARGET]
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(2, len(tasks))) as p:
        for row in p.imap_unordered(_run, tasks):
            rows.append(row)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "method", "question", "answer", "gold",
                                          "em", "f1", "containment", "correct",
                                          "prompt_tokens", "completion_tokens",
                                          "latency_s", "n_calls"])
        w.writeheader(); w.writerows(rows)
    print(f"\n[wrote] {OUT}")
    for r in rows:
        print(f"  Q{r['query_id']}: correct={r['correct']} EM={r['em']} ans={r['answer'][:60]}")


if __name__ == "__main__":
    main()