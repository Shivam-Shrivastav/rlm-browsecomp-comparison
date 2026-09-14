#!/usr/bin/env python3
"""Run RLM vs RAG vs ReAct+BM25 on REAL BrowseComp-Plus multi-hop queries.

Each BrowseComp-Plus query ships with its own gold/evidence/negative documents
(full text included in the decrypted JSONL). We build a per-query corpus from
those docs -- exactly the paper's offline setup -- then run all three methods on
that corpus + question, scored against the gold answer.

Usage:
    export OPENAI_API_KEY=ollama
    export OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama
    python run_browsecomp.py --queries 3 --model gpt-oss:120b-cloud \\
        --embed-model nomic-embed-text --out bc_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import retrievers
import methods
import scoring

ALL_METHODS = ["rag", "rag_dense", "react_bm25", "rlm"]


def load_decrypted(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_corpus_dir(rec, out_dir, extra_random_docs=None):
    """Write gold+evidence+negative docs (de-duped) as .txt files. Returns count."""
    os.makedirs(out_dir, exist_ok=True)
    seen, n = set(), 0
    for key in ("gold_docs", "evidence_docs", "negative_docs"):
        for d in rec.get(key, []):
            did = str(d.get("docid"))
            if did in seen:
                continue
            seen.add(did)
            txt = (d.get("text") or "").strip()
            if not txt:
                continue
            with open(os.path.join(out_dir, f"{did}.txt"), "w", encoding="utf-8") as f:
                f.write(f"docid: {did}\nurl: {d.get('url','')}\n\n{txt}")
            n += 1
    for did, txt in (extra_random_docs or []):
        if str(did) in seen:
            continue
        seen.add(str(did))
        with open(os.path.join(out_dir, f"{did}.txt"), "w", encoding="utf-8") as f:
            f.write(f"docid: {did}\n\n{txt}")
        n += 1
    return n


def _process_query_task(task):
    """Top-level picklable wrapper that unpacks a task tuple for imap_unordered."""
    return _process_query(*task)


def _process_query(qi, rec, cfg, extra_pool):
    """Run all methods on one query. Self-contained so it can run in a worker
    process (multiprocessing). Returns a dict with rows + log lines + agg deltas.
    """
    # ensure env is present in spawned workers (spawn start method re-imports)
    os.environ.setdefault("OPENAI_API_KEY", getattr(cfg, "_api_key", "ollama"))
    os.environ.setdefault("OPENAI_BASE_URL", getattr(cfg, "_base_url", "http://localhost:11434/v1"))

    log = []
    q, gold = rec["query"], rec["answer"]
    method_names = [m.strip() for m in cfg.methods.split(",") if m.strip() in ALL_METHODS]
    cdir = f"./query_corpus/q{rec['query_id']}"
    extra = None
    if cfg.extra_negatives and extra_pool:
        extra = random.sample(extra_pool, min(cfg.extra_negatives, len(extra_pool)))
    ndocs = build_corpus_dir(rec, cdir, extra)
    full_text = retrievers.load_text(cdir)
    est_tok = len(full_text) // 4
    chunks = retrievers.chunk_text(full_text, size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    bm25 = retrievers.BM25Retriever.fit(chunks)
    retriever = bm25
    dense = None
    need_dense = "rag_dense" in method_names
    if cfg.rag_retriever == "dense" or need_dense:
        dense = retrievers.DenseRetriever.fit(
            chunks, os.environ.get("OPENAI_API_KEY"), cfg.embed_model,
            os.environ.get("OPENAI_BASE_URL"))
        if cfg.rag_retriever == "dense":
            retriever = dense
    log.append(f"\n─ Q{qi} (id={rec['query_id']}) | {ndocs} docs | ~{est_tok:,} tok | gold: {gold}")
    log.append(f"  {q[:120]}")
    rows = []
    agg_delta = {m: {"correct": 0, "em": 0, "f1": 0.0, "ptok": 0, "ctok": 0, "lat": 0.0, "calls": 0}
                 for m in method_names}
    for m in method_names:
        try:
            r = run_one(m, q, full_text, retriever, bm25, dense, cfg)
            sc = scoring.score(r["answer"], gold)
            status = "✓" if sc["correct"] else ("~" if sc["f1"] > 0 else "✗")
            log.append(f"   {m:11s} {status} EM={sc['em']:.0f} F1={sc['f1']:.2f} "
                       f"cont={sc['containment']:.0f} "
                       f"P={r['prompt_tokens']} C={r['completion_tokens']} "
                       f"calls={r.get('n_calls')} {r['latency_s']:.1f}s")
            log.append(f"              ans: {r['answer'][:140]}")
            rows.append({"query_id": rec["query_id"], "method": m, "question": q[:300],
                         "answer": r["answer"][:500], "gold": gold,
                         "em": sc["em"], "f1": sc["f1"],
                         "containment": sc["containment"], "correct": sc["correct"],
                         "prompt_tokens": r["prompt_tokens"],
                         "completion_tokens": r["completion_tokens"],
                         "latency_s": round(r["latency_s"], 2),
                         "n_calls": r.get("n_calls")})
            agg_delta[m]["correct"] = sc["correct"]; agg_delta[m]["em"] = sc["em"]
            agg_delta[m]["f1"] = sc["f1"]
            agg_delta[m]["ptok"] = r["prompt_tokens"]; agg_delta[m]["ctok"] = r["completion_tokens"]
            agg_delta[m]["lat"] = r["latency_s"]; agg_delta[m]["calls"] = r.get("n_calls") or 0
        except Exception as e:
            import traceback
            log.append(f"   {m:11s} ERROR: {e}")
            log.append(traceback.format_exc())
            rows.append({"query_id": rec["query_id"], "method": m, "question": q[:300],
                         "answer": f"ERROR: {e}", "gold": gold, "em": 0, "f1": 0,
                         "containment": 0, "correct": 0,
                         "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0, "n_calls": 0})
    return {"qi": qi, "query_id": rec["query_id"], "rows": rows, "log": log, "agg_delta": agg_delta}


def run_one(method_name, question, full_text, retriever, bm25, dense, cfg):
    if method_name == "rag":
        return methods.rag_answer(question, retriever, cfg.model, k=cfg.top_k)
    if method_name == "rag_dense":
        if dense is None:
            raise RuntimeError("rag_dense requested but dense retriever not built")
        return methods.rag_answer(question, dense, cfg.model, k=cfg.top_k)
    if method_name == "react_bm25":
        return methods.react_bm25_answer(question, bm25, cfg.model, max_steps=cfg.react_steps)
    if method_name == "rlm":
        return methods.rlm_answer(
            question, full_text, cfg.rlm_model or cfg.model, env=cfg.rlm_env,
            max_depth=cfg.rlm_depth, max_budget=cfg.rlm_budget,
            max_iterations=cfg.rlm_iters, log_dir=cfg.log_dir,
            max_tokens=cfg.rlm_max_tokens or None,
            per_call_max_tokens=cfg.rlm_per_call_tokens)
    raise ValueError(method_name)


def main():
    ap = argparse.ArgumentParser(description="RLM vs RAG vs ReAct+BM25 on BrowseComp-Plus.")
    ap.add_argument("--decrypted", default="browsecomp_plus_decrypted.jsonl")
    ap.add_argument("--queries", type=int, default=3, help="Number of queries to evaluate.")
    ap.add_argument("--skip", type=int, default=0, help="Skip first N queries.")
    ap.add_argument("--seed", type=int, default=0, help="Seed for query sampling.")
    ap.add_argument("--model", default="gpt-oss:120b-cloud", help="Base chat model for all 3 methods.")
    ap.add_argument("--rlm-model", default=None,
                    help="Model for the RLM method only (defaults to --model). Set to a "
                         "non-thinking model e.g. qwen3-coder:480b-cloud to test the RLM "
                         "scaffold with a model that doesn't burn output budget on reasoning.")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--rag-retriever", default="dense", choices=["dense", "bm25"])
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--chunk-size", type=int, default=1500,
                    help="Chunk size (chars). Larger = fewer chunks = faster dense "
                         "embedding at coarser retrieval granularity.")
    ap.add_argument("--chunk-overlap", type=int, default=300)
    ap.add_argument("--react-steps", type=int, default=6)
    ap.add_argument("--rlm-env", default="local")
    ap.add_argument("--rlm-depth", type=int, default=2)
    ap.add_argument("--rlm-budget", type=float, default=3.0)
    ap.add_argument("--rlm-iters", type=int, default=60,
                    help="Max RLM iterations. Hard multi-hop queries need ~60 to converge.")
    ap.add_argument("--rlm-max-tokens", type=int, default=0,
                    help="RLM cumulative token budget across all sub-calls. 0 = unlimited "
                         "(bounded by --rlm-iters). Reasoning models need this unset.")
    ap.add_argument("--rlm-per-call-tokens", type=int, default=8192,
                    help="Per-call completion cap for RLM sub-LLM calls.")
    ap.add_argument("--extra-negatives", type=int, default=0,
                    help="Add N random docs from the full corpus parquet to grow the haystack.")
    ap.add_argument("--corpus-parquet-dir", default="corpus/bc_plus",
                    help="Full corpus parquet dir (only used if --extra-negatives > 0).")
    ap.add_argument("--log-dir", default="./logs_bc")
    ap.add_argument("--out", default="bc_results.csv")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of queries to run in parallel (multiprocessing). "
                         "Each worker gets its own RLM/REPL. >1 speeds up wall-clock "
                         "since RLM per query is the bottleneck. Beware cloud rate limits.")
    cfg = ap.parse_args()

    # env defaults for Ollama
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11434/v1")
    cfg._api_key = os.environ.get("OPENAI_API_KEY")
    cfg._base_url = os.environ.get("OPENAI_BASE_URL")

    method_names = [m.strip() for m in cfg.methods.split(",") if m.strip() in ALL_METHODS]
    records = load_decrypted(cfg.decrypted)
    random.seed(cfg.seed)
    pool = records[cfg.skip:]
    chosen = pool[: cfg.queries] if cfg.seed == 0 else random.sample(pool, min(cfg.queries, len(pool)))
    print(f"[run] {len(chosen)} queries × {method_names} | model={cfg.model}")

    # optional extra negatives from full corpus
    extra_pool = []
    if cfg.extra_negatives:
        import corpus_index
        import pyarrow.parquet as pq, glob
        shards = sorted(glob.glob(os.path.join(cfg.corpus_parquet_dir, "data", "train-*.parquet")))
        random.shuffle(shards)
        for shard in shards:
            if len(extra_pool) >= cfg.extra_negatives * 3:
                break
            t = pq.read_table(shard, columns=["docid", "text"])
            ids, texts = t.column("docid").to_pylist(), t.column("text").to_pylist()
            for i, did in enumerate(ids):
                extra_pool.append((str(did), texts[i] or ""))
                if len(extra_pool) >= cfg.extra_negatives * 3:
                    break

    rows, agg = [], {m: {"correct": [], "em": [], "f1": [], "ptok": 0, "ctok": 0, "lat": 0.0, "calls": 0}
                     for m in method_names}

    tasks = [(qi, rec, cfg, extra_pool) for qi, rec in enumerate(chosen, 1)]

    def _absorb(res):
        for line in res["log"]:
            print(line, flush=True)
        rows.extend(res["rows"])
        for m, d in res["agg_delta"].items():
            agg[m]["correct"].append(d["correct"]); agg[m]["em"].append(d["em"])
            agg[m]["f1"].append(d["f1"])
            agg[m]["ptok"] += d["ptok"]; agg[m]["ctok"] += d["ctok"]
            agg[m]["lat"] += d["lat"]; agg[m]["calls"] += d["calls"]

    if cfg.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")  # spawn: cleanest isolation for per-task RLM REPLs
        with ctx.Pool(min(cfg.workers, len(tasks))) as p:
            # imap_unordered: stream results as they complete (not in submission
            # order), so progress is visible even when one query is slow.
            for res in p.imap_unordered(_process_query_task, tasks):
                _absorb(res)
    else:
        for t in tasks:
            res = _process_query(*t)
            _absorb(res)

    with open(cfg.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "method", "question", "answer", "gold",
                                          "em", "f1", "containment", "correct",
                                          "prompt_tokens", "completion_tokens",
                                          "latency_s", "n_calls"])
        w.writeheader(); w.writerows(rows)
    print(f"\n[wrote] {cfg.out}")

    print("\n===== SUMMARY =====")
    print(f"{'method':12s} {'Corr':>6s} {'EM':>6s} {'F1':>6s} {'P-tok':>9s} {'C-tok':>9s} {'tot':>9s} {'lat':>7s} {'calls':>6s}")
    nq = len(chosen) or 1
    for m in method_names:
        a = agg[m]
        print(f"{m:12s} {sum(a['correct'])/nq:6.2f} {sum(a['em'])/nq:6.2f} {sum(a['f1'])/nq:6.2f} "
              f"{a['ptok']:9,} {a['ctok']:9,} {a['ptok']+a['ctok']:9,} {a['lat']:6.1f}s {a['calls']:6,}")
    print(f"\nRLM trajectories: {cfg.log_dir}/")


if __name__ == "__main__":
    main()