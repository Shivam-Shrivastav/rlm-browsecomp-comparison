"""The three answer methods, all using the SAME base model and provider so the
comparison is fair. Each returns a dict:
    {answer, prompt_tokens, completion_tokens, latency_s, n_calls, extra}

The LLM client is OpenAI-compatible (works with OpenAI, OpenRouter, DeepSeek,
local vLLM, etc.). Set OPENAI_API_KEY and (optional) OPENAI_BASE_URL.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any


# --------------------------------------------------------------------------- #
# Shared LLM client
# --------------------------------------------------------------------------- #
def _client():
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it (or point OPENAI_BASE_URL at an "
            "OpenAI-compatible provider) before running the methods."
        )
    return OpenAI(api_key=key, base_url=os.environ.get("OPENAI_BASE_URL") or None,
                  timeout=600, max_retries=4)


def _llm(messages, model, temperature=0.0, max_tokens=1024):
    """One chat completion. Returns (text, prompt_tokens, completion_tokens)."""
    c = _client()
    r = c.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    txt = r.choices[0].message.content or ""
    usage = getattr(r, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) if usage else 0
    ct = getattr(usage, "completion_tokens", 0) if usage else 0
    return txt, pt, ct


def _usage_str(u):
    return f"P={u['prompt_tokens']} C={u['completion_tokens']}"


# --------------------------------------------------------------------------- #
# 1) RAG -- dense retrieval, stuff top-k into one prompt
# --------------------------------------------------------------------------- #
def rag_answer(question, retriever, model, k=12):
    t0 = time.time()
    ctx = "\n\n---\n\n".join(retriever.retrieve(question, k))
    msgs = [
        {"role": "system", "content": "Answer the question using only the provided context. "
                                      "If the context is insufficient, say you don't know. "
                                      "Be concise."},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"},
    ]
    ans, pt, ct = _llm(msgs, model, max_tokens=4096)
    return {"answer": ans.strip(), "prompt_tokens": pt, "completion_tokens": ct,
            "latency_s": time.time() - t0, "n_calls": 1, "extra": {"k": k, "ctx_chars": len(ctx)}}


# --------------------------------------------------------------------------- #
# 2) ReAct + BM25 -- text-based ReAct loop with a retrieve_bm25 tool
# --------------------------------------------------------------------------- #
REACT_SYS = (
    "You are a ReAct agent. You answer a question by reasoning step by step and "
    "calling a retrieval tool. You have ONE tool:\n"
    "  retrieve_bm25(query, k=8) -> returns text passages from the document ranked "
    "by lexical (BM25) match.\n\n"
    "At each step output EXACTLY one block in this format:\n"
    "Thought: <your reasoning>\n"
    "Action: retrieve_bm25\n"
    "Action Input: <a search query, possibly refined from the last observation>\n"
    "When you have enough information, output:\n"
    "Thought: <your reasoning>\n"
    "Final Answer: <concise answer to the question>\n"
    "You may take at most a few retrieval steps. Prefer targeted, specific queries."
)

_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)\s*(?:Action:|Final Answer:)", re.S)
_ACTION_RE = re.compile(r"Action:\s*([A-Za-z_]+)\s*\n\s*Action Input:\s*(.*?)(?:\n|$)", re.S)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.S)


def react_bm25_answer(question, bm25, model, max_steps=6):
    """Text-based ReAct with a retrieve_bm25 tool.

    To make this a FAIR comparison against RAG/RLM (which always see retrieved
    text), we FORCE the first retrieval: the question is run through BM25 up
    front and the result is injected as the first observation. This removes the
    failure mode we observed with gpt-oss -- the model skipping the tool and
    guessing with zero context -- so any further weakness is about reasoning,
    not about never having looked.
    """
    t0 = time.time()
    history = [{"role": "system", "content": REACT_SYS},
               {"role": "user", "content": f"Question: {question}"}]
    pt = ct = 0
    # Force the first retrieval with the raw question so the agent starts from
    # real evidence, exactly like RAG/RLM do.
    first_passages = bm25.retrieve(question, k=8)
    first_obs = "\n\n".join(first_passages) if first_passages else "(no matching passages)"
    history.append({"role": "assistant",
                    "content": f"Thought: I'll start by searching the documents for the question.\n"
                               f"Action: retrieve_bm25\nAction Input: {question}"})
    history.append({"role": "user", "content": f"Observation: {first_obs[:8000]}"})
    obs_count = 1
    for _ in range(max_steps):
        out, p, c = _llm(history, model, max_tokens=4096)
        pt += p; ct += c
        history.append({"role": "assistant", "content": out})
        m_final = _FINAL_RE.search(out)
        if m_final:
            return {"answer": m_final.group(1).strip(), "prompt_tokens": pt,
                    "completion_tokens": ct, "latency_s": time.time() - t0,
                    "n_calls": len([h for h in history if h["role"] == "assistant"]),
                    "extra": {"retrievals": obs_count}}
        m_act = _ACTION_RE.search(out)
        if m_act and m_act.group(1).strip() == "retrieve_bm25":
            query = m_act.group(2).strip().strip('"').strip("'")
            passages = bm25.retrieve(query, k=8)
            obs = "\n\n".join(passages) if passages else "(no matching passages)"
            history.append({"role": "user", "content": f"Observation: {obs[:8000]}"})
            obs_count += 1
            continue
        # model didn't follow format -> nudge once
        history.append({"role": "user",
            "content": "Respond with either an Action block or a Final Answer."})
    # No Final Answer after max steps: salvage a concise answer from the last
    # assistant turn (same fallback RLM uses), so ReAct always returns something
    # rather than "(no final answer produced)".
    last_asst = next((h["content"] for h in reversed(history)
                     if h["role"] == "assistant"), "")
    sal = "(no final answer produced)"
    if last_asst.strip():
        try:
            sal, sp, sc = _salvage_answer(question, last_asst, model)
            pt += sp; ct += sc
            sal = (sal or "").strip() or sal
        except Exception:
            sal = last_asst[:200]
    return {"answer": sal, "prompt_tokens": pt,
            "completion_tokens": ct, "latency_s": time.time() - t0,
            "n_calls": len([h for h in history if h["role"] == "assistant"]),
            "extra": {"retrievals": obs_count, "truncated": True, "salvaged": True}}


# --------------------------------------------------------------------------- #
# 3) RLM -- the `rlm` repo. The whole PDF is the `context` variable; the agent
#    writes code to peek/grep/chunk and recursively calls sub-LLMs.
# --------------------------------------------------------------------------- #
_RLM_CACHE: dict[str, Any] = {}


def _get_rlm(model, env, max_depth, max_budget, max_iterations, log_dir,
             max_tokens=None, per_call_max_tokens=8192):
    """Build (and cache) an RLM instance using the SAME provider/model as baselines.

    max_tokens         : RLM cumulative token budget across ALL sub-calls (input+output).
                         None = unlimited (bounded by max_iterations instead). Reasoning
                         models burn tokens fast, so default is None.
    per_call_max_tokens: per-call completion cap (sampling_args.max_tokens), keeps each
                         call from rambling.
    """
    # Guard: the local `rlm/` repo dir can shadow the installed package when cwd
    # is the project root. Insert the inner package path so `import rlm` resolves.
    import os as _os
    import sys as _sys
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    inner_repo = _os.path.join(root, "rlm")
    if inner_repo not in _sys.path:
        _sys.path.insert(0, inner_repo)
    from rlm import RLM
    from rlm.logger import RLMLogger

    kwargs = dict(
        backend="openai",
        backend_kwargs={
            "model_name": model,
            "api_key": _os.environ.get("OPENAI_API_KEY"),
            "base_url": _os.environ.get("OPENAI_BASE_URL") or None,
            # Cloud models (Ollama cloud) can have slow/loaded spells where a
            # single call takes minutes. Default RLM timeout is 300s; raise it
            # and add retries so RLM doesn't hard-fail on a transient slow call.
            "timeout": 600,
            "max_retries": 4,
        },
        environment=env,
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_budget=max_budget,
        max_tokens=max_tokens,
        sampling_args={"temperature": 0, "max_tokens": per_call_max_tokens},
        sub_sampling_args={"temperature": 0, "max_tokens": per_call_max_tokens},
        logger=RLMLogger(log_dir=log_dir) if log_dir else None,
        verbose=False,
    )
    if env == "docker":
        # DockerREPL needs the host-side proxy; defaults are fine for a python:3.11-slim image.
        kwargs["environment_kwargs"] = {}
    return RLM(**kwargs)


def _looks_like_code(s: str) -> bool:
    """Heuristic: RLM returned raw partial code/reasoning instead of a clean FINAL answer."""
    if not s:
        return False
    head = s[:200].lower().lstrip()
    return ("```repl" in s or "```python" in s or "```" in s
            or s.lstrip().startswith(("import ", "from ", "for ", "while ", "def ", "print("))
            or "\nimport " in s
            or head.startswith(("we need to", "i need to", "let me", "let's",
                                "i'll", "i will", "first", "search the", "look for",
                                "to answer", "i should", "the context")))


def _salvage_answer(question, partial, model):
    """Last-resort: ask the model for a concise final answer from RLM's partial work."""
    msg = [
        {"role": "system", "content": "An agent was working on the question below but did not emit a "
                                      "clean final answer. Extract the concise final answer from its "
                                      "work. Reply with ONLY the answer (a name/phrase), or "
                                      "'I don't know' if it hasn't been found yet."},
        {"role": "user", "content": f"Question: {question}\n\nAgent's last partial work:\n{partial[:6000]}\n\nFinal answer:"},
    ]
    return _llm(msg, model, max_tokens=2048)


def rlm_answer(question, full_text, model, env="local", max_depth=2,
               max_budget=2.0, max_iterations=40, log_dir="./logs", max_tokens=None,
               per_call_max_tokens=8192):
    t0 = time.time()
    rlm = _get_rlm(model, env, max_depth, max_budget, max_iterations, log_dir,
                   max_tokens, per_call_max_tokens)
    prompt = f"{question}\n\nContext:\n{full_text}"
    result = rlm.completion(prompt)
    us = getattr(result, "usage_summary", None)
    pt = getattr(us, "total_input_tokens", 0) if us else 0
    ct = getattr(us, "total_output_tokens", 0) if us else 0
    n_calls = 0
    if us and getattr(us, "model_usage_summaries", None):
        n_calls = sum(getattr(s, "total_calls", 0) or 0
                      for s in us.model_usage_summaries.values())
    ans = (getattr(result, "response", "") or "").strip()
    salvaged = False
    # RLM didn't reach FINAL(...) -> response is raw partial code/reasoning. Salvage it.
    if _looks_like_code(ans):
        try:
            sal, sp, sc = _salvage_answer(question, ans, model)
            n_calls += 1
            pt += sp
            ct += sc
            sal = (sal or "").strip()
            # strip any code fences the salvage model may have emitted
            sal = re.sub(r"```[a-z]*\n?", "", sal).strip()
            if sal and sal.lower() not in ("i don't know", "i don't know.", ""):
                ans = sal
                salvaged = True
        except Exception:
            pass
    return {"answer": ans, "prompt_tokens": pt, "completion_tokens": ct,
            "latency_s": time.time() - t0, "n_calls": n_calls,
            "extra": {"env": env, "max_depth": max_depth, "salvaged": salvaged}}