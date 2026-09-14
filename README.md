# RLM vs RAG vs ReAct+BM25 — comparison harness

A fair, same-model comparison of three answer strategies on a long PDF:

| Method | How it answers |
|---|---|
| **RAG** | Embed/TF-IDF retrieve top-k chunks, stuff them into one prompt, answer. |
| **ReAct+BM25** | Text-based ReAct loop (Thought/Action/Observation) calling a `retrieve_bm25` tool — adaptive, multi-query. |
| **RLM** | The `rlm` repo. The whole PDF text is exposed as a `context` REPL variable; the model writes code (peek/grep/chunk/summarize) and can spawn recursive sub-LLMs. |

All three use the **same base model and provider** (OpenAI or any OpenAI-compatible endpoint via `OPENAI_BASE_URL`) and the **same PDF + questions**. The only variable is the retrieval/reasoning strategy.

Scoring is **SQuAD-style Exact Match + token F1** (max over multiple acceptable gold answers). Per-question **tokens / latency / #LLM-calls** are recorded and aggregated. Results go to a CSV; RLM trajectories are saved as JSONL under `logs/`.

---

## Setup

The deps are installed in the project's `.venv` (one level up):

```bash
cd /Users/XYZ/RLM-Project
.venv/bin/pip install pdfplumber rank_bm25 scikit-learn numpy openai
.venv/bin/pip install -e ./rlm          # the RLM repo, editable
```

## How this harness uses the `rlm` repo

**RLM is an inference-time scaffold, not a retrieval method.** The whole corpus
is loaded as a **string variable inside a Python REPL** — kept *out* of the
model's context window — and the model **writes code** (`peek()`, `grep()`,
`chunk()`, `summarize()`) to inspect it. It can spawn **recursive sub-LLMs**
(`sub_rlm()` / `llm_query()`) to work sub-problems, then emits `FINAL(answer)`.
This is exactly what lets RLM follow a multi-hop chain across documents that
keyword search cannot surface. Paper: arXiv:2512.24601 · upstream repo:
github.com/alexzhang13/rlm.

### 1. Get the `rlm` repo as a sibling directory

The harness expects `rlm` to live at `../rlm` (i.e. one level up from
`comparison/`), installed editable into the same virtualenv:

```bash
cd /Users/XYZ/RLM-Project
git clone https://github.com/alexzhang13/rlm.git rlm     # or your fork
.venv/bin/pip install -e ./rlm
```

### 2. How `methods.py` calls into it

`rlm_answer()` in `methods.py` is the only integration point:

1. It inserts the inner package path so `import rlm` resolves (the local `rlm/`
   dir can shadow the installed package when cwd is the project root), then
   imports `from rlm import RLM` and `from rlm.logger import RLMLogger`.
2. It builds an `RLM(backend="openai", backend_kwargs={model_name, api_key,
   base_url, timeout, max_retries}, environment="local", ...)` using the
   **same OpenAI-compatible client** RAG and ReAct use — so the base model is
   held constant across all three methods; only the strategy differs.
3. It calls `rlm.completion(prompt)` where
   `prompt = question + "\n\nContext:\n" + full_text`. RLM exposes `full_text`
   to the model **only as a REPL `context` variable** — not as tokens the model
   attends to — which is how it handles >10M-token corpora.
4. The answer is read from `result.response` (RLM emits `FINAL(answer)`); token
   and call counts come from `result.usage_summary`. If RLM hits the iteration
   cap without emitting `FINAL(...)`, a salvage call extracts a best guess from
   its partial work so every query still returns something.

RLM trajectory logs are written to `--log-dir` as JSONL and can be viewed with
the `rlm` repo's own visualizer.

### RLM knobs

| Flag | Default | Meaning |
|---|---|---|
| `--rlm-model` | = `--model` | Model for RLM only (e.g. a coding/non-thinking model) |
| `--rlm-env` | `local` | REPL: `local` / `docker` / `e2b` / `modal` |
| `--rlm-depth` | `2` | Max recursive sub-LLM depth |
| `--rlm-iters` | `60` | Max RLM iterations (hard multi-hop needs ~60) |
| `--rlm-budget` | `3.0` | USD budget cap |
| `--rlm-per-call-tokens` | `8192` | Per-call completion cap |
| `--log-dir` | `./logs_bc` | RLM JSONL trajectories |

> **Model choice matters.** The RLM paper deliberately uses a *non-thinking*
> model (Qwen3-8B) — thinking/reasoning tokens eat the output budget that
> should go to scaffold code + `FINAL(...)`. We found the same: a
> thinking-heavy model (gpt-oss:120b) scored RLM 0.20; switching to
> `kimi-k2.7-code:cloud` (coding-focused, low thinking) lifted it to 0.72.

## Two inputs you need to actually run it

1. **A corpus** — a single PDF **or a directory of text/PDF files**. A single PDF
   past ~10M tokens (~40M chars, ~15K pages) is rare in the wild, so for the
   >10M-token regime use a **corpus directory**. The canonical choice — the same
   one the RLM paper (arXiv:2512.24601) uses — is **BrowseComp-Plus**
   (see "Large-corpus experiment" below).
2. **An API key** — `OPENAI_API_KEY`. Optional `OPENAI_BASE_URL` to point at
   OpenRouter / DeepSeek / a local vLLM / **Ollama**. All three methods use this
   same client.

```bash
export OPENAI_API_KEY=sk-...
# optional: export OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

### Using Ollama (local, no cloud key)

Ollama exposes an OpenAI-compatible API. The `/v1` path is required.

```bash
# 1. pull a chat model + an embedding model (RLM needs a model that writes good code)
ollama pull qwen2.5-coder:7b          # or qwen3:8b, llama3.1:8b
ollama pull nomic-embed-text          # for the dense RAG retriever

# 2. point the harness at Ollama
export OPENAI_API_KEY=ollama                       # required by SDK, ignored by Ollama
export OPENAI_BASE_URL=http://localhost:11434/v1

# 3. run with an Ollama model tag
../.venv/bin/python compare.py --pdf corpus/bc_plus_100k \
    --model qwen2.5-coder:7b --embed-model nomic-embed-text --out results.csv
```

**Large `num_ctx` for RLM.** RLM keeps the long text out of context (it lives in
the REPL), but its root/sub-LLM calls still accumulate stdout snippets. Ollama's
default context is small, so create a wide-context alias:

```bash
printf 'FROM qwen2.5-coder:7b\nPARAMETER num_ctx 32768\n' > Modfile.ctx
ollama create qwen2.5-coder-32k -f Modfile.ctx
# then run with --model qwen2.5-coder-32k
```

**Honest caveat:** the RLM paper used GPT-5. A local 7–8B model writes weaker code
and reasons less, so absolute accuracy will be lower and RLM's emergent
peek/grep/partition strategies may show up less reliably. The *comparison
structure* (RLM vs RAG vs ReAct+BM25, same model) is still valid. The paper's own
distilled recursive model is RLM-Qwen3-8B, so `qwen3:8b` is the closest local match.

### Large-corpus experiment (the RLM paper's regime)

The RLM paper evaluates on **BrowseComp-Plus** with a ~1,000-document subset
= **6M–11M tokens** of multi-hop, "unknown information location" questions —
exactly your use case.

```bash
# full corpus: 100,195 plain-text .txt files (~3.2 GB unzipped)
# from HuggingFace: lossisnotannumber/browsecomp-plus-100k-corpus-as-local-folder
huggingface-cli download lossisnotannumber/browsecomp-plus-100k-corpus-as-local-folder \
    --repo-type dataset --local-dir corpus/bc_plus_100k

# queries + gold answers (obfuscated; decrypt with texttron/BrowseComp-Plus scripts)
huggingface-cli download Tevatron/browsecomp-plus --repo-type dataset --local-dir bc_queries
```

At ~7K tokens/doc, **~1,500 docs ≈ 10M tokens**. To run on a controlled subset,
point `--pdf` at a subfolder of ~1,000–1,500 .txt files. The harness loads any
directory of `.txt`/`.md`/`.pdf` recursively.

## Dry run (no API key needed)

Verifies PDF parsing, chunking, and both retrievers:

```bash
cd comparison
../.venv/bin/python compare.py --pdf /path/to/doc.pdf --dry-run
```

## Full run

```bash
../.venv/bin/python compare.py \
  --pdf /path/to/big.pdf \
  --questions questions.example.jsonl \
  --model gpt-5-mini \
  --out results.csv
```

### Knobs

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `gpt-5-mini` | Base model for **all three** methods |
| `--methods` | `rag,react_bm25,rlm` | Comma-separated subset to run |
| `--rag-retriever` | `dense` | `dense` (OpenAI embeddings, TF-IDF fallback) or `bm25` |
| `--embed-model` | `text-embedding-3-small` | OpenAI embedding model |
| `--top-k` | `12` | Chunks retrieved per query |
| `--chunk-size` / `--chunk-overlap` | `1500` / `300` | Chunking |
| `--react-steps` | `6` | Max ReAct iterations |
| `--rlm-env` | `local` | RLM REPL: `local` / `docker` / `e2b` / `modal` |
| `--rlm-depth` | `2` | Max recursive sub-LLM depth |
| `--rlm-budget` | `2.0` | RLM USD budget cap |
| `--rlm-iters` | `40` | RLM max iterations |
| `--log-dir` | `./logs` | RLM JSONL trajectories |
| `--limit` | `0` | Only first N questions (0 = all) |

## Questions file

JSONL, one per line. `answer` may be a string or a list of acceptable answers
(empty string = unscored / open-ended, scored as F1 against `""` i.e. only
credit if the model also says nothing — so fill golds for real evaluation):

```json
{"question": "What is the candidate's total years of experience?", "answer": "2 years"}
{"question": "Which languages are listed as strongest?", "answer": ["Python", "Python and SQL"]}
```

## Output

Console: per-question `EM/F1/tokens/latency` + an aggregate summary table.
CSV (`results.csv`): one row per (question, method) with the answer, scores, and cost.
`logs/`: the RLM agent's full trajectory (JSONL) — visualize with the `rlm`
repo's own visualizer.

## Why this is a fair comparison

- Same model + provider for all three (only the strategy differs).
- Same PDF text and same questions.
- Same scoring (SQuAD EM/F1) applied identically.
- Cost/latency reported so accuracy can be read against tokens and time.

RLM's advantage on very long, multi-hop, "unknown information location" tasks is
exactly the regime where a single stuff-the-context RAG call fails (context rot,
lost-in-the-middle) and where ReAct's fixed tool set can't re-chunk or summarize
intermediate results — RLM can do all of that at inference time.
