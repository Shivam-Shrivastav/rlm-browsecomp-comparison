"""Index/extract docs from the BrowseComp-Plus corpus parquet shards.

The corpus (Tevatron/browsecomp-plus-corpus) is 7 parquet files with columns
docid, text, url. We only ever need a small subset of docs per query
(gold + evidence + sampled negatives), so we scan the lightweight `docid`
column to locate rows, then read just those texts.
"""
from __future__ import annotations

import glob
import os
import pyarrow.parquet as pq


def _shards(shard_dir: str):
    return sorted(glob.glob(os.path.join(shard_dir, "data", "train-*.parquet")))


def extract_docs(shard_dir: str, docids: set[str]) -> dict[str, dict]:
    """Return {docid: {"text":..., "url":...}} for every docid found."""
    wanted = {str(d) for d in docids}
    out: dict[str, dict] = {}
    for shard in _shards(shard_dir):
        if not wanted:
            break
        # read only docid column first to find row positions
        t = pq.read_table(shard, columns=["docid"])
        ids = t.column("docid").to_pylist()
        hits = [i for i, d in enumerate(ids) if str(d) in wanted]
        if not hits:
            continue
        # now read full text for just those rows
        full = pq.read_table(shard, columns=["docid", "text", "url"])
        text_col = full.column("text").to_pylist()
        url_col = full.column("url").to_pylist()
        id_col = full.column("docid").to_pylist()
        for i in hits:
            d = str(id_col[i])
            if d not in out:
                out[d] = {"text": text_col[i] or "", "url": url_col[i] or ""}
                wanted.discard(d)
    return out


def write_corpus_dir(docs: dict[str, dict], out_dir: str):
    """Write extracted docs as .txt files for the harness's load_text()."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for did, d in docs.items():
        txt = (d.get("text") or "").strip()
        if not txt:
            continue
        with open(os.path.join(out_dir, f"{did}.txt"), "w", encoding="utf-8") as f:
            f.write(f"docid: {did}\nurl: {d.get('url','')}\n\n{txt}")
        n += 1
    return n


if __name__ == "__main__":
    import sys
    shard_dir, out_dir = sys.argv[1], sys.argv[2]
    docids = set(sys.argv[3:])
    docs = extract_docs(shard_dir, docids)
    n = write_corpus_dir(docs, out_dir)
    print(f"extracted {n}/{len(docids)} docs -> {out_dir}")