"""SQuAD-style scoring: Exact Match + token F1.

Handles multiple acceptable gold answers (takes the max), and is forgiving
about articles, punctuation, and case -- the standard SQuAD normalization.
"""
from __future__ import annotations

import re
import string


def _normalize(s: str) -> str:
    """Lowercase, drop articles & punctuation, strip accents, collapse whitespace.

    Stripping diacritics means "Rodríguez" == "Rodriguez" and "café" == "cafe" --
    a correct answer with the wrong accenting should not be marked wrong.
    (SQuAD-style normalization + Unicode accent folding via NFKD decomposition.)
    """
    import unicodedata
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # fold accents/diacritics to their base letter: decompose then drop combining marks
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # remove punctuation
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def _tokens(s: str) -> list[str]:
    return _normalize(s).split()


def _f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    # weighted by token frequency
    from collections import Counter

    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def _em(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def _contains(pred: str, gold: str) -> float:
    """Containment: the normalized gold answer appears as a substring of the
    normalized prediction. Credits generative answers that wrap the gold in
    prose -- e.g. pred "The new youth-focused brand was Vakkorama" vs gold
    "Vakkorama" -> strict SQuAD F1~0.33 but containment=1.0 (a correct answer).
    """
    npred, ngold = _normalize(pred), _normalize(gold)
    if not ngold:
        return 0.0
    return float(ngold in npred)


def score(prediction: str, gold) -> dict:
    """Score one prediction against a gold answer or list of golds (max over golds).

    Returns:
        em          : strict SQuAD exact match
        f1          : strict SQuAD token F1
        containment : normalized gold substring in normalized prediction
        correct     : lenient correctness = max(em, containment)  -- the metric
                      to judge methods by, since BrowseComp answers are short
                      gold strings that a fair answerer may phrase in a sentence.
    """
    golds = gold if isinstance(gold, list) else [gold]
    em = max(_em(prediction, g) for g in golds) if golds else 0.0
    f1 = max(_f1(prediction, g) for g in golds) if golds else 0.0
    cont = max(_contains(prediction, g) for g in golds) if golds else 0.0
    return {"em": em, "f1": f1, "containment": cont, "correct": max(em, cont)}