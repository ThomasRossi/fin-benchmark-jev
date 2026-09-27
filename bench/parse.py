"""Map a model's raw reply onto a canonical answer, and score it against gold."""

import json
import re

from bench import config
from bench.config import Benchmark

INVALID = "INVALID"
NUM_RTOL = 0.01  # 1% relative tolerance for numeric answers


def _strip(raw: str) -> str:
    # Drop any <think>...</think> block a model leaks into content.
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.S)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip("\"'`*.:-").lower())


def parse_label(raw: str | None, labels: tuple[str, ...]) -> str:
    if not raw:
        return INVALID
    norm = _norm(_strip(raw))
    for label in labels:
        if norm == label.lower():
            return label
    # Otherwise accept only a single, unambiguous label mentioned in the reply.
    # Longest labels first so "Stock Movement" isn't shadowed by a shorter overlapping label.
    hits = [l for l in sorted(labels, key=len, reverse=True) if l.lower() in norm]
    hits = [h for h in hits if not any(h != o and h.lower() in o.lower() for o in hits)]
    return hits[0] if len(hits) == 1 else INVALID


def parse_number(raw: str | None) -> str:
    if not raw:
        return INVALID
    text = _strip(raw).replace("−", "-")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+", text)
    if not nums:
        return INVALID
    # The last number is the final answer if a model adds any working despite the instructions.
    return f"{float(nums[-1].replace(',', '')):g}"


def parse_idset(raw: str | None) -> str:
    if not raw:
        return INVALID
    text = _strip(raw)
    ids = sorted({int(x) for x in re.findall(r"\b\d{1,3}\b", text) if 1 <= int(x) <= config.GROUP_SIZE})
    if ids:
        return ", ".join(map(str, ids))
    return "NONE" if re.search(r"\bnone\b", text, re.I) else INVALID


def parse_decision(bench: Benchmark, raw: str | None) -> str:
    """Jev 1.13 answers (JSON of typed answers) -> the same canonical form as the chat parsers."""
    if not raw:
        return INVALID
    answers = json.loads(raw)
    if bench.kind == "label":
        return answers["answer"]["choice"]
    ids = sorted(int(k[1:]) for k, a in answers.items() if a["noul"] > config.NOUL_YES)
    return ", ".join(map(str, ids)) if ids else "NONE"


def parse(bench: Benchmark, raw: str | None, api: str = "chat") -> str:
    if api == "decisions":
        return parse_decision(bench, raw)
    if bench.kind == "label":
        return parse_label(raw, bench.labels)
    if bench.kind == "number":
        return parse_number(raw)
    return parse_idset(raw)


def _ids(s: str) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", s)} if s not in (INVALID, "NONE") else set()


def set_f1(pred: str, gold: str) -> float:
    p, g = _ids(pred), _ids(gold)
    if not p and not g:
        return 1.0
    tp = len(p & g)
    return 2 * tp / (len(p) + len(g))


def score(bench: Benchmark, pred: str, gold: str) -> int:
    """1 if correct. Numbers: within 1% of gold, also accepting a x100 percent/ratio mismatch. ID sets: exact set."""
    if bench.kind == "label":
        return int(pred == gold)
    if pred in (INVALID, "ERROR", "MISSING"):
        return 0
    if bench.kind == "number":
        p, g = float(pred), float(gold)
        return int(any(abs(c - g) <= NUM_RTOL * abs(g) for c in (p, p / 100, p * 100)))
    return int(_ids(pred) == _ids(gold) and pred != INVALID)
