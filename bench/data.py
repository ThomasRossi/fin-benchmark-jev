"""Load benchmark datasets from Hugging Face and draw deterministic samples."""

import hashlib
import random
import re
from collections import defaultdict

from datasets import load_dataset

from bench import config
from bench.config import Benchmark


def _load(bench: Benchmark):
    return load_dataset(bench.hf_path, split=bench.split, revision=bench.hf_revision)


def _twitter_rows(bench: Benchmark) -> list[dict]:
    return [
        {"id": f"tw{i}", "text": r["text"], "gold": config.TWITTER_TOPICS[r["label"]]}
        for i, r in enumerate(_load(bench))
    ]


def _rows(bench: Benchmark) -> list[dict]:
    if bench.key == "fpb":
        return [{"id": r["id"], "text": r["text"], "gold": r["answer"]} for r in _load(bench)]
    if bench.key == "twitter_topic":
        return _twitter_rows(bench)
    if bench.key == "finqa":
        rows = []
        for r in _load(bench):
            # Keep numeric answers only (drops ~20 yes/no items) and near-zero values a 2-decimal answer can't express.
            if not re.fullmatch(r"-?\d+(\.\d+)?", r["answer"]) or abs(float(r["answer"])) < 1e-3:
                continue
            # Drop FLARE's own instruction line and trailing "Answer:"; our prompt supplies both.
            body = r["query"].split("\n", 1)[1].removesuffix("Answer:").strip()
            rows.append({"id": r["id"], "text": body, "gold": r["answer"]})
        return rows
    raise ValueError(f"unknown benchmark {bench.key}")


def _balanced(bench: Benchmark, n: int, rng: random.Random) -> list[dict]:
    """Round-robin across labels so every label is represented (balanced, not proportional)."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in _rows(bench):
        by_label[row["gold"]].append(row)
    for rows in by_label.values():
        rng.shuffle(rows)

    out: list[dict] = []
    labels = [l for l in bench.labels if by_label[l]]
    depth = 0
    while len(out) < n and any(depth < len(by_label[l]) for l in labels):
        for label in labels:
            if len(out) < n and depth < len(by_label[label]):
                out.append(by_label[label][depth])
        depth += 1
    rng.shuffle(out)
    return out


def _idsets(bench: Benchmark, n: int, rng: random.Random) -> list[dict]:
    """Each item: GROUP_SIZE numbered tweets, 1-5 of them on the target topic; gold = their IDs.

    Target tweets are drawn from a per-topic shuffled queue, so none repeats until the topic's pool is used up.
    Filler tweets never come from the target's NEIGHBOR_TOPICS.
    """
    rows = _twitter_rows(bench)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["gold"]].append(row)
    queues: dict[str, list[dict]] = {}
    items = []
    for k in range(n):
        topic = bench.labels[k % len(bench.labels)]
        n_pos = rng.randint(1, 5)
        positives = []
        while len(positives) < n_pos:
            if not queues.get(topic):
                queues[topic] = rng.sample(by_label[topic], len(by_label[topic]))
            candidate = queues[topic].pop()
            if candidate not in positives:
                positives.append(candidate)
        excluded = {topic} | config.NEIGHBOR_TOPICS[topic]
        others = [r for r in rows if r["gold"] not in excluded]
        group = positives + rng.sample(others, config.GROUP_SIZE - n_pos)
        rng.shuffle(group)
        lines = [f"{i}. {' '.join(r['text'].split())}" for i, r in enumerate(group, 1)]
        gold = ", ".join(str(i) for i, r in enumerate(group, 1) if r["gold"] == topic)
        text = "\n".join(lines)
        items.append({
            # Content hash in the ID so a changed generator never reuses cached answers for different items.
            "id": f"mh{k}-{hashlib.sha1(text.encode()).hexdigest()[:8]}",
            "text": text,
            "gold": gold,
            "task": bench.task.format(n=config.GROUP_SIZE, topic=topic),
        })
    return items


def sample(bench: Benchmark, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if bench.kind == "label":
        return _balanced(bench, n, rng)
    if bench.kind == "number":
        rows = _rows(bench)
        return rng.sample(rows, min(n, len(rows)))
    if bench.kind == "idset":
        return _idsets(bench, n, rng)
    raise ValueError(f"unknown kind {bench.kind}")
