"""Run every benchmark x model x item, cache raw responses, then build the Excel report.

    uv run python -m bench.run --n 50
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from bench import config, data, report
from bench.clients import chat
from bench.parse import parse

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _load_cache(path: Path) -> dict[tuple, dict]:
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            cache[(rec["model"], rec["bench"], rec["item_id"])] = rec
    return cache


async def _run(models, benches, samples, cache_path: Path) -> None:
    cache = _load_cache(cache_path)
    todo = [
        (m, b, item)
        for b in benches
        for item in samples[b.key]
        for m in models
        if not ((rec := cache.get((m.key, b.key, item["id"]))) and not rec.get("error"))
        and not (m.api == "decisions" and config.build_decision(b, item) is None)  # not applicable, e.g. FinQA
    ]
    print(f"{len(todo)} calls to make ({len(cache)} cached)")
    sem = asyncio.Semaphore(config.CONCURRENCY)
    lock = asyncio.Lock()
    done = 0

    async def one(client, m, b, item):
        nonlocal done
        async with sem:
            payload = config.build_decision(b, item) if m.api == "decisions" else config.build_messages(b, item)
            res = await chat(client, m, payload)
        rec = {"model": m.key, "bench": b.key, "item_id": item["id"], "api": m.api, **res}
        rec["parsed"] = parse(b, res.get("raw_text"), m.api) if not res.get("error") else None
        async with lock:
            with cache_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            done += 1
            status = f"ERROR {res['error'][:80]}" if res.get("error") else rec["parsed"]
            print(f"[{done}/{len(todo)}] {m.key:<18} {b.key:<14} {item['id']:<10} {status}")

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(client, m, b, item) for m, b, item in todo))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.N_DEFAULT, help="items per benchmark")
    ap.add_argument("--models", nargs="*", default=[m.key for m in config.MODELS])
    ap.add_argument("--benchmarks", nargs="*", default=[b.key for b in config.BENCHMARKS])
    ap.add_argument("--tag", default="", help="run name; each tag gets its own raw log and report")
    args = ap.parse_args()

    load_dotenv()
    models = [m for m in config.MODELS if m.key in args.models]
    benches = [b for b in config.BENCHMARKS if b.key in args.benchmarks]
    missing = sorted({m.key_env for m in models if not os.environ.get(m.key_env)})
    if missing:
        raise SystemExit(f"Missing API keys in .env: {', '.join(missing)}")

    samples = {b.key: data.sample(b, args.n, config.SEED) for b in benches}
    RESULTS.mkdir(exist_ok=True)
    run_name = f"{args.tag}_n{args.n}" if args.tag else f"n{args.n}"
    cache_path = RESULTS / f"raw_{run_name}.jsonl"
    print(f"Run '{run_name}': raw log {cache_path.name}")
    asyncio.run(_run(models, benches, samples, cache_path))

    out = report.build(models, benches, samples, _load_cache(cache_path), RESULTS, args.n, run_name)
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
