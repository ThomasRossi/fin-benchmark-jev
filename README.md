# fin-benchmark-jev

Benchmarks **Jev 1.13** (TypeSafe's decisions model, `typesafe/jev-1.13` on OpenRouter's `/api/alpha/decisions` endpoint)
against **Qwen 3.8 27B on Cerebras** on four financial tasks with constrained output: a label, a number, or a set of IDs.

Two secondary models are run as controls and shown greyed out in the reports:

- **Qwen 3.8 27B via OpenRouter, pinned to DeepInfra bf16**: checks that the Cerebras deployment isn't degraded.
- **Jev Router** (`typesafe/jev-router`): a different product from Jev 1.13. It forwards each request to a third-party
  LLM (mostly GPT-6 Luna and DeepSeek v4.1 Flash in our runs).

## Benchmarks

All samples are deterministic (seed 42) and every dataset is pinned to a Hugging Face revision.

| Benchmark | Dataset | Task | Scoring |
|---|---|---|---|
| **FPB** | `ChanceFocus/en-fpb` (test) | Sentiment of a financial-news sentence: positive / neutral / negative | Exact label; accuracy + macro-F1 |
| **Twitter Topic** | `zeroshot/twitter-financial-news-topic` (validation) | Topic of a finance tweet, 20 classes | Exact label; accuracy + macro-F1 |
| **FinQA** | `ChanceFocus/flare-finqa` (test) | Multi-step arithmetic over a report excerpt and table, answer is one number | Within 1% of gold (also accepting a x100 percent/ratio mismatch) |
| **Multi-headline** | `zeroshot/twitter-financial-news-topic` (train+validation) | Given 40 tweets, list the 1-5 that are about a target topic | Exact ID set; mean set-F1 as partial credit |

The label benchmarks are sampled balanced across labels. Multi-headline items are built synthetically: filler tweets never
come from topics close to the target (`NEIGHBOR_TOPICS` in `bench/config.py`), so the dataset's single label doesn't mark a
reasonable pick as wrong.

**Jev 1.13 does not run FinQA.** The decisions endpoint only answers typed questions (multiple choice, yes/no probability)
about a given state and cannot return a number. For Multi-headline it gets each tweet tagged `T01`-`T40` and one yes/no
question per tweet (yes = probability > 0.5), following TypeSafe's line-search pattern.

## Results (n = 200 per benchmark)

| Benchmark | Jev 1.13 | Qwen 3.8 27B (Cerebras) | Jev Router (secondary) |
|---|---|---|---|
| FPB | 87.5% | 87.0% | 89.0% |
| Twitter Topic | 71.0% | 73.0% | 76.0% |
| FinQA | n/a | 80.0% | 80.0% |
| Multi-headline | 42.0% | 47.5% | 40.5% |

Latency, wall-clock per request, median / p95 in ms:

| | Jev 1.13 | Qwen 3.8 27B (Cerebras) |
|---|---|---|
| Labels (FPB + Twitter Topic) | 468 / 646 | 364 / 955 |
| FinQA | n/a | 558 / 1,755 |
| Multi-headline | 549 / 732 | 1,010 / 3,793 |
| **Total cost, the 3 shared benchmarks (600 calls)** | **$0.031** | **$0.979** |

Each item is 0.5 percentage points, so accuracy gaps of a few points are within noise. Jev 1.13's cost comes from
OpenRouter's `usage.cost`. Cerebras returns no cost, so Qwen's is computed from token counts at list price
($0.99 / $1.49 per million input / output tokens) and is mostly reasoning tokens (`reasoning_effort: high`).

The full reports, with every input, raw output, parsed answer and score, are in `results/`:

- `jev_vs_qwen_jev113_p1_n200_*.xlsx`: FPB and Twitter Topic
- `jev_vs_qwen_jev113_p2_n200_*.xlsx`: FinQA and Multi-headline
- `raw_jev113_p{1,2}_n200.jsonl`: the raw API responses behind them. The runner uses these as its cache, so rerunning the
  same command rebuilds the report without new API calls.

## Running

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```sh
uv sync
cp .env.example .env   # then fill in OPENROUTER_API_KEY and CEREBRAS_API_KEY
```

Reproduce the published runs:

```sh
uv run python -m bench.run --n 200 --tag jev113_p1 --benchmarks fpb twitter_topic
uv run python -m bench.run --n 200 --tag jev113_p2 --benchmarks finqa multi_headline \
    --models jev_1_13 qwen_cerebras jev_router
```

`--models` and `--benchmarks` take the keys defined in `bench/config.py`. Every run with a new `--tag` gets its own raw log
and report. Each report build writes a new timestamped `.xlsx` and never overwrites an old one.

## Layout

| File | Purpose |
|---|---|
| `bench/config.py` | Models, benchmarks, prompts and request builders: everything that defines a run |
| `bench/data.py` | Loads the datasets and draws the deterministic samples |
| `bench/clients.py` | API calls with retries, timing and cost accounting |
| `bench/parse.py` | Maps raw replies to canonical answers and scores them |
| `bench/report.py` | Builds the Excel report |
| `bench/run.py` | CLI entry point |
