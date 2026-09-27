"""Models, benchmarks and prompts. Everything a reviewer needs to know about a run lives here."""

from dataclasses import dataclass, field

N_DEFAULT = 50
SEED = 42
CONCURRENCY = 4
MAX_TOKENS = 8192  # headroom for Qwen's reasoning tokens; the final answer is one label
QWEN_REASONING_EFFORT = "high"  # Cerebras default; set explicitly so both Qwen columns match

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"  # Jev 1.13 (decisions models only)
NOUL_YES = 0.5  # Jev yes/no threshold for multi-headline, as in TypeSafe's docs
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"


@dataclass(frozen=True)
class Model:
    key: str  # short column name used in the Excel
    label: str  # human-readable description
    url: str
    model_id: str
    key_env: str
    extra_body: dict = field(default_factory=dict)
    # $ per token, only for providers that don't return usage.cost (Cerebras)
    price_in: float | None = None
    price_out: float | None = None
    primary: bool = True  # secondary models are shown greyed out and excluded from the agreement column
    api: str = "chat"  # chat: OpenAI-style chat completions | decisions: TypeSafe state + typed questions


MODELS = [
    Model(
        key="jev_1_13",
        label="Jev 1.13 (TypeSafe System One decisions model) via OpenRouter decisions endpoint",
        url=OPENROUTER_DECISIONS_URL,
        model_id="typesafe/jev-1.13",
        key_env="OPENROUTER_API_KEY",
        api="decisions",
    ),
    Model(
        key="qwen_cerebras",
        label="Qwen 3.8 27B on Cerebras (direct API)",
        url=CEREBRAS_URL,
        model_id="qwen-3.8-27b",
        key_env="CEREBRAS_API_KEY",
        extra_body={"temperature": 0, "max_tokens": MAX_TOKENS, "reasoning_effort": QWEN_REASONING_EFFORT},
        price_in=0.99e-6,
        price_out=1.49e-6,
    ),
    Model(
        key="qwen_or_deepinfra",
        label="Qwen 3.8 27B via OpenRouter, pinned to DeepInfra bf16 (secondary control)",
        url=OPENROUTER_URL,
        model_id="qwen/qwen3.8-27b",
        key_env="OPENROUTER_API_KEY",
        extra_body={
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "reasoning": {"effort": QWEN_REASONING_EFFORT},
            "provider": {"order": ["DeepInfra"], "allow_fallbacks": False},
            "usage": {"include": True},
        },
        primary=False,
    ),
    Model(
        key="jev_router",
        label="Jev Router (typesafe/jev-router): routes each request to a third-party LLM (secondary)",
        url=OPENROUTER_URL,
        model_id="typesafe/jev-router",
        key_env="OPENROUTER_API_KEY",
        # The router lists no supported parameters: send model + messages only.
        extra_body={"usage": {"include": True}},
        primary=False,
    ),
]

GROUP_SIZE = 40  # tweets per multi-headline item

SYSTEM_PROMPTS = {
    "label": (
        "You are a financial text classifier. "
        "Reply with exactly one label from the allowed list, copied verbatim. "
        "Do not add explanations, punctuation or any other text."
    ),
    "number": (
        "You are a financial analyst. Answer the question using only the given context. "
        "Reply with only the final numeric answer: no words, units, currency symbols or working."
    ),
    "idset": (
        "You are a financial news screener. "
        "Reply with only the IDs of the matching items as a comma-separated list, or NONE if no item matches. "
        "Do not add explanations or any other text."
    ),
}

USER_TEMPLATES = {
    "label": "Task: {task}\n\nAllowed labels:\n{labels}\n\nText:\n{text}\n\nLabel:",
    "number": (
        "Task: {task}\n\n{text}\n\n"
        "Answer format: a single number rounded to 2 decimals. Give percentages and percentage changes "
        "in percent (14.46 for 14.46%), without the % sign.\n\nAnswer:"
    ),
    "idset": "Task: {task}\n\nTweets:\n{text}\n\nMatching IDs:",
}


@dataclass(frozen=True)
class Benchmark:
    key: str
    sheet: str
    task: str
    hf_path: str
    hf_revision: str
    split: str
    labels: tuple[str, ...] = ()  # label kind: allowed labels; idset kind: topics to ask about
    kind: str = "label"  # label | number | idset


TWITTER_TOPICS = (
    "Analyst Update",
    "Fed | Central Banks",
    "Company | Product News",
    "Treasuries | Corporate Debt",
    "Dividend",
    "Earnings",
    "Energy | Oil",
    "Financials",
    "Currencies",
    "General News | Opinion",
    "Gold | Metals | Materials",
    "IPO",
    "Legal | Regulation",
    "M&A | Investments",
    "Macro",
    "Markets",
    "Politics",
    "Personnel Change",
    "Stock Commentary",
    "Stock Movement",
)

BENCHMARKS = [
    Benchmark(
        key="fpb",
        sheet="FPB",
        task="Classify the sentiment of this sentence from a financial news article, from an investor's point of view.",
        hf_path="ChanceFocus/en-fpb",
        hf_revision="7de7bf7af4b987fce965cc324ce0d898da679e76",
        split="test",
        labels=("positive", "neutral", "negative"),
    ),
    Benchmark(
        key="twitter_topic",
        sheet="Twitter Topic",
        task="Classify the topic of this finance-related tweet.",
        hf_path="zeroshot/twitter-financial-news-topic",
        hf_revision="acbc8af2a35ccf0916124efcbe9e6cf25f191012",
        split="validation",
        labels=TWITTER_TOPICS,
    ),
    Benchmark(
        key="finqa",
        sheet="FinQA",
        task="Answer the financial question using the report excerpt and table below.",
        hf_path="ChanceFocus/flare-finqa",
        hf_revision="4343aee8c5d8d4469b411407f0e3fb3eaeba6bff",
        split="test",
        kind="number",
    ),
    Benchmark(
        key="multi_headline",
        sheet="Multi-headline",
        task="Which of these {n} tweets are about the topic '{topic}'?",
        hf_path="zeroshot/twitter-financial-news-topic",
        hf_revision="acbc8af2a35ccf0916124efcbe9e6cf25f191012",
        # Both splits (21k tweets, no overlap) so rare topics have enough distinct tweets; nothing is trained here.
        split="train+validation",
        # Topics with clear-cut definitions, to keep dataset label noise out of the retrieval score.
        labels=(
            "Dividend",
            "Earnings",
            "IPO",
            "Personnel Change",
            "Fed | Central Banks",
            "M&A | Investments",
            "Energy | Oil",
            "Currencies",
            "Gold | Metals | Materials",
            "Legal | Regulation",
        ),
        kind="idset",
    ),
]


# Topics never used as filler for a target topic: their tweets often also fit the target,
# and the dataset's single label would then count a reasonable pick as wrong.
NEIGHBOR_TOPICS = {
    "Dividend": {"Stock Commentary", "Earnings", "Company | Product News"},
    "Earnings": {"Stock Commentary", "Stock Movement", "Company | Product News", "Analyst Update", "Dividend"},
    "IPO": {"Stock Commentary", "Stock Movement", "Company | Product News", "M&A | Investments", "Markets"},
    "Personnel Change": {"Company | Product News", "General News | Opinion"},
    "Fed | Central Banks": {"Macro", "Treasuries | Corporate Debt", "Markets", "Currencies", "General News | Opinion"},
    "M&A | Investments": {"Company | Product News", "Stock Commentary", "Stock Movement", "IPO"},
    "Energy | Oil": {"Gold | Metals | Materials", "Macro", "Markets", "Politics", "Stock Commentary",
                     "Stock Movement", "Company | Product News"},
    "Currencies": {"Macro", "Fed | Central Banks", "Markets", "Gold | Metals | Materials"},
    "Gold | Metals | Materials": {"Energy | Oil", "Markets", "Macro", "Stock Movement", "Stock Commentary", "Currencies"},
    "Legal | Regulation": {"Politics", "Company | Product News", "General News | Opinion", "Financials"},
}


def build_decision(bench: Benchmark, item: dict) -> dict | None:
    """TypeSafe request body (state + typed questions), or None where Jev can't answer (numeric generation)."""
    if bench.kind == "label":
        return {
            "state": item["text"],
            "questions": {"answer": {"type": "choice", "instructions": bench.task,
                                     "criteria": {label: None for label in bench.labels}}},
        }
    if bench.kind == "idset":
        # TypeSafe's line-search pattern: tag each line with a short distinctive ID and ask about the tag,
        # not about a list position (positional references are the "indirection" Jev 1.13 is weak at).
        tweets = [line.split(". ", 1)[1] for line in item["text"].split("\n")]
        topic = item["task"].split("'")[1]
        return {
            "state": "\n".join(f"T{i + 1:02d}| {t}" for i, t in enumerate(tweets)),
            "questions": {f"t{i + 1}": {"type": "noul", "instructions": f"Is tweet T{i + 1:02d} about the topic '{topic}'?"}
                          for i in range(len(tweets))},
        }
    return None


def build_messages(bench: Benchmark, item: dict) -> list[dict]:
    user = USER_TEMPLATES[bench.kind].format(
        task=item.get("task", bench.task),
        labels="\n".join(f"- {l}" for l in bench.labels),
        text=item["text"],
    )
    return [{"role": "system", "content": SYSTEM_PROMPTS[bench.kind]}, {"role": "user", "content": user}]
