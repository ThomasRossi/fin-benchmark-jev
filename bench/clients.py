"""One OpenAI-compatible chat call with retries, timing and cost accounting."""

import asyncio
import json
import os
import time
from datetime import datetime, timezone

import httpx

from bench.config import Model

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5


async def chat(client: httpx.AsyncClient, model: Model, payload: list[dict] | dict) -> dict:
    """payload: chat messages, or for api == "decisions" a {state, questions} dict."""
    headers = {"Authorization": f"Bearer {os.environ[model.key_env]}"}
    if model.api == "decisions":
        body = {"model": model.model_id, **payload}
    else:
        body = {"model": model.model_id, "messages": payload, **model.extra_body}
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        start = time.perf_counter()
        try:
            resp = await client.post(model.url, json=body, headers=headers, timeout=180)
            latency_ms = (time.perf_counter() - start) * 1000
            if resp.status_code in RETRY_STATUS:
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            elif resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:500]}", "attempts": attempt + 1}
            else:
                data = resp.json()
                if "error" in data:  # OpenRouter can return 200 with an error body
                    last_error = str(data["error"])[:500]
                else:
                    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    return {**_extract(model, data, latency_ms), "attempts": attempt + 1, "finished_at": finished}
        except (httpx.TransportError, ValueError) as e:
            last_error = f"{type(e).__name__}: {e}"
        await asyncio.sleep(2**attempt)
    return {"error": last_error, "attempts": MAX_ATTEMPTS}


def _extract(model: Model, data: dict, latency_ms: float) -> dict:
    if model.api == "decisions":
        usage = data.get("usage") or {}
        return {
            "raw_text": json.dumps(data["answers"], separators=(",", ":")),
            "routed_model": data.get("model"),
            "provider": data.get("provider"),
            "prompt_toks": usage.get("input_tokens") or 0,
            "completion_toks": usage.get("output_tokens") or 0,
            "reasoning_toks": None,
            "cost": usage.get("cost"),
            "latency_ms": round(latency_ms),
            "error": None,
        }
    msg = data["choices"][0]["message"]
    usage = data.get("usage") or {}
    prompt_toks = usage.get("prompt_tokens") or 0
    completion_toks = usage.get("completion_tokens") or 0
    details = usage.get("completion_tokens_details") or {}
    cost = usage.get("cost")
    if cost is None and model.price_in is not None:
        cost = prompt_toks * model.price_in + completion_toks * model.price_out
    return {
        "raw_text": msg.get("content") or "",
        "routed_model": data.get("model"),
        "provider": data.get("provider"),
        "prompt_toks": prompt_toks,
        "completion_toks": completion_toks,
        "reasoning_toks": details.get("reasoning_tokens"),
        "cost": cost,
        "latency_ms": round(latency_ms),
        "error": None,
    }
