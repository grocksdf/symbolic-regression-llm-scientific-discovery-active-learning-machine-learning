"""One-call GLM-5.3 provider smoke with no benchmark or scientific content."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import requests


MODEL = "glm-5.3"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def _load_env(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def run_smoke(env_path, request_post=requests.post):
    values = _load_env(env_path)
    key = str(values.get("OPENAI_API_KEY") or os.environ.get(
        "OPENAI_API_KEY") or "")
    if not key:
        raise ValueError("OPENAI_API_KEY is unavailable")
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": "Return exactly this JSON object: {\"status\":\"ok\"}"}],
        "temperature": 0.0,
        "max_tokens": 1024,
        "reasoning_effort": "low",
        "do_sample": False,
        "response_format": {"type": "json_object"},
    }
    response = request_post(
        BASE_URL + "/chat/completions",
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"},
        json=payload, timeout=60)
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("provider response has no choices")
    message = choices[0].get("message", {})
    content = str(message.get("content") or "").strip()
    reasoning = str(message.get("reasoning_content") or "").strip()
    if not content:
        raise ValueError(
            "provider response content is empty after bounded reasoning")
    parsed = json.loads(content)
    if parsed != {"status": "ok"}:
        raise ValueError("provider JSON smoke content changed")
    return {
        "schema": "scientific-aistats-drr-provider-live-smoke-v1",
        "passed": True,
        "provider_public_identity": sha256(
            f"{BASE_URL}|{MODEL}".encode()).hexdigest(),
        "model_requested": MODEL,
        "endpoint": BASE_URL,
        "response_model": body.get("model"),
        "response_id_present": bool(body.get("id")),
        "choice_count": len(choices),
        "finish_reason": choices[0].get("finish_reason"),
        "content_sha256": sha256(content.encode()).hexdigest(),
        "content_length": len(content),
        "reasoning_content_present": bool(reasoning),
        "reasoning_content_length": len(reasoning),
        "reasoning_effort": "low",
        "response_format": "json_object",
        "api_key_present": True,
        "api_key_published": False,
        "benchmark_task_arrays_accessed": False,
        "scientific_prompt_sent": False,
        "test_or_ood_accessed": False,
        "execution_authorized": False,
        "claim_boundary": (
            "One fixed non-scientific provider request only; no benchmark "
            "execution, efficacy, or superiority evidence."),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("provider smoke output must be new")
    result = run_smoke(args.env)
    args.output_dir.mkdir(parents=True)
    target = args.output_dir / "AISTATS_DRR_PROVIDER_LIVE_SMOKE.json"
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
