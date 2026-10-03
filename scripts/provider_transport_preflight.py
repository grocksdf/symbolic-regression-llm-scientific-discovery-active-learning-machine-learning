"""One response-free provider request, run by the user before a new freeze.

This uses no benchmark prompts, covariates, responses, or target metadata.
The tool never prints the token or a raw provider error body.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def classify_transport(status: int, error_code: str) -> str:
    code = error_code.lower()
    if 200 <= status < 300:
        return "transport_2xx"
    if status == 429:
        if any(value in code for value in ("quota", "balance", "credit")):
            return "account_quota_or_balance"
        if any(value in code for value in ("rate", "limit", "concurr")):
            return "rate_or_concurrency_limit"
        return "429_cause_unresolved"
    return "http_failure"


def _key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            key = line.split("=", 1)[1].strip()
            if key:
                return key
    raise ValueError("registered provider env has no OPENAI_API_KEY")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--thinking-type", choices=("enabled", "disabled", ""), default="")
    parser.add_argument(
        "--reasoning-effort",
        choices=("max", "high", "medium", "low", "minimal", "none", ""),
        default="")
    parser.add_argument("--provider-env", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.base_url.startswith("https://"):
        raise ValueError("HTTPS provider required")
    if args.output.exists():
        raise ValueError("preflight output must be a new file")
    endpoint = args.base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    payload = {"model": args.model, "messages": [
        {"role": "user", "content": "Return the JSON object {\"ok\":true}."}],
        "max_tokens": 64, "temperature": 0,
        "response_format": {"type": "json_object"}}
    if args.thinking_type:
        payload["thinking"] = {"type": args.thinking_type}
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = Request(endpoint, data=data, method="POST", headers={
        "Authorization": "Bearer " + _key(args.provider_env),
        "Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urlopen(request, timeout=30) as response:
            status, headers, body = response.status, response.headers, response.read(2048)
    except HTTPError as error:
        status, headers, body = error.code, error.headers, error.read(2048)
    except URLError as error:
        status, headers, body = 0, {}, b""
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        parsed = {}
    error = parsed.get("error") if isinstance(parsed, dict) else None
    error_code = str((error.get("code") or error.get("type") or "")
                     if isinstance(error, dict) else "")[:80]
    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    content = (
        str((choices[0].get("message") or {}).get("content") or "")
        if isinstance(choices, list) and choices
        and isinstance(choices[0], dict) else "")
    try:
        completion = json.loads(content)
    except (ValueError, TypeError):
        completion = None
    content_type = str(headers.get("Content-Type", ""))[:120]
    contract_valid = (
        200 <= status < 300
        and "json" in content_type.lower()
        and isinstance(choices, list) and len(choices) == 1
        and isinstance(completion, dict)
        and completion == {"ok": True})
    result = {"schema": "response-free-provider-transport-preflight-v2",
              "http_status": status,
              "error_code": error_code,
              "classification": classify_transport(status, error_code),
              "content_type": content_type,
              "openai_completion_contract_valid": contract_valid,
              "choice_count": len(choices) if isinstance(choices, list) else 0,
              "strict_json_content_valid": completion == {"ok": True},
              "thinking_type": args.thinking_type,
              "reasoning_effort": args.reasoning_effort,
              "retry_after": str(headers.get("Retry-After", ""))[:80],
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "request_utf8_bytes": len(data),
              "scientific_data_accessed": False,
              "model_content_assessed": False}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if contract_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
