"""No-data import and provider compatibility Gate for the LLM-SR baseline."""

from __future__ import annotations

import argparse
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "methods"))


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provider_key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            value = line.split("=", 1)[1].strip()
            if value:
                return value
    raise ValueError("provider env has no OPENAI_API_KEY")


async def _smoke(api_key: str) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=api_key,
        timeout=60.0,
    )
    response = await client.chat.completions.create(
        model="glm-5.3",
        messages=[{
            "role": "user",
            "content": "Return exactly the token: OK",
        }],
        temperature=0.0,
        # glm-5.3 is a reasoning model: every token below its thinking budget
        # is consumed by ``reasoning_content`` and ``message.content`` comes
        # back empty, which would make this smoke test fail closed on a
        # perfectly healthy provider. 256 clears the thinking budget.
        max_tokens=256,
    )
    content = str(response.choices[0].message.content or "").strip()
    return {
        "actual_model": str(response.model or ""),
        "response_nonempty": bool(content),
        "response_sha256": sha256(content.encode()).hexdigest(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-env", type=Path, required=True)
    parser.add_argument("--external-site", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--live-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("external LLM-SR resource Gate output must be new")
    if not args.external_site.is_dir():
        raise ValueError("external LLM-SR site directory is unavailable")
    sys.path.insert(0, str(args.external_site.resolve()))

    import openai
    from methods.llmsr import sampler

    config = ROOT / "configs/aistats_external_llmsr_glm53.yaml"
    decisions = {
        "openai_client_imported": bool(openai.__version__),
        "generic_openai_compatible_endpoint_supported":
            "else:" in Path(sampler.__file__).read_text(encoding="utf-8"),
        "provider_attempts_are_bounded":
            "LLMSR_PROVIDER_MAX_ATTEMPTS" in
            Path(sampler.__file__).read_text(encoding="utf-8"),
        "config_requests_glm_5_3":
            "api_model: glm-5.3" in config.read_text(encoding="utf-8"),
        "benchmark_task_arrays_stayed_closed": True,
    }
    live = {}
    if args.live_smoke:
        os.environ["LLMSR_PROVIDER_MAX_ATTEMPTS"] = "1"
        live = asyncio.run(_smoke(_provider_key(args.provider_env)))
        decisions["live_provider_response_received"] = live["response_nonempty"]
    result = {
        "schema": "scientific-aistats-external-llmsr-resource-gate-v1",
        "passed": all(decisions.values()),
        "decisions": decisions,
        "python_executable": sys.executable,
        "openai_version": openai.__version__,
        "external_site": str(args.external_site.resolve()),
        "config_sha256": _sha(config),
        "sampler_sha256": _sha(Path(sampler.__file__)),
        "live_smoke": live,
        "scientific_prompt_sent": False,
        "benchmark_task_arrays_accessed": False,
        "candidate_response_accessed": False,
        "test_or_ood_accessed": False,
        "heldout_opened": False,
        "execution_authorized": False,
        "claim_boundary": (
            "External LLM-SR dependency and provider compatibility only; "
            "no benchmark task, scientific hypothesis, efficacy, or "
            "superiority claim."
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "EXTERNAL_LLMSR_RESOURCE_GATE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
