"""Provider smoke schema tests without network access."""

import json

from scripts.run_aistats_drr_provider_live_smoke import run_smoke


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "id": "fixture", "model": "glm-5.3",
            "choices": [{"message": {"content": "{\"status\":\"ok\"}"}}]}


def test_provider_smoke_never_publishes_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=secret-fixture\n", encoding="utf-8")
    captured = {}

    def post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return _Response()

    result = run_smoke(env, request_post=post)
    assert result["passed"] is True
    assert result["api_key_published"] is False
    assert result["benchmark_task_arrays_accessed"] is False
    assert "secret-fixture" not in json.dumps(result)
    assert captured["headers"]["Authorization"] == "Bearer secret-fixture"
    assert "thinking" not in captured["json"]
    assert captured["json"]["reasoning_effort"] == "low"
    assert captured["json"]["response_format"] == {"type": "json_object"}
