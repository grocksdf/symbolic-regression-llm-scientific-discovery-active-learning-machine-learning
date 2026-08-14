from pathlib import Path

from hypothesis_mvp.discovery.equation_runtime import EquationRuntime
from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime, ProviderSettings


def test_bigmodel_glm_5_2_route_is_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = ProviderSettings.from_file(root / "config" / "bigmodel_glm_5_2.json")
    route = settings.routes[0]
    assert route.provider == "zhipu_bigmodel"
    assert route.api_method == "POST"
    assert route.endpoint == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert route.model == "glm-5.2"
    assert settings.thinking_type == "enabled"
    assert settings.reasoning_effort == "max"


def test_glm_5_2_reasoning_payload_is_sent(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = ProviderSettings.from_file(root / "config" / "bigmodel_glm_5_2.json")
    captured = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "{}"}}]}

    def fake_post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr("hypothesis_mvp.discovery.proposal_runtime.requests.post", fake_post)
    runtime = ProposalRuntime(EquationRuntime(1), 1, settings, candidates_per_island=1)
    runtime._post(settings.routes[0], [{"role": "user", "content": "{}"}])
    assert captured["json"]["model"] == "glm-5.2"
    assert captured["json"]["thinking"] == {"type": "enabled"}
    assert captured["json"]["reasoning_effort"] == "max"
    assert captured["json"]["response_format"] == {"type": "json_object"}
