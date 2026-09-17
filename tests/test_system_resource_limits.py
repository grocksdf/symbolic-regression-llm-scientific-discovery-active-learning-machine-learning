"""Real process/transport-control correctness; never experimental efficacy."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import multiprocessing as mp
import time

import pytest

from hypothesis_mvp.discovery.resource_limits import (
    ResourceLimitExceeded, before_provider_transport, run_bounded, stage_budget,
)
from hypothesis_mvp.discovery.pcpi_adapter import DiscoveryAdapterError


def _sleep():
    time.sleep(30)


def _swallow_denial():
    before_provider_transport()
    try: before_provider_transport()
    except ResourceLimitExceeded: pass
    return "must not publish success"


def _adapter_failure():
    raise DiscoveryAdapterError("candidate-not-adaptable:fixture-safe-code")


def test_bounded_process_success():
    value, usage = run_bounded(abs, args=(-3,), seconds=15, provider_attempts=0)
    assert value == 3 and usage["provider_transports_used"] == 0


def test_timeout_terminates_and_joins_owned_child():
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(ResourceLimitExceeded):
        run_bounded(_sleep, seconds=.3, provider_attempts=0)
    assert {p.pid for p in mp.active_children()} == before


def test_swallowed_quota_denial_still_blocks_success():
    with pytest.raises(RuntimeError, match="ResourceLimitExceeded"):
        run_bounded(_swallow_denial, seconds=15, provider_attempts=1)


def test_only_explicit_public_adapter_diagnostic_crosses_process_boundary():
    with pytest.raises(RuntimeError, match="candidate-not-adaptable:fixture-safe-code"):
        run_bounded(_adapter_failure, seconds=15, provider_attempts=0)


def test_provider_limit_before_network_shared_across_runtimes(monkeypatch):
    from hypothesis_mvp.discovery.proposal_runtime import ProposalRuntime, ProviderSettings, ProviderRoute
    calls = []
    def post(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(status_code=200,
            json=lambda: {"choices": [{"message": {"content": "{}"}}]})
    monkeypatch.setattr("hypothesis_mvp.discovery.proposal_runtime.requests.post", post)
    route = ProviderRoute("https://fixture.invalid", "fixture-model", "fixture-key")
    settings = ProviderSettings(routes=(route,), min_request_interval_s=0)
    def runtime():
        instance = ProposalRuntime.__new__(ProposalRuntime)
        instance.settings = settings
        instance._wait_for_rate_limit = lambda: None
        return instance
    with pytest.raises(ResourceLimitExceeded):
        with stage_budget(10, 1):
            runtime()._post(route, [])
            runtime()._post(route, [])
    assert len(calls) == 1


def test_threaded_provider_quota_is_atomic():
    def consume(_):
        try: before_provider_transport(); return 1
        except ResourceLimitExceeded: return 0
    with pytest.raises(ResourceLimitExceeded):
        with stage_budget(10, 3):
            with ThreadPoolExecutor(max_workers=4) as executor:
                assert sum(executor.map(consume, range(20))) == 3


@pytest.mark.parametrize("seconds,attempts", [(0, 1), (float("nan"), 1), (1, -1), (1, True)])
def test_invalid_limits_rejected_before_spawn(seconds, attempts):
    with pytest.raises(ValueError): run_bounded(abs, args=(-3,), seconds=seconds, provider_attempts=attempts)
