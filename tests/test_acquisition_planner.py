from pathlib import Path

import pytest

from hypothesis_mvp.discovery.agent import DiscoveryAgent, DiscoveryAgentConfig


def test_legacy_acquisition_module_is_absent() -> None:
    package = Path(__file__).resolve().parents[1] / "hypothesis_mvp" / "discovery"
    assert not (package / "acquisition.py").exists()


def test_discovery_agent_fails_closed_on_legacy_acquisition_request() -> None:
    with pytest.raises(ValueError, match="canonical P3B"):
        DiscoveryAgent(DiscoveryAgentConfig(acquisition_enabled=True))
