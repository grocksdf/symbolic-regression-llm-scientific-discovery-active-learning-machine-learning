"""Canonical real-only scientific-discovery API."""

from .agent import DiscoveryAgent, DiscoveryAgentConfig, DiscoveryAgentResult
from .api import DiscoveryRunResult, discover_from_selection
from .confirmation import ConfirmationResult, confirm_frozen_hypothesis
from .contracts import DiscoveryConfig, DiscoveryPhase, DiscoveryState, RuntimeEvent
from .proposal_runtime import ProviderSettings

__all__ = [
    "ConfirmationResult", "DiscoveryAgent", "DiscoveryAgentConfig",
    "DiscoveryAgentResult", "DiscoveryConfig", "DiscoveryPhase",
    "DiscoveryRunResult", "DiscoveryState",
    "ProviderSettings", "RuntimeEvent", "confirm_frozen_hypothesis",
    "discover_from_selection",
]
