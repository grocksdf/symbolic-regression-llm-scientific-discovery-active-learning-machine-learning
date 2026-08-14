"""Single audited OpenAI-compatible transport for every LLM proposal."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping, Sequence

import requests

from .contracts import DISCOVERY_RUNTIME_ID, json_safe
from .equation_runtime import EquationRuntime, sha256_text

PROPOSAL_PROTOCOL_ID = "hypothesis-proposal-v1"
ALLOWED_ACTIONS = frozenset({
    "ADD", "DELETE", "REPLACE", "REPARAMETERIZE",
    "CHANGE_OPERATOR", "CHANGE_INTERACTION",
})


class ProtocolError(ValueError):
    pass


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ProtocolError(f"duplicate_json_key:{key}")
        value[key] = child
    return value


def strict_json_loads(text: str) -> Any:
    raw = str(text or "")
    if not raw.strip() or raw.lstrip().startswith("```"):
        raise ProtocolError("provider content must be unfenced JSON")
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    value, end = decoder.raw_decode(raw.lstrip())
    consumed = len(raw) - len(raw.lstrip()) + end
    if raw[consumed:].strip():
        raise ProtocolError("trailing_non_json_content")
    return value


def _environment(name: str, aliases: Sequence[str] = ()) -> str:
    return next((os.environ[key] for key in (name, *aliases) if key in os.environ), "")


@dataclass(frozen=True)
class ProviderRoute:
    base_url: str
    model: str
    api_key: str
    provider: str = "openai_compatible"
    api_path: str = "/chat/completions"
    api_method: str = "POST"
    role: str = "primary"

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        path = "/" + self.api_path.strip("/")
        return base if base.endswith(path) else base + path

    @property
    def key(self) -> tuple[str, str]:
        return self.endpoint, self.model

    def validate(self) -> None:
        if not self.base_url.startswith("https://"):
            raise ValueError("LLM base URL must use HTTPS")
        if not self.model or not self.api_key:
            raise ValueError("LLM model/endpoint ID and API key are required")
        if self.api_method.upper() != "POST":
            raise ValueError("the audited chat transport supports POST only")


@dataclass(frozen=True)
class ProviderSettings:
    routes: tuple[ProviderRoute, ...] = ()
    attempts: int = 3
    connect_timeout_s: float = 15.0
    read_timeout_s: float = 150.0
    retry_backoff_s: float = 4.0
    rate_limit_backoff_s: float = 30.0
    min_request_interval_s: float = 2.0
    temperature: float = 0.25
    max_tokens: int = 4096
    thinking_type: str = ""
    reasoning_effort: str = ""
    do_sample: bool | None = None

    @classmethod
    def from_environment(cls, **overrides: Any) -> "ProviderSettings":
        route = ProviderRoute(
            base_url=str(overrides.get("base_url") or _environment(
                "HYPOTHESIS_LLM_API_BASE", ("API_BASE_URL", "OPENAI_BASE_URL")
            )).strip(),
            model=str(overrides.get("model") or _environment(
                "HYPOTHESIS_LLM_MODEL", ("MODEL",)
            )).strip(),
            api_key=str(overrides.get("api_key") or _environment(
                "HYPOTHESIS_LLM_API_KEY", ("API_KEY", "OPENAI_API_KEY")
            )).strip(),
            provider=_environment("HYPOTHESIS_LLM_PROVIDER") or "openai_compatible",
            api_path=_environment("HYPOTHESIS_LLM_API_PATH", ("API_PATH",)) or "/chat/completions",
            api_method=(_environment("HYPOTHESIS_LLM_API_METHOD", ("API_METHOD",)) or "POST").upper(),
        )
        if not route.base_url and not route.model and not route.api_key:
            return cls()
        route.validate()
        fields = {
            name: overrides[name]
            for name in cls.__dataclass_fields__
            if name != "routes" and name in overrides
        }
        return cls(routes=(route,), **fields)

    @classmethod
    def from_file(cls, path: str | Path) -> "ProviderSettings":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"api_base_url", "api_path", "api_method", "model", "api_key"}
        if not isinstance(values, dict) or required - set(values):
            raise ValueError("LLM config is missing required provider fields")
        route = ProviderRoute(
            base_url=str(values["api_base_url"]).strip(),
            model=str(values["model"]).strip(),
            api_key=str(values["api_key"]).strip(),
            provider=str(values.get("provider") or "openai_compatible"),
            api_path=str(values["api_path"]).strip(),
            api_method=str(values["api_method"]).upper().strip(),
        )
        route.validate()
        thinking_type = str(values.get("thinking_type") or "").strip()
        reasoning_effort = str(values.get("reasoning_effort") or "").strip()
        if thinking_type not in {"", "enabled", "disabled"}:
            raise ValueError("thinking_type must be enabled, disabled or empty")
        if reasoning_effort not in {"", "max", "xhigh", "high", "medium", "low", "minimal", "none"}:
            raise ValueError("unsupported reasoning_effort")
        return cls(
            routes=(route,), attempts=max(1, int(values.get("attempts", 3))),
            connect_timeout_s=max(1.0, float(values.get("connect_timeout_s", 15.0))),
            read_timeout_s=max(5.0, float(values.get("read_timeout_s", 150.0))),
            retry_backoff_s=max(0.0, float(values.get("retry_backoff_s", 4.0))),
            rate_limit_backoff_s=max(0.0, float(values.get("rate_limit_backoff_s", 30.0))),
            min_request_interval_s=max(0.0, float(values.get("min_request_interval_s", 2.0))),
            temperature=float(values.get("temperature", 0.25)),
            max_tokens=max(512, int(values.get("max_tokens", 4096))),
            thinking_type=thinking_type,
            reasoning_effort=reasoning_effort,
            do_sample=(
                bool(values["do_sample"]) if "do_sample" in values else None
            ),
        )


@dataclass(frozen=True)
class ProposalContext:
    round_id: int
    island: str
    parent_hash: str
    n_features: int
    max_candidates: int


@dataclass(frozen=True)
class ProposalCandidate:
    candidate_id: str
    island: str
    equation: str
    action: str
    rationale: str
    parent_hash: str
    expected_effect: Mapping[str, Any] = field(default_factory=dict)
    library_refs: tuple[str, ...] = ()
    lineage_id: str = ""
    prompt_hash: str = ""
    response_hash: str = ""
    proposal_index: int = 0


@dataclass(frozen=True)
class ProposalBatch:
    candidates: tuple[ProposalCandidate, ...]
    protocol_valid: bool
    reason: str
    prompt_hash: str = ""
    response_hash: str = ""
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class _ValidatedResponse:
    candidates: tuple[ProposalCandidate, ...]
    rejections: tuple[Mapping[str, Any], ...]
    normalizations: tuple[Mapping[str, Any], ...]
    prompt_hash: str
    response_hash: str
    telemetry: Mapping[str, Any]


class ProposalRuntime:
    """Strict proposal protocol with explicit provider failure semantics."""

    def __init__(
        self, equation_runtime: EquationRuntime, n_features: int,
        settings: ProviderSettings | None, candidates_per_island: int,
    ) -> None:
        self.equation_runtime = equation_runtime
        self.registry = equation_runtime.registry
        self.variable_metadata = dict(equation_runtime.variable_metadata)
        self.n_features = int(n_features)
        self.settings = settings
        self.candidates_per_island = max(1, int(candidates_per_island))
        self.call_count = 0
        self.attempt_count = 0
        self._errors: list[str] = []
        self._telemetry: list[dict[str, Any]] = []
        self._disabled: set[tuple[str, str]] = set()
        self._last_request = 0.0
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return bool(self.settings and self.settings.routes)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(self._errors)

    @property
    def telemetry(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._telemetry]

    @property
    def disabled_routes(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._disabled))

    def reset(self) -> None:
        with self._lock:
            self.call_count = 0
            self.attempt_count = 0
            self._errors.clear()
            self._telemetry.clear()
            self._disabled.clear()
            self._last_request = 0.0

    def _messages(self, payload: Mapping[str, Any], system: str) -> tuple[list[dict[str, str]], str]:
        content = json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(content.encode()).hexdigest()
        return [{"role": "system", "content": system}, {"role": "user", "content": content}], digest

    def _wait_for_rate_limit(self) -> None:
        assert self.settings is not None
        with self._lock:
            wait = self.settings.min_request_interval_s - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def _post(self, route: ProviderRoute, messages: Sequence[Mapping[str, str]]) -> tuple[str, dict[str, Any]]:
        assert self.settings is not None
        route.validate()
        self._wait_for_rate_limit()
        started = time.monotonic()
        request_body: dict[str, Any] = {
            "model": route.model, "messages": list(messages),
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens, "stream": False,
            "response_format": {"type": "json_object"},
        }
        if self.settings.thinking_type:
            request_body["thinking"] = {"type": self.settings.thinking_type}
        if self.settings.reasoning_effort:
            request_body["reasoning_effort"] = self.settings.reasoning_effort
        if self.settings.do_sample is not None:
            request_body["do_sample"] = self.settings.do_sample
        response = requests.post(
            route.endpoint,
            headers={"Authorization": f"Bearer {route.api_key}", "Content-Type": "application/json"},
            json=request_body,
            timeout=(self.settings.connect_timeout_s, self.settings.read_timeout_s),
        )
        telemetry = {
            "actual_provider": route.provider, "actual_model": route.model,
            "provider_role": route.role, "provider_url": route.endpoint,
            "provider_api_method": route.api_method,
            "provider_http_status": response.status_code,
            "provider_elapsed_s": time.monotonic() - started,
        }
        if response.status_code >= 400:
            telemetry["provider_error_body_excerpt"] = str(response.text or "")[:500]
            raise RuntimeError(json.dumps(telemetry, sort_keys=True))
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ProtocolError("provider_content_missing")
        return content, telemetry

    def _request(self, messages: Sequence[Mapping[str, str]], prompt_hash: str) -> tuple[str, dict[str, Any]]:
        if not self.enabled or self.settings is None:
            raise RuntimeError("ProposalRuntime has no configured provider route")
        route = self.settings.routes[0]
        if route.key in self._disabled:
            raise RuntimeError("LLM provider route was disabled after an authentication failure")
        outcomes: list[dict[str, Any]] = []
        for attempt in range(1, self.settings.attempts + 1):
            self.attempt_count += 1
            try:
                content, telemetry = self._post(route, messages)
                outcomes.append({**telemetry, "provider_attempt_index": attempt})
                aggregate = {
                    "prompt_hash": prompt_hash,
                    "provider_retry_attempt_count": len(outcomes),
                    "provider_outcomes": outcomes,
                    "provider_final_outcome": dict(outcomes[-1]),
                    "provider_all_attempts_preserved": True,
                }
                self._telemetry.append(aggregate)
                return content, aggregate
            except Exception as error:
                outcome = {"provider_attempt_index": attempt, "provider_error": repr(error)}
                outcomes.append(outcome)
                if "401" in str(error) or "402" in str(error) or "403" in str(error):
                    self._disabled.add(route.key)
                    break
                if attempt < self.settings.attempts:
                    delay = (
                        self.settings.rate_limit_backoff_s
                        if "429" in str(error)
                        else self.settings.retry_backoff_s * (2 ** (attempt - 1))
                    )
                    time.sleep(delay)
        self._errors.append(outcomes[-1]["provider_error"])
        self._telemetry.append({
            "prompt_hash": prompt_hash,
            "provider_retry_attempt_count": len(outcomes),
            "provider_outcomes": outcomes,
            "provider_final_outcome": dict(outcomes[-1]),
            "provider_all_attempts_preserved": True,
        })
        raise RuntimeError(f"LLM provider exhausted: {outcomes[-1]['provider_error']}")

    def complete_json(
        self, *, system_message: str, payload: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        self.call_count += 1
        messages, prompt_hash = self._messages(payload, system_message)
        content, telemetry = self._request(messages, prompt_hash)
        parsed = strict_json_loads(content)
        if not isinstance(parsed, dict):
            raise ProtocolError("root_must_be_object")
        return parsed, telemetry

    def _proposal_payload(
        self, task_name: str, task_desc: str, context: ProposalContext,
        island_context: Mapping[str, Any], library_rows: Sequence[Mapping[str, Any]],
        refinements: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "runtime_id": DISCOVERY_RUNTIME_ID,
            "protocol_id": PROPOSAL_PROTOCOL_ID,
            "task": {"name": task_name, "description": task_desc, "n_features": self.n_features},
            "round_id": context.round_id, "island": context.island,
            "parent_hash": context.parent_hash,
            "scientific_context": dict(island_context),
            "verified_structure_library": [dict(row) for row in library_rows],
            "accepted_refinements": [dict(row) for row in refinements],
            "contract": {
                "max_candidates": context.max_candidates,
                "equation_format": "right_hand_side_only_without_assignment",
                "allowed_symbols": [f"x{i}" for i in range(context.n_features)],
                "forbidden_symbols": ["y", "y_hat"],
                "required_candidate_fields": [
                    "candidate_id", "parent_hash", "action", "equation", "rationale"
                ],
                "instruction": (
                    "Each equation is only an executable right-hand-side expression. "
                    "Never emit y=, f(x)=, y_hat, or any symbol outside allowed_symbols. "
                    "Diagnostic y_hat text describes the current predictor, not an output variable."
                ),
            },
        }

    def _equation(
        self, raw: Any, context: ProposalContext,
    ) -> tuple[str, dict[str, Any]]:
        text = str(raw or "").strip()
        audit: dict[str, Any] = {"assignment_removed": False}
        if "=" in text:
            if text.count("=") != 1:
                raise ProtocolError("equation_has_multiple_assignments")
            left, text = (part.strip() for part in text.split("=", 1))
            if left.lower() != "y" or not text:
                raise ProtocolError("equation_assignment_lhs_must_be_y")
            audit = {"assignment_removed": True, "original_lhs": left}
        equation = self.registry.normalize(text)
        forbidden = sorted(set(re.findall(r"\b(?:y_hat|y)\b", equation)))
        if forbidden:
            raise ProtocolError("forbidden_output_symbol:" + ",".join(forbidden))
        self.registry.parse(equation, context.n_features, evaluate=False)
        return equation, audit

    def _candidate(
        self, item: Mapping[str, Any], index: int, context: ProposalContext,
        prompt_hash: str, response_hash: str,
    ) -> tuple[ProposalCandidate, dict[str, Any]]:
        identifier = str(item.get("candidate_id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", identifier):
            raise ProtocolError("invalid_candidate_id")
        if str(item.get("parent_hash")) != context.parent_hash:
            raise ProtocolError("parent_hash_mismatch")
        action = str(item.get("action") or "").upper()
        if action not in ALLOWED_ACTIONS:
            raise ProtocolError("invalid_action")
        equation, normalization = self._equation(item.get("equation"), context)
        rationale = str(item.get("rationale") or "").strip()
        if not rationale or len(rationale) > 1200:
            raise ProtocolError("invalid_rationale")
        lineage = sha256_text(
            f"{prompt_hash}|{response_hash}|{identifier}|{context.parent_hash}|{equation}", 32
        )
        candidate = ProposalCandidate(
            identifier, context.island, equation, action, rationale, context.parent_hash,
            dict(item.get("expected_effect") or {}),
            tuple(str(value) for value in item.get("library_refs") or ()),
            lineage, prompt_hash, response_hash, index,
        )
        return candidate, {
            "candidate_id": identifier, "proposal_index": index, **normalization,
        }

    def _validate_batch(
        self, parsed: Mapping[str, Any], context: ProposalContext,
        prompt_hash: str, response_hash: str,
    ) -> tuple[
        tuple[ProposalCandidate, ...],
        tuple[Mapping[str, Any], ...],
        tuple[Mapping[str, Any], ...],
    ]:
        if parsed.get("protocol_id") != PROPOSAL_PROTOCOL_ID:
            raise ProtocolError("protocol_id_mismatch")
        if int(parsed.get("round_id", -1)) != context.round_id or parsed.get("island") != context.island:
            raise ProtocolError("round_or_island_mismatch")
        items = parsed.get("candidates")
        if not isinstance(items, list) or not 1 <= len(items) <= context.max_candidates:
            raise ProtocolError("candidate_count_out_of_range")
        candidates: list[ProposalCandidate] = []
        rejections: list[Mapping[str, Any]] = []
        normalizations: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(items):
            identifier = str(item.get("candidate_id") or "") if isinstance(item, Mapping) else ""
            try:
                if not isinstance(item, Mapping):
                    raise ProtocolError("candidate_must_be_object")
                candidate, normalization = self._candidate(
                    item, index, context, prompt_hash, response_hash
                )
                if candidate.candidate_id in seen:
                    raise ProtocolError("duplicate_candidate_id")
                seen.add(candidate.candidate_id)
                candidates.append(candidate)
                if normalization.get("assignment_removed"):
                    normalizations.append(normalization)
            except Exception as error:
                rejections.append({
                    "proposal_index": index,
                    "candidate_id": identifier,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
        return tuple(candidates), tuple(rejections), tuple(normalizations)

    def _system_message(self) -> str:
        allowed = ", ".join(f"x{i}" for i in range(self.n_features))
        return (
            "You propose falsifiable structural equations as exactly one unfenced JSON object. "
            "Use protocol_id='hypothesis-proposal-v1', runtime_id='canonical-real-only-discovery', "
            "and preserve the requested round_id, island, and parent_hash. Every candidate needs "
            "candidate_id, parent_hash, action, equation, and rationale. The equation field is RHS only: "
            f"use only {allowed}; never use '=', y, y_hat, or prose. Diagnostic y_hat is not an allowed "
            "variable: express every final candidate completely in x variables."
        )

    def _request_validated(
        self, payload: Mapping[str, Any], context: ProposalContext,
    ) -> _ValidatedResponse:
        messages, prompt_hash = self._messages(payload, self._system_message())
        self.call_count += 1
        content, telemetry = self._request(messages, prompt_hash)
        response_hash = hashlib.sha256(content.encode()).hexdigest()
        parsed = strict_json_loads(content)
        if not isinstance(parsed, Mapping):
            raise ProtocolError("root_must_be_object")
        candidates, rejections, normalizations = self._validate_batch(
            parsed, context, prompt_hash, response_hash
        )
        return _ValidatedResponse(
            candidates, rejections, normalizations,
            prompt_hash, response_hash, telemetry,
        )

    @staticmethod
    def _response_telemetry(
        responses: Sequence[_ValidatedResponse],
        library_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        library_text = json.dumps(json_safe(list(library_rows)), sort_keys=True)
        return {
            "provider_requests": [dict(response.telemetry) for response in responses],
            "protocol_repair_attempted": len(responses) > 1,
            "candidate_validation_rejections": [
                dict(row) for response in responses for row in response.rejections
            ],
            "candidate_normalizations": [
                dict(row) for response in responses for row in response.normalizations
            ],
            "retrieved_structure_entry_ids": [
                str(row.get("entry_id")) for row in library_rows if row.get("entry_id")
            ],
            "retrieved_structure_count": len(library_rows),
            "structure_prompt_context_hash": hashlib.sha256(library_text.encode()).hexdigest(),
        }

    def propose(
        self, *, task_name: str, task_desc: str, round_id: int, island: str,
        parent_hash: str, island_context: Mapping[str, Any],
        library_rows: Sequence[Mapping[str, Any]],
        ephemeral_refinements: Sequence[Mapping[str, Any]],
    ) -> ProposalBatch:
        context = ProposalContext(
            round_id, island, parent_hash, self.n_features, self.candidates_per_island
        )
        payload = self._proposal_payload(
            task_name, task_desc, context, island_context,
            library_rows, ephemeral_refinements,
        )
        error_count_before = len(self._errors)
        try:
            first = self._request_validated(payload, context)
            responses = [first]
            selected = first
            if not first.candidates:
                repair_payload = {
                    **payload,
                    "protocol_repair": {
                        "previous_candidate_rejections": [dict(row) for row in first.rejections],
                        "instruction": "Regenerate the complete candidates array under the unchanged contract.",
                    },
                }
                selected = self._request_validated(repair_payload, context)
                responses.append(selected)
            telemetry = self._response_telemetry(responses, library_rows)
            protocol_valid = bool(selected.candidates and not selected.rejections)
            reason = (
                "ok_after_protocol_repair" if len(responses) > 1 and protocol_valid
                else "ok" if protocol_valid
                else "partial_candidates_rejected" if selected.candidates
                else "all_candidates_rejected"
            )
            error = "" if selected.candidates else json.dumps(
                [dict(row) for row in selected.rejections], ensure_ascii=False
            )
            if error:
                self._errors.append(error)
            return ProposalBatch(
                selected.candidates, protocol_valid, reason,
                selected.prompt_hash, selected.response_hash, telemetry, error,
            )
        except Exception as error:
            rendered = f"configured LLM proposal failed explicitly: {error}"
            if len(self._errors) == error_count_before:
                self._errors.append(rendered)
            return ProposalBatch((), False, "provider_or_protocol_failure", error=rendered)


__all__ = [
    "PROPOSAL_PROTOCOL_ID", "ProposalBatch", "ProposalCandidate", "ProposalRuntime",
    "ProviderRoute", "ProviderSettings", "ProtocolError", "strict_json_loads",
]
