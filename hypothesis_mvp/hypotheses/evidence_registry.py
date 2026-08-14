from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterator, Mapping
import uuid


EVIDENCE_EVENT_SCHEMA_VERSION = "hypothesis-evidence-event-v1"
GENESIS_HASH = "0" * 64
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvidenceEventType(str, Enum):
    PROPOSED = "proposed"
    TEST_PLANNED = "test_planned"
    TEST_OBSERVED = "test_observed"
    EVIDENCE_ATTACHED = "evidence_attached"
    POSTERIOR_UPDATED = "posterior_updated"
    STATUS_CHANGED = "status_changed"
    REFINED = "refined"
    MERGED = "merged"


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("evidence records must contain JSON-safe finite values") from error


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json(dict(value)))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class EvidenceEvent:
    event_id: str
    hypothesis_id: str
    event_type: EvidenceEventType
    timestamp_utc: str
    payload: Mapping[str, Any]
    evidence_sha256: str | None
    previous_event_hash: str
    event_hash: str
    schema_version: str = EVIDENCE_EVENT_SCHEMA_VERSION

    def hash_material(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "hypothesis_id": self.hypothesis_id,
            "event_type": self.event_type.value,
            "timestamp_utc": self.timestamp_utc,
            "payload": _thaw_json(self.payload),
            "evidence_sha256": self.evidence_sha256,
            "previous_event_hash": self.previous_event_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.hash_material(), "event_hash": self.event_hash}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvidenceEvent":
        required = {
            "schema_version",
            "event_id",
            "hypothesis_id",
            "event_type",
            "timestamp_utc",
            "payload",
            "evidence_sha256",
            "previous_event_hash",
            "event_hash",
        }
        if set(raw) != required:
            raise ValueError(
                f"event fields mismatch; missing={sorted(required - set(raw))}, "
                f"unknown={sorted(set(raw) - required)}"
            )
        payload = raw["payload"]
        if not isinstance(payload, Mapping):
            raise TypeError("event payload must be an object")
        event = cls(
            schema_version=str(raw["schema_version"]),
            event_id=str(raw["event_id"]),
            hypothesis_id=str(raw["hypothesis_id"]),
            event_type=EvidenceEventType(str(raw["event_type"])),
            timestamp_utc=str(raw["timestamp_utc"]),
            payload=_freeze_json(_json_copy(payload)),
            evidence_sha256=(
                None
                if raw["evidence_sha256"] is None
                else str(raw["evidence_sha256"])
            ),
            previous_event_hash=str(raw["previous_event_hash"]),
            event_hash=str(raw["event_hash"]),
        )
        event.validate_fields()
        return event

    def validate_fields(self) -> None:
        if self.schema_version != EVIDENCE_EVENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence event schema: {self.schema_version}")
        try:
            uuid.UUID(self.event_id)
        except ValueError as error:
            raise ValueError("event_id must be a UUID") from error
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id must be non-empty")
        try:
            timestamp = datetime.fromisoformat(
                self.timestamp_utc.replace("Z", "+00:00")
            )
        except ValueError as error:
            raise ValueError("timestamp_utc must be ISO-8601") from error
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp_utc must include a timezone")
        if self.evidence_sha256 is not None and not _SHA256.fullmatch(
            self.evidence_sha256
        ):
            raise ValueError("evidence_sha256 must be a lowercase SHA-256 digest")
        if not _SHA256.fullmatch(self.previous_event_hash):
            raise ValueError("previous_event_hash must be a lowercase SHA-256 digest")
        if not _SHA256.fullmatch(self.event_hash):
            raise ValueError("event_hash must be a lowercase SHA-256 digest")
        _canonical_json(_thaw_json(self.payload))

    @property
    def computed_hash(self) -> str:
        return sha256(_canonical_json(self.hash_material()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RegistryVerification:
    valid: bool
    event_count: int
    head_hash: str
    errors: tuple[str, ...]


class EvidenceRegistry:
    """Store immutable events in a JSONL hash chain.

    Appends are serialized with a sibling lock file on Windows and POSIX.
    Historical lines are never rewritten; tampering is detected by ``verify``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.suffix.lower() != ".jsonl":
            raise ValueError("evidence registry path must end in .jsonl")
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @staticmethod
    def hash_evidence_file(path: str | Path) -> str:
        digest = sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def append(
        self,
        *,
        hypothesis_id: str,
        event_type: EvidenceEventType | str,
        payload: Mapping[str, Any],
        evidence_sha256: str | None = None,
    ) -> EvidenceEvent:
        if not hypothesis_id:
            raise ValueError("hypothesis_id must be non-empty")
        kind = (
            event_type
            if isinstance(event_type, EvidenceEventType)
            else EvidenceEventType(str(event_type))
        )
        payload_copy = _json_copy(payload)
        if evidence_sha256 is not None and not _SHA256.fullmatch(evidence_sha256):
            raise ValueError("evidence_sha256 must be a lowercase SHA-256 digest")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._exclusive_lock():
            existing = self._read_events_unlocked()
            verification = self._verify_events(existing)
            if not verification.valid:
                raise RuntimeError(
                    "refusing to append to an invalid evidence chain: "
                    + "; ".join(verification.errors)
                )
            material = {
                "schema_version": EVIDENCE_EVENT_SCHEMA_VERSION,
                "event_id": str(uuid.uuid4()),
                "hypothesis_id": hypothesis_id,
                "event_type": kind.value,
                "timestamp_utc": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat(),
                "payload": payload_copy,
                "evidence_sha256": evidence_sha256,
                "previous_event_hash": verification.head_hash,
            }
            event = EvidenceEvent(
                schema_version=EVIDENCE_EVENT_SCHEMA_VERSION,
                event_id=str(material["event_id"]),
                hypothesis_id=hypothesis_id,
                event_type=kind,
                timestamp_utc=str(material["timestamp_utc"]),
                payload=_freeze_json(payload_copy),
                evidence_sha256=evidence_sha256,
                previous_event_hash=str(material["previous_event_hash"]),
                event_hash=sha256(
                    _canonical_json(material).encode("utf-8")
                ).hexdigest(),
            )
            event.validate_fields()
            line = (_canonical_json(event.to_dict()) + "\n").encode("utf-8")
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                written = os.write(descriptor, line)
                if written != len(line):
                    raise OSError("short append while writing evidence event")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return event

    def verify(self) -> RegistryVerification:
        with self._exclusive_lock():
            try:
                events = self._read_events_unlocked()
            except Exception as error:
                return RegistryVerification(
                    valid=False,
                    event_count=0,
                    head_hash=GENESIS_HASH,
                    errors=(f"could not parse registry: {error}",),
                )
            return self._verify_events(events)

    def events(self, *, hypothesis_id: str | None = None) -> tuple[EvidenceEvent, ...]:
        with self._exclusive_lock():
            events = self._read_events_unlocked()
            verification = self._verify_events(events)
            if not verification.valid:
                raise RuntimeError("invalid evidence chain: " + "; ".join(verification.errors))
            if hypothesis_id is None:
                return tuple(events)
            return tuple(event for event in events if event.hypothesis_id == hypothesis_id)

    def _read_events_unlocked(self) -> list[EvidenceEvent]:
        if not self.path.exists():
            return []
        events: list[EvidenceEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank line at registry line {line_number}")
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise TypeError(f"registry line {line_number} is not an object")
                events.append(EvidenceEvent.from_dict(raw))
        return events

    @staticmethod
    def _verify_events(events: list[EvidenceEvent]) -> RegistryVerification:
        previous = GENESIS_HASH
        errors: list[str] = []
        seen_ids: set[str] = set()
        for index, event in enumerate(events, start=1):
            if event.event_id in seen_ids:
                errors.append(f"line {index}: duplicate event_id")
            seen_ids.add(event.event_id)
            if event.previous_event_hash != previous:
                errors.append(f"line {index}: broken previous_event_hash")
            if event.computed_hash != event.event_hash:
                errors.append(f"line {index}: event_hash mismatch")
            previous = event.event_hash
        return RegistryVerification(
            valid=not errors,
            event_count=len(events),
            head_hash=previous,
            errors=tuple(errors),
        )

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

