from __future__ import annotations

"""The only owner of durable verified knowledge and audit logs."""

import contextlib
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import sympy as sp

from .contracts import EquationState, LineageStep, RuntimeEvent, DISCOVERY_RUNTIME_ID, json_safe

KNOWLEDGE_VALIDATOR_ID = "confirmed-knowledge-edit-validator"


@contextlib.contextmanager
def file_lock(target: Path, timeout_s: float = 10.0, stale_after_s: float = 120.0):
    """Cross-process lock with bounded stale-lock recovery.

    Knowledge writes are short atomic operations. If a process is killed while
    holding the lock, the old implementation left a permanent ``.lock`` file.
    A sufficiently old lock is now recovered before the caller times out.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_suffix(target.suffix + ".lock")
    started = time.monotonic()
    fd: Optional[int] = None
    while fd is None:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = max(0.0, time.time() - lock.stat().st_mtime)
            except FileNotFoundError:
                continue
            if age >= max(10.0, float(stale_after_s)):
                try:
                    lock.unlink()
                    continue
                except FileNotFoundError:
                    continue
                except OSError:
                    pass
            if time.monotonic() - started > timeout_s:
                raise TimeoutError(f"lock_timeout:{lock}")
            time.sleep(0.05)
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "created_at": time.time()}).encode("utf-8"))
        yield
    finally:
        try:
            os.close(fd)
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass


def atomic_write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
            json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
            temp = Path(handle.name)
        os.replace(temp, path)


class AtomicJSONL:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Mapping[str, Any]) -> None:
        encoded = json.dumps(json_safe(dict(row)), ensure_ascii=False, sort_keys=True) + "\n"
        with file_lock(self.path):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except Exception:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        return rows

    def read(self) -> list[dict[str, Any]]:
        # Rewrites use os.replace, so an unlocked reader sees either the old or
        # new complete file. Appends are newline-delimited and malformed tails
        # are ignored defensively.
        return self._read_unlocked()

    def _rewrite_unlocked(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent,
                delete=False, suffix=".tmp",
            ) as handle:
                for row in rows:
                    handle.write(json.dumps(json_safe(dict(row)), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                temp = Path(handle.name)
            os.replace(temp, self.path)
        finally:
            if temp is not None and temp.exists():
                try:
                    temp.unlink()
                except OSError:
                    pass

    def rewrite(self, rows: Sequence[Mapping[str, Any]]) -> None:
        with file_lock(self.path):
            self._rewrite_unlocked(rows)

    def update(
        self,
        transform: Callable[[list[dict[str, Any]]], tuple[Sequence[Mapping[str, Any]], Any]],
    ) -> Any:
        """Atomically read, transform and rewrite without lost updates."""
        with file_lock(self.path):
            rows, result = transform(self._read_unlocked())
            self._rewrite_unlocked(rows)
            return result


class KnowledgeRuntime:
    """Retrieve verified structures and stage/promote accepted LLM lineages."""

    def __init__(self, library_path: Path | str, ledger_path: Path | str, max_entries: int = 64) -> None:
        library_path = Path(library_path)
        self.library = AtomicJSONL(library_path)
        suffix = library_path.suffix or ".jsonl"
        self.staging = AtomicJSONL(
            library_path.with_name(f"{library_path.stem}.staged{suffix}")
        )
        self.ledger = AtomicJSONL(ledger_path)
        self.max_entries = max(8, int(max_entries))
        self._last_commit_rejections: list[dict[str, Any]] = []

    @property
    def library_path(self) -> Path:
        return self.library.path

    @property
    def ledger_path(self) -> Path:
        return self.ledger.path

    @property
    def staging_path(self) -> Path:
        return self.staging.path

    @property
    def last_commit_rejections(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._last_commit_rejections]

    def library_digest(self) -> str:
        try:
            payload = self.library.path.read_bytes() if self.library.path.exists() else b""
        except OSError:
            payload = b""
        return hashlib.sha256(payload).hexdigest()

    def log_event(self, event: RuntimeEvent | Mapping[str, Any]) -> None:
        if isinstance(event, RuntimeEvent):
            payload = event.as_dict()
        else:
            payload = dict(event)
        self.ledger.append({"runtime_id": DISCOVERY_RUNTIME_ID, "timestamp": time.time(), **payload})

    @staticmethod
    def _failure_tokens(row: Mapping[str, Any]) -> set[str]:
        return {str(v) for v in row.get("failure_signature", []) if v}

    @staticmethod
    def _prompt_view(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "entry_id": row.get("entry_id"),
            "name": row.get("name"),
            "description": row.get("description"),
            "failure_signature": row.get("failure_signature", []),
            "before_pattern": row.get("before_pattern"),
            "after_pattern": row.get("after_pattern"),
            "edit_program": row.get("edit_program"),
            "action": row.get("action"),
            "evidence": row.get("evidence", {}),
            "successful_task_count": row.get("successful_task_count", 1),
        }

    def retrieve(self, failure_signature: Sequence[str], topk: int = 8) -> list[dict[str, Any]]:
        query = {str(v) for v in failure_signature if v}
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self.library.read():
            try:
                self._validate_entry(row)
            except Exception:
                # Fail closed: an invalid or corrupted memory entry is never
                # exposed to the proposal distribution.
                continue
            tokens = self._failure_tokens(row)
            overlap = len(query & tokens) / max(1, len(query | tokens)) if query or tokens else 0.0
            evidence = row.get("evidence") if isinstance(row.get("evidence"), Mapping) else {}
            gain = max(0.0, float(evidence.get("relative_val_gain") or 0.0))
            stability = float(evidence.get("ood_proxy_stability") or 0.0)
            scored.append((overlap + 0.20 * gain + 0.05 * stability, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._prompt_view(row) for _, row in scored[:max(0, int(topk))]]

    @staticmethod
    def _ordered_variables(before: str, after: str) -> list[str]:
        values: list[str] = []
        for token in re.findall(r"\bx\d+\b", before + " " + after):
            if token not in values:
                values.append(token)
        return values

    @staticmethod
    def _role_generalize(text: str, variables: Sequence[str]) -> tuple[str, dict[str, str]]:
        role_names = ("target_variable", "conditioning_variable", "context_variable")
        mapping: dict[str, str] = {}
        output = str(text)
        for index, variable in enumerate(variables):
            role = role_names[index] if index < len(role_names) else f"auxiliary_variable_{index - len(role_names) + 1}"
            mapping[variable] = role
            output = re.sub(rf"\b{re.escape(variable)}\b", role, output)
        return output, mapping

    _PATTERN_FUNCTIONS: Mapping[str, Any] = {
        "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "cbrt": sp.Function("cbrt"), "abspow": sp.Function("abspow"),
        "signpow": sp.Function("signpow"), "sigmoid": sp.Function("sigmoid"),
    }

    @staticmethod
    def _safe_symbol_names(values: Sequence[str]) -> tuple[str, ...]:
        out: list[str] = []
        for value in values:
            name = str(value)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"invalid_placeholder:{name}")
            if name not in out:
                out.append(name)
        return tuple(out)

    @classmethod
    def _parse_pattern(
        cls, text: str, variable_symbols: Sequence[str], coefficient_symbols: Sequence[str],
    ) -> sp.Expr:
        raw = str(text or "").strip()
        if not raw or len(raw) > 24000:
            raise ValueError("invalid_pattern_length")
        lowered = raw.lower()
        if any(token in lowered for token in ("__", "import", "lambda", "exec", "eval", "open(", "compile(")):
            raise ValueError("unsafe_pattern")
        variables = cls._safe_symbol_names(variable_symbols)
        coefficients = cls._safe_symbol_names(coefficient_symbols)
        local = dict(cls._PATTERN_FUNCTIONS)
        local.update({name: sp.Symbol(name, real=True) for name in (*variables, *coefficients)})
        expr = sp.sympify(raw, locals=local, evaluate=False)
        allowed_symbols = set(variables) | set(coefficients)
        unknown = sorted(str(symbol) for symbol in expr.free_symbols if str(symbol) not in allowed_symbols)
        if unknown:
            raise ValueError("unknown_pattern_symbols:" + ",".join(unknown))
        allowed_functions = set(cls._PATTERN_FUNCTIONS)
        bad_functions = sorted({
            str(getattr(atom.func, "__name__", atom.func))
            for atom in expr.atoms(sp.Function)
            if str(getattr(atom.func, "__name__", atom.func)) not in allowed_functions
        })
        if bad_functions:
            raise ValueError("unknown_pattern_functions:" + ",".join(bad_functions))
        return expr

    @classmethod
    def _coefficient_template(
        cls, text: str, prefix: str, role_symbols: Sequence[str],
    ) -> tuple[str, list[str]]:
        """Replace non-integer numeric atoms through the SymPy AST.

        String regex replacement corrupts scientific notation (for example,
        ``5.085e-5`` became ``5.AC8``). Traversing numeric atoms preserves the
        expression grammar and treats the complete scientific literal as one
        coefficient. Integer structural constants/exponents remain literal.
        """
        expr = cls._parse_pattern(text, role_symbols, ())
        names: list[str] = []

        def rewrite(node: sp.Expr) -> sp.Expr:
            if getattr(node, "is_Number", False):
                if bool(getattr(node, "is_Integer", False)):
                    return node
                name = f"{prefix}{len(names)}"
                names.append(name)
                return sp.Symbol(name, real=True)
            if not getattr(node, "args", ()):
                return node
            args = tuple(rewrite(child) for child in node.args)
            if isinstance(node, sp.Add):
                return sp.Add(*args, evaluate=False)
            if isinstance(node, sp.Mul):
                return sp.Mul(*args, evaluate=False)
            if isinstance(node, sp.Pow):
                return sp.Pow(*args, evaluate=False)
            return node.func(*args)

        rewritten = rewrite(expr)
        pattern = sp.sstr(rewritten, order="lex")
        # Round-trip immediately so malformed abstractions never reach commit.
        cls._parse_pattern(pattern, role_symbols, names)
        return pattern, names

    @classmethod
    def _validate_entry(cls, row: Mapping[str, Any]) -> dict[str, Any]:
        role_map = row.get("variable_roles") if isinstance(row.get("variable_roles"), Mapping) else {}
        variables = cls._safe_symbol_names([str(value) for value in role_map.values()])
        coeff_map = row.get("coefficient_symbols") if isinstance(row.get("coefficient_symbols"), Mapping) else {}
        before_coeffs = cls._safe_symbol_names([str(value) for value in (coeff_map.get("before") or [])])
        after_coeffs = cls._safe_symbol_names([str(value) for value in (coeff_map.get("after") or [])])
        before = cls._parse_pattern(str(row.get("before_pattern") or ""), variables, before_coeffs)
        after = cls._parse_pattern(str(row.get("after_pattern") or ""), variables, after_coeffs)
        program = row.get("edit_program")
        if not isinstance(program, Mapping):
            raise ValueError("missing_structured_edit_program")
        if str(program.get("schema") or "") != "knowledge-edit-v1":
            raise ValueError("unsupported_edit_program_schema")
        if str(program.get("operation") or "") != "replace_complete_equation":
            raise ValueError("unsupported_edit_program_operation")
        if str(program.get("before_pattern") or "") != str(row.get("before_pattern") or ""):
            raise ValueError("edit_program_before_mismatch")
        if str(program.get("after_pattern") or "") != str(row.get("after_pattern") or ""):
            raise ValueError("edit_program_after_mismatch")
        if program.get("requires_global_refit") is not True:
            raise ValueError("edit_program_requires_global_refit")
        coefficient_values = {
            name: 0.5 + 0.125 * index
            for index, name in enumerate((*before_coeffs, *after_coeffs))
        }
        modules = [{
            "cbrt": np.cbrt,
            "abspow": lambda value, power: np.abs(value) ** power,
            "signpow": lambda value, power: np.sign(value) * np.abs(value) ** power,
            "sigmoid": lambda value: 1.0 / (1.0 + np.exp(-value)),
        }, "numpy"]
        symbol_objects = [sp.Symbol(name, real=True) for name in variables]
        substitutions = {
            sp.Symbol(name, real=True): value for name, value in coefficient_values.items()
        }
        probes = [
            np.linspace(0.25 + 0.1 * index, 1.75 + 0.1 * index, 19)
            for index in range(len(symbol_objects))
        ]
        for label, expression in (("before", before), ("after", after)):
            fn = sp.lambdify(symbol_objects, expression.subs(substitutions), modules=modules)
            with np.errstate(all="ignore"):
                values = np.asarray(fn(*probes), dtype=float)
            if values.ndim == 0:
                values = np.full(19, float(values), dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{label}_pattern_nonfinite_on_replay_probe")
        serialized_program = json.dumps(json_safe(dict(program)), sort_keys=True, separators=(",", ":"))
        return {
            "passed": True,
            "replay_passed": True,
            "before_canonical": sp.sstr(before, order="lex"),
            "after_canonical": sp.sstr(after, order="lex"),
            "edit_program_sha256": hashlib.sha256(serialized_program.encode("utf-8")).hexdigest(),
            "validator_id": KNOWLEDGE_VALIDATOR_ID,
        }

    @staticmethod
    def _relative_gain(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> float:
        left = float(before.get(key, 0.0) or 0.0)
        right = float(after.get(key, 0.0) or 0.0)
        return 1.0 - right / max(abs(left), 1.0e-15)

    def _entry(self, step: LineageStep, failure_signature: Sequence[str]) -> dict[str, Any]:
        variables = self._ordered_variables(step.before_expression, step.after_expression)
        before_role, role_map = self._role_generalize(step.before_expression, variables)
        after_role, _ = self._role_generalize(step.after_expression, variables)
        role_symbols = tuple(role_map.values())
        before_pattern, before_coeffs = self._coefficient_template(before_role, "BC", role_symbols)
        after_pattern, after_coeffs = self._coefficient_template(after_role, "AC", role_symbols)
        action = str(step.actual_action or step.declared_action or "REPLACE")
        evidence = {
            "absolute_val_gain": float(step.before_metrics.get("val_nmse", 0.0)) - float(step.after_metrics.get("val_nmse", 0.0)),
            "relative_val_gain": self._relative_gain(step.before_metrics, step.after_metrics, "val_nmse"),
            "relative_tail_gain": self._relative_gain(step.before_metrics, step.after_metrics, "val_p99"),
            "relative_strict_gain": self._relative_gain(step.before_metrics, step.after_metrics, "val_strict"),
            "complexity_delta": float(step.after_metrics.get("complexity", 0.0)) - float(step.before_metrics.get("complexity", 0.0)),
            "stress_after": {
                "nmse": step.after_metrics.get("stress_nmse"),
                "p99": step.after_metrics.get("stress_p99"),
                "strict": step.after_metrics.get("stress_strict"),
            },
            "ood_proxy_after": {
                "nmse": step.after_metrics.get("ood_proxy_nmse"),
                "strict": step.after_metrics.get("ood_proxy_strict"),
            },
            "ood_proxy_stability": max(0.0, 1.0 - float(step.after_metrics.get("ood_stability_penalty", 0.0) or 0.0)),
        }
        fingerprint = json.dumps({"before": before_pattern, "after": after_pattern, "action": action}, sort_keys=True)
        entry_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
        entry = {
            "entry_id": entry_id,
            "runtime_id": DISCOVERY_RUNTIME_ID,
            "name": f"{action.replace('_', ' ').title()} refinement",
            "description": f"Apply a verified {action.lower()} edit when the listed failure signature recurs; globally refit all coefficients.",
            "failure_signature": sorted({str(v) for v in failure_signature if v}),
            "action": action,
            "before_pattern": before_pattern,
            "after_pattern": after_pattern,
            "edit_program": {
                "schema": "knowledge-edit-v1",
                "operation": "replace_complete_equation",
                "before_pattern": before_pattern,
                "after_pattern": after_pattern,
                "requires_global_refit": True,
            },
            "variable_roles": role_map,
            "coefficient_symbols": {"before": before_coeffs, "after": after_coeffs},
            "evidence": evidence,
            "successful_task_count": 1,
            "source_lineage_id": step.lineage_id,
            "created_at": time.time(),
        }
        entry["validation"] = self._validate_entry(entry)
        return entry

    def stage_final_lineage(
        self,
        final: EquationState,
        failure_signature: Sequence[str],
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Persist an internally accepted lineage outside the reusable library."""
        self._last_commit_rejections = []
        if not enabled or not final.is_llm or not final.lineage:
            return {"stage_id": "", "status": "not_staged", "entries": [], "rejections": []}

        entries: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        for step in final.lineage:
            try:
                entries.append(self._entry(step, failure_signature))
            except Exception as exc:
                rejections.append({
                    "stage": "new_entry_validation",
                    "lineage_id": step.lineage_id,
                    "reason": repr(exc),
                })
        stage_fingerprint = json.dumps({
            "final_hash": final.dag.canonical_hash,
            "lineage": [step.lineage_id for step in final.lineage],
            "entries": [entry["entry_id"] for entry in entries],
        }, sort_keys=True)
        stage_id = hashlib.sha256(stage_fingerprint.encode("utf-8")).hexdigest()[:32]
        record = {
            "stage_id": stage_id,
            "status": "staged" if entries and not rejections else "rejected",
            "runtime_id": DISCOVERY_RUNTIME_ID,
            "final_canonical_hash": final.dag.canonical_hash,
            "final_lineage_id": final.lineage_id,
            "final_lineage_depth": len(final.lineage),
            "entries": entries,
            "rejections": rejections,
            "created_at": time.time(),
            "promoted_at": None,
        }

        def transform(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            for existing in rows:
                if existing.get("stage_id") == stage_id:
                    return rows, existing
            return [*rows, record], record

        staged = self.staging.update(transform)
        self._last_commit_rejections = [dict(row) for row in staged.get("rejections", [])]
        for rejection in self._last_commit_rejections:
            self.log_event({"event": "structure_library_stage_rejected", **rejection})
        return dict(staged)

    def _promote_entries(
        self, stage_id: str, entries: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        def transform(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            existing: dict[str, dict[str, Any]] = {}
            rejections: list[dict[str, Any]] = []
            for row in rows:
                try:
                    validation = self._validate_entry(row)
                except Exception as exc:
                    rejections.append({
                        "stage": "existing_entry_validation",
                        "entry_id": row.get("entry_id"),
                        "reason": repr(exc),
                    })
                    continue
                clean = dict(row)
                clean["validation"] = validation
                if clean.get("entry_id"):
                    existing[str(clean["entry_id"])] = clean

            promoted: list[dict[str, Any]] = []
            for source in entries:
                try:
                    entry = dict(source)
                    entry["validation"] = self._validate_entry(entry)
                except Exception as exc:
                    rejections.append({
                        "stage": "promotion_entry_validation",
                        "entry_id": source.get("entry_id"),
                        "reason": repr(exc),
                    })
                    continue
                previous = existing.get(entry["entry_id"])
                promotion_ids = list(previous.get("promotion_stage_ids", [])) if previous else []
                if previous:
                    entry["successful_task_count"] = int(previous.get("successful_task_count", 1))
                    entry["created_at"] = previous.get("created_at", entry["created_at"])
                if stage_id not in promotion_ids:
                    if previous:
                        entry["successful_task_count"] += 1
                    promotion_ids.append(stage_id)
                entry["promotion_stage_ids"] = promotion_ids
                entry["last_promoted_at"] = time.time()
                existing[entry["entry_id"]] = entry
                promoted.append(entry)
            ordered = sorted(
                existing.values(),
                key=lambda row: (
                    int(row.get("successful_task_count", 1)),
                    float(row.get("evidence", {}).get("relative_val_gain", 0.0)),
                ),
                reverse=True,
            )[:self.max_entries]
            return ordered, {"promoted": promoted, "rejections": rejections}

        result = self.library.update(transform)
        self._last_commit_rejections = [dict(row) for row in result.get("rejections", [])]
        for rejection in self._last_commit_rejections:
            self.log_event({"event": "structure_library_promotion_rejected", **rejection})
        return result

    def promote_stage(
        self,
        stage_id: str,
        *,
        evidence_registry_path: Path | str,
        hypothesis_id: str,
    ) -> dict[str, Any]:
        """Promote only after a verified, passing untouched confirmation."""
        stage_key = str(stage_id or "").strip()
        if not stage_key:
            raise ValueError("stage_id is required")
        evidence_registry = EvidenceRegistry(evidence_registry_path)
        verification = evidence_registry.verify()
        if not verification.valid:
            raise RuntimeError("cannot promote from an invalid evidence registry")
        confirmation = [
            event
            for event in evidence_registry.events(hypothesis_id=str(hypothesis_id))
            if event.event_type is EvidenceEventType.TEST_OBSERVED
            and dict(event.payload).get("stage") == "untouched_heldout_confirmation"
            and dict(event.payload).get("independent_confirmation") is True
            and dict(event.payload).get("passed") is True
        ]
        if not confirmation:
            raise RuntimeError(
                "knowledge promotion requires a passing independent untouched confirmation"
            )
        matches = [row for row in self.staging.read() if row.get("stage_id") == stage_key]
        if not matches:
            raise KeyError(f"unknown_stage_id:{stage_key}")
        staged = matches[-1]
        if staged.get("status") == "promoted":
            return dict(staged)
        if staged.get("status") != "staged":
            raise ValueError(f"stage_not_promotable:{staged.get('status')}")
        result = self._promote_entries(stage_key, staged.get("entries") or [])
        rejections = list(result.get("rejections") or [])
        status = "promoted" if result.get("promoted") and not rejections else "promotion_rejected"
        promoted_at = time.time() if status == "promoted" else None

        def transform(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            updated: list[dict[str, Any]] = []
            promoted_record: dict[str, Any] = {}
            for row in rows:
                if row.get("stage_id") != stage_key:
                    updated.append(row)
                    continue
                promoted_record = {
                    **row,
                    "status": status,
                    "promoted_at": promoted_at,
                    "confirmation_hypothesis_id": str(hypothesis_id),
                    "confirmation_evidence_head_hash": verification.head_hash,
                    "promotion_rejections": rejections,
                    "promoted_entry_count": len(result.get("promoted") or []),
                }
                updated.append(promoted_record)
            return updated, promoted_record

        record = self.staging.update(transform)
        self.log_event({
            "event": "structure_library_stage_promotion",
            "stage_id": stage_key,
            "status": status,
            "promoted_entry_count": record.get("promoted_entry_count", 0),
        })
        return dict(record)

    def library_size(self, valid_only: bool = False) -> int:
        rows = self.library.read()
        if not valid_only:
            return len(rows)
        count = 0
        for row in rows:
            try:
                self._validate_entry(row)
            except Exception:
                continue
            count += 1
        return count

    def invalid_library_entry_count(self) -> int:
        return self.library_size(False) - self.library_size(True)
