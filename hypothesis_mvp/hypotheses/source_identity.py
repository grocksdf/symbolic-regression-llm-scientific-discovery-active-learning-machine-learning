"""Stable, non-secret source identities for formal PCPI runs."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable
from zipfile import BadZipFile, ZipFile


def file_sha256(path: str | Path) -> str:
    """Hash one file without loading it into memory."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(root: Path, paths: Iterable[Path]) -> str:
    digest = sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def production_code_hash(project_root: str | Path) -> str:
    """Hash executable Python source while excluding data, outputs, and tests.

    This identity is intentionally unaffected by local datasets, run outputs,
    credentials, caches, test fixtures, and delivery metadata.
    """

    root = Path(project_root).resolve()
    paths: list[Path] = []
    for relative in ("hypothesis_mvp", "scripts"):
        base = root / relative
        if not base.is_dir():
            raise FileNotFoundError(f"production source directory is missing: {base}")
        paths.extend(
            path
            for path in base.rglob("*.py")
            if "__pycache__" not in path.relative_to(root).parts
        )
    return _tree_digest(root, paths)


def _git(
    root: Path,
    *arguments: str,
    text: bool = True,
) -> str | bytes:
    command = [
        "git",
        "-c",
        f"safe.directory={root.as_posix()}",
        "-C",
        str(root),
        *arguments,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=text,
        )
    except FileNotFoundError as error:
        raise RuntimeError("git is required for direct formal source identity") from error
    except subprocess.CalledProcessError as error:
        raw_stderr = error.stderr or ("" if text else b"")
        stderr = (
            str(raw_stderr)
            if text
            else bytes(raw_stderr).decode(errors="replace")
        )
        raise RuntimeError(f"git source identity failed: {str(stderr).strip()}") from error
    return completed.stdout


def verify_clean_git_source(project_root: str | Path) -> dict[str, object]:
    """Identify the exact clean Git worktree used by a formal run.

    This is the archive-free evidence path. It fails closed on staged, unstaged,
    or untracked files so the recorded commit and the bytes Python executes cannot
    silently diverge.
    """

    root = Path(project_root).resolve()
    top_level = Path(str(_git(root, "rev-parse", "--show-toplevel")).strip()).resolve()
    if top_level != root:
        raise RuntimeError(
            f"formal source root must be the Git worktree root: {top_level}"
        )
    status = str(
        _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    ).strip()
    if status:
        preview = "; ".join(status.splitlines()[:5])
        raise RuntimeError(f"formal source worktree is not clean: {preview}")
    commit = str(_git(root, "rev-parse", "HEAD")).strip().lower()
    git_tree = str(_git(root, "rev-parse", "HEAD^{tree}")).strip().lower()
    tracked_raw = bytes(_git(root, "ls-files", "--stage", "-z", text=False))
    entries = [item for item in tracked_raw.split(b"\0") if item]
    if not entries:
        raise RuntimeError("formal source Git worktree has no tracked files")
    digest = sha256()
    for entry in entries:
        try:
            metadata_bytes, relative_bytes = entry.split(b"\t", 1)
            mode_bytes, object_bytes, stage_bytes = metadata_bytes.split(b" ", 2)
        except ValueError as error:
            raise RuntimeError("Git returned an invalid tracked-file record") from error
        relative = os.fsdecode(relative_bytes)
        if not _safe_relative_path(relative):
            raise RuntimeError(f"Git returned an unsafe tracked path: {relative}")
        if stage_bytes != b"0":
            raise RuntimeError(f"formal source has an unresolved index stage: {relative}")
        digest.update(relative_bytes)
        digest.update(b"\0")
        digest.update(mode_bytes)
        digest.update(b"\0")
        if mode_bytes == b"160000":
            # A gitlink may be intentionally uninitialized. Its pinned commit is
            # still part of the exact parent-tree identity.
            digest.update(object_bytes)
            continue
        path = root / relative
        if path.is_symlink():
            content_hash = sha256(os.fsencode(os.readlink(path))).digest()
        elif path.is_file():
            content_hash = bytes.fromhex(file_sha256(path))
        else:
            raise RuntimeError(f"tracked formal source file is missing: {relative}")
        digest.update(content_hash)
    return {
        "source_identity_kind": "clean_git_worktree",
        "source_package_hash": None,
        "source_tree_hash": digest.hexdigest(),
        "source_git_commit": commit,
        "source_git_tree": git_tree,
        "source_git_dirty": False,
        "source_tracked_file_count": len(entries),
    }


def resolve_formal_source_identity(
    project_root: str | Path,
    source_artifact: str | Path | None = None,
) -> dict[str, object]:
    """Resolve either a verified ZIP identity or a clean local Git identity."""

    if source_artifact is None:
        return verify_clean_git_source(project_root)
    archive = Path(source_artifact).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"source artifact does not exist: {archive}")
    return {
        "source_identity_kind": "verified_source_archive",
        "source_package_hash": file_sha256(archive),
        "source_tree_hash": verify_source_artifact(project_root, archive),
        "source_git_commit": None,
        "source_git_tree": None,
        "source_git_dirty": None,
        "source_tracked_file_count": None,
    }


def delivery_source_tree_hash(project_root: str | Path) -> str:
    """Read the immutable source-tree identity from DELIVERY_MANIFEST.json."""

    path = Path(project_root).resolve() / "DELIVERY_MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "pcpi-delivery-manifest-v1":
        raise ValueError("unsupported delivery manifest schema")
    value = str(manifest.get("source_tree_sha256", ""))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("delivery source-tree hash is not a lowercase SHA-256 digest")
    return value


def _safe_relative_path(relative: str) -> bool:
    path = Path(relative)
    return bool(
        relative
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in relative
    )


def _validated_manifest(
    value: object,
) -> tuple[
    str,
    list[dict[str, object]],
    tuple[str, ...],
    list[dict[str, object]],
]:
    if not isinstance(value, dict) or value.get("schema") != "pcpi-delivery-manifest-v1":
        raise ValueError("unsupported delivery manifest schema")
    tree_hash = str(value.get("source_tree_sha256", ""))
    if len(tree_hash) != 64 or any(character not in "0123456789abcdef" for character in tree_hash):
        raise ValueError("delivery source-tree hash is not a lowercase SHA-256 digest")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("delivery manifest must contain a non-empty file inventory")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("delivery manifest file rows must be objects")
        relative = str(item.get("path", ""))
        path = Path(relative)
        digest = str(item.get("sha256", ""))
        size = item.get("size_bytes")
        if (
            not _safe_relative_path(relative)
            or relative in seen
        ):
            raise ValueError("delivery manifest contains an unsafe or duplicate path")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("delivery manifest contains an invalid file digest")
        if not isinstance(size, int) or size < 0:
            raise ValueError("delivery manifest contains an invalid file size")
        seen.add(relative)
        rows.append({"path": relative, "sha256": digest, "size_bytes": size})
    digest = sha256()
    for row in rows:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(str(row["sha256"])))
    if digest.hexdigest() != tree_hash:
        raise ValueError("delivery manifest source-tree hash does not match its inventory")
    archive_value = value.get("archive_only_files", [])
    if not isinstance(archive_value, list):
        raise ValueError("delivery manifest archive-only files must be a list")
    archive_rows: list[dict[str, object]] = []
    for item in archive_value:
        if not isinstance(item, dict):
            raise ValueError("delivery manifest archive-only rows must be objects")
        relative = str(item.get("path", ""))
        file_digest = str(item.get("sha256", ""))
        size = item.get("size_bytes")
        if not _safe_relative_path(relative) or relative in seen:
            raise ValueError("delivery manifest contains an unsafe or duplicate archive-only path")
        if len(file_digest) != 64 or any(
            character not in "0123456789abcdef" for character in file_digest
        ):
            raise ValueError("delivery manifest contains an invalid archive-only digest")
        if not isinstance(size, int) or size < 0:
            raise ValueError("delivery manifest contains an invalid archive-only file size")
        seen.add(relative)
        archive_rows.append(
            {"path": relative, "sha256": file_digest, "size_bytes": size}
        )
    mutable_value = value.get("mutable_unregistered_paths", [])
    if not isinstance(mutable_value, list):
        raise ValueError("delivery manifest mutable paths must be a list")
    mutable = tuple(str(item) for item in mutable_value)
    if (
        len(mutable) != len(set(mutable))
        or any(not _safe_relative_path(item) for item in mutable)
        or any(item in seen for item in mutable)
    ):
        raise ValueError("delivery manifest contains unsafe or conflicting mutable paths")
    return tree_hash, rows, mutable, archive_rows


def verify_local_delivery(project_root: str | Path) -> str:
    """Verify every manifest-registered local byte before a formal run."""

    root = Path(project_root).resolve()
    manifest_path = root / "DELIVERY_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tree_hash, rows, _, _ = _validated_manifest(manifest)
    for row in rows:
        path = (root / str(row["path"])).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"registered delivery file is missing or unsafe: {row['path']}")
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"registered delivery file size mismatch: {row['path']}")
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"registered delivery file hash mismatch: {row['path']}")
    return tree_hash


def verify_source_artifact(
    project_root: str | Path,
    source_artifact: str | Path,
) -> str:
    """Verify an archive inventory and require identity with the running tree."""

    local_tree_hash = verify_local_delivery(project_root)
    archive_path = Path(source_artifact).resolve()
    try:
        with ZipFile(archive_path) as archive:
            normalized: dict[str, str] = {}
            for name in archive.namelist():
                canonical = name.replace("\\", "/").strip("/")
                if not canonical or name.endswith(("/", "\\")):
                    continue
                if canonical in normalized:
                    raise ValueError("source archive contains duplicate normalized paths")
                normalized[canonical] = name
            manifests = [
                canonical for canonical in normalized
                if canonical == "DELIVERY_MANIFEST.json"
                or canonical.endswith("/DELIVERY_MANIFEST.json")
            ]
            if len(manifests) != 1:
                raise ValueError("source archive must contain one delivery manifest")
            manifest_name = manifests[0]
            prefix = manifest_name.rsplit("/", 1)[0] if "/" in manifest_name else ""
            manifest = json.loads(archive.read(normalized[manifest_name]).decode("utf-8"))
            archive_tree_hash, rows, mutable, archive_rows = _validated_manifest(manifest)
            expected = {manifest_name}
            for row in rows:
                relative = str(row["path"])
                canonical = f"{prefix}/{relative}" if prefix else relative
                expected.add(canonical)
                if canonical not in normalized:
                    raise ValueError(f"source archive is missing a registered file: {relative}")
                content = archive.read(normalized[canonical])
                if len(content) != int(row["size_bytes"]):
                    raise ValueError(f"source archive file size mismatch: {relative}")
                if sha256(content).hexdigest() != row["sha256"]:
                    raise ValueError(f"source archive file hash mismatch: {relative}")
            for relative in mutable:
                canonical = f"{prefix}/{relative}" if prefix else relative
                expected.add(canonical)
                if canonical not in normalized:
                    raise ValueError(f"source archive is missing a declared mutable file: {relative}")
            for row in archive_rows:
                relative = str(row["path"])
                canonical = f"{prefix}/{relative}" if prefix else relative
                expected.add(canonical)
                if canonical not in normalized:
                    raise ValueError(f"source archive is missing an archive-only file: {relative}")
                content = archive.read(normalized[canonical])
                if len(content) != int(row["size_bytes"]):
                    raise ValueError(f"source archive archive-only size mismatch: {relative}")
                if sha256(content).hexdigest() != row["sha256"]:
                    raise ValueError(f"source archive archive-only hash mismatch: {relative}")
            extras = set(normalized) - expected
            if extras:
                raise ValueError("source archive contains unregistered files")
    except BadZipFile as error:
        raise ValueError("source artifact is not a valid ZIP archive") from error
    if archive_tree_hash != local_tree_hash:
        raise ValueError("source archive and running delivery tree identities differ")
    return local_tree_hash


__all__ = [
    "delivery_source_tree_hash",
    "file_sha256",
    "production_code_hash",
    "resolve_formal_source_identity",
    "verify_clean_git_source",
    "verify_local_delivery",
    "verify_source_artifact",
]
