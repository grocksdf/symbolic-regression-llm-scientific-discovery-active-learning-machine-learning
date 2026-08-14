# Public canonical-source synchronization audit — 2026-08-14

Decision: **SOURCE SYNCHRONIZATION RESOLVED; P3D.2 REAL INTEGRATION IMPLEMENTED;
LOCAL MEASURED-POOL EXECUTION REMAINS PENDING.**

## Git identity

- Repository: `grocksdf/symbolic-regression-llm-scientific-discovery-active-learning-machine-learning`.
- Imported base commit: `5fa4f3c9080c31208135dcada80a8f86121a199a`.
- Base history contains one clean-import commit; the historical P3C.1 run commit
  `d4b24b2075926db4fd9ca3cefc5637a7c2378d13` is not an ancestor and cannot be
  diffed against this import.
- P3C.1 result identity remains the audited historical identity; it is not
  rewritten as the current source identity.
- Upstream restoration commit:
  `81f7cde` (`Restore hypothesis_mvp.data package excluded by overbroad data ignore`).
- Merge commit on the P3D branch: `bb69853`.

## Manifest/source mismatch and resolution

The imported `DELIVERY_MANIFEST.json` lists six production files that are not
present in the Git tree. The repository-level `.gitignore` used the recursive
pattern `data/`, which also ignored the Python package directory
`hypothesis_mvp/data`. This branch narrows that rule to `/data/`; it does not
reconstruct missing source from tests or from memory.

| Missing path | Recorded SHA-256 | Bytes |
|---|---|---:|
| `hypothesis_mvp/data/__init__.py` | `8c716df00bb38748efae6fa2224ad714ed8c9b7ee7df1f05f1e8ddb77b5e6477` | 1,041 |
| `hypothesis_mvp/data/oracle.py` | `24cfec4ce0dc7819325e6df08088f36cf1205e3d7e6541d1b3e1979aa9ac70f6` | 2,445 |
| `hypothesis_mvp/data/real_data.py` | `bb1968688b9e9d96d6bea6822cd79372aed6acbace09b7d03dcd5559b10073eb` | 6,668 |
| `hypothesis_mvp/data/real_protocol.py` | `78b4ef59035a9056806d9f7a282351972849da82d1690487598358f5013132cb` | 7,370 |
| `hypothesis_mvp/data/real_registry.py` | `a1bf21c4630416771fffdd628056c795c7ec6096724ee3146a9f2f9baa680999` | 10,321 |
| `hypothesis_mvp/data/roles.py` | `9da8ed319a61ebe59ca98b978c1f4351c7291e8b46d655a811f446d8c139453a` | 7,473 |

Upstream commit `81f7cde` restores these exact files. Each returned SHA-256 and
byte count matches the table, so the files are accepted as the canonical
historical source rather than reconstructions from test expectations.

## Test consequence and closure

Before restoration, collection stopped with 14 import errors, principally
`ModuleNotFoundError: hypothesis_mvp.data`. After merging the verified files,
the complete current suite returns `186 passed` with zero failures, skips, or
collection errors. The regenerated delivery manifest therefore records
`source_identity.complete=true` and `suite_status=passed`.

## Gate boundary

- Static AST/source integrity and full regression are now evaluable.
- P3D.1 exact finite fixture and its isolated tests remain correctness evidence.
- P3D.2 adds analytic information-inequality bounds, a real-runtime integration,
  a frozen real-only protocol, and explicit closed-heldout enforcement at
  implementation commit `cd05e1a`; the expanded suite passes 199 tests.
- These source changes do not constitute a real result. A new CCPP/Gas run must
  be executed on the user's official local measurements without changing the
  frozen config, then audited before any efficacy statement. P4/P5, motif
  efficacy, VED, and held-out confirmation remain blocked.
