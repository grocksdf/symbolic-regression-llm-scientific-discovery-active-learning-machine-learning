# Public canonical-source synchronization audit — 2026-08-14

Decision: **P3D.1 correctness work may proceed in isolation; full-source
regression and any real run remain blocked by an incomplete public import.**

## Git identity

- Repository: `grocksdf/symbolic-regression-llm-scientific-discovery-active-learning-machine-learning`.
- Imported base commit: `5fa4f3c9080c31208135dcada80a8f86121a199a`.
- Base history contains one clean-import commit; the historical P3C.1 run commit
  `d4b24b2075926db4fd9ca3cefc5637a7c2378d13` is not an ancestor and cannot be
  diffed against this import.
- P3C.1 result identity remains the audited historical identity; it is not
  rewritten as the current source identity.

## Manifest/source mismatch

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

These exact files must be restored from the user's canonical Windows source or
another provenance-verified artifact and checked against the recorded hashes.
Creating replacements from test expectations would destroy source identity.

## Test consequence

The full suite reaches collection and stops with 14 import errors, principally
`ModuleNotFoundError: hypothesis_mvp.data`. Therefore the old manifest's
`146 passed` field is historical and cannot be represented as a current-HEAD
rerun. P3D.1's isolated test module does not import the missing package and is
reported separately. A regenerated manifest must mark the full suite
`blocked_missing_source` until the six files are restored and all tests are
rerun.

## Gate boundary

- Static AST/source integrity may be evaluated on the files actually present.
- P3D.1 exact finite fixture and its isolated tests may be evaluated.
- Full regression, real-data smoke, CCPP/Gas development, held-out
  confirmation, P4/P5, motif efficacy, and VED remain blocked.
