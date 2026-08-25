# P3H.2 canonical runtime freeze

## Resolution of the apparent interpreter failure

The canonical launchers initially returned an `Unable to create process`
message when invoked inside the restricted Codex filesystem sandbox. The base
directory was outside that sandbox's readable roots. Direct read-only
verification in the approved host execution context established that the
interpreter had not been deleted and the virtual environment was not damaged.
No Python installation, virtual-environment rebuild, dependency installation
or package upgrade was performed.

The canonical executable remains
`D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe`. It resolves to
CPython 3.11.9 with base prefix
`C:\Users\19708\AppData\Local\Programs\Python\Python311`.

## Frozen dependency identity

The complete current runtime dependency hash is

`b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6`.

This is byte-for-byte the dependency hash recorded by the user-executed P3G.5
archive at source commit `86e37abb5ef5ff13c3cf4e151df60cee90c23df2`.
The full identities include:

- CPython `3.11.9`, `MSC v.1938 64 bit (AMD64)`;
- Windows, release `10`, machine `AMD64`;
- NumPy `2.4.6`;
- SciPy `1.17.1`;
- python-flint `0.8.0`;
- pytest `9.1.1`.

The environment contains 43 uniquely versioned distributions and `pip check`
returns `No broken requirements found`. The P3G.5 `summary.json` still matches
its manifest, so the historical identity source is intact.

## Frozen interpreter binaries

P3H.2 also freezes the executable surface that launches this environment:

| Object | Bytes | SHA-256 |
|---|---:|---|
| base `python.exe` | 103192 | `5f7b89a612c9b8af1d6456cdfcd1dbe5ca630849e79aebced9bee9a6694952ec` |
| base `python311.dll` | 5800216 | `0817a2a657a24c0d5fbb60df56960f42fc66b3039d522ec952dab83e2d869364` |
| canonical venv launcher | 274712 | `21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082` |
| canonical `pyvenv.cfg` | 321 | `b75562dbf80f212122390c6163f96335d54ed588187ea693d021393bd8b70de4` |

## Fail-closed execution order

The P3H.2 configuration contains this complete dependency and binary identity.
The real runner now performs the following before loading a registered dataset:

1. require a clean tracked worktree and record commit/tree;
2. re-execute all P3H.1 algebraic and leakage decisions;
3. materialize the complete runtime dependency snapshot;
4. require its hash, Python, platform and critical packages to equal the P3G.5
   identity;
5. hash the base executable, Python DLL, virtual-environment launcher and
   `pyvenv.cfg` and require exact equality;
6. only then permit provenance-verified dataset loading.

Any drift fails before response access. There is no fallback interpreter,
automatic environment repair, dependency installation or retry surface.
Candidate responses, acquisition and held-out remain blocked independently of
the runtime result.
