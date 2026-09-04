# P3K.7 pre-data runtime failure and P3K.8 complete freeze

## Preserved P3K.7 failure

P3K.7 launched with the expected source, configuration, branch, tree, and
virtual-environment launcher hash, then failed at the dependency-environment
gate. The failure occurred before `_prepare_output` and before registered-data
loading. Consequently the named experiment directory was never created. The
only artifacts are adjacent stdout/stderr process logs containing the runtime
exception. There is no acquisition decision, revealed response, validation
access, held-out access, or efficacy result to resume or interpret.

P3K.7 is frozen as `P3K.7/pre-data-mutable-base-runtime-drift`. Its current
runner and supervisor reject execution and direct all future work to P3K.8.

## Root cause

The P3K.7 preflight hashed
`D:/01/666/.venvs/hypothesis_mvp_p3j15_py312/Scripts/python.exe`. That file is
a virtual-environment launcher, not the base interpreter. Its hash remained
`560b...`, while `pyvenv.cfg` continued to reference a Codex-managed runtime.
The manager replaced that base interpreter in place from CPython 3.12.13 to
3.12.14. Installed distribution versions were unchanged, but the Python version
inside the dependency snapshot changed, yielding `f2c0...` instead of the
frozen `6b8c...` hash.

Thus the old preflight tested a stable pointer while omitting the mutable object
to which it pointed. Updating the formal configuration to accept 3.12.14 would
violate the registered runtime identity and is not used as a repair.

## Restored immutable environment

P3K.8 uses the workspace-local base runtime
`D:/01/666/.runtimes/python-3.12.13-p3j15-frozen` and virtual environment
`D:/01/666/.venvs/hypothesis_mvp_p3k8_py31213`. The interpreter reports CPython
3.12.13, NumPy 2.5.2, SciPy 1.18.1, python-flint 0.8.0, and pytest 9.1.1. Its
complete dependency snapshot exactly recovers the registered hash
`6b8c2611d77caea375d0d90d2e84627c725f0df845194bff14e25d44b4492ad9`.

The following identities are additionally frozen:

| Component | Length | SHA-256 |
|---|---:|---|
| base executable | 91648 | `d8e3f0adf246db00358c0c4ed349cf714898178f9558fb0e944f79f5c07f8eaa` |
| versioned Python DLL | 6969856 | `64a1dad031e97f13b1a0bac26c689d8e14a18d7dd1eab06e17f70e22373f4eec` |
| stable-ABI DLL | 56320 | `2d2330ce33d1443c67b1804bbf1a561653499dd4dcf8918048747cc17d1a63c4` |
| virtual-environment launcher | 262144 | `560b9ef7d856608ab8da02ded2dc8a1951ad1f424c382c0ec6a698874165a18e` |
| `pyvenv.cfg` | 306 | `215ca7981b0a4d50d5eee7a23d5157c05d4ea867e8182c04f3cfcc26bdaf5106` |

The formal runner recomputes the dependency and complete binary identities
before validating the data directory. The supervisor performs the same checks
before either preflight success or child-process launch. A future in-place
interpreter update therefore fails during supervised preflight rather than
after a formal process starts.

## P3K.8 boundary

P3K.8 changes only runtime storage and identity verification. It preserves all
P3K.7 datasets, seeds, splits, budgets, baselines, likelihood powers,
shared-innovation posterior state, semiparametric utility, minimum-violation
representative projection, singleton certificate, numerical schedule,
assessment rules, fail-fast behavior, and held-out-closed boundary. It remains
failure-informed real-development evidence and is not independent confirmation.
