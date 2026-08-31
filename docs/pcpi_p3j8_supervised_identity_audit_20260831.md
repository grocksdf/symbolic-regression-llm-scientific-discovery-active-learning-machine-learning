# P3J.8 supervised query-identity audit (2026-08-31)

P3J.8 binds each resumable query workspace to the exact source tree, frozen
configuration, dataset, seed, query index, ordered candidate identifiers and
coordinates, predictive target domain, representative observed domain, and
complete operational state. Any crossed identity fails before checkpointed
ranking.

The identity and terminal-failure artifacts use fsync staging followed by a
no-overwrite hardlink. Progress may be atomically replaced but carries the same
identity hash. A terminal failure cannot be overwritten or converted into a
retry.

The PowerShell supervisor is ASCII-safe for Windows PowerShell 5.1 and performs
only source/config/Python/worktree preflight. It contains no process-launch
path. Real-data execution remains blocked until P3J.9 composes the formal runner
with query workspaces, progress callbacks, fail-fast termination, and terminal
manifest verification.
