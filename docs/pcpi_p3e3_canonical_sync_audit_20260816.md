# P3E.3 canonical source synchronization audit — 2026-08-16

Decision: **P3E.3 SOURCE RECOVERED AND VERIFIED; FROZEN REAL VALIDATION-ROLE
AUDIT REMAINS PENDING.**

## Recovered identities

- Uploaded Git bundle: `PCPI_P3E3_CALIBRATION_c0b48ab.bundle`.
- Bundle SHA-256:
  `55ff6f3086435871158965a4fca517d22520f1e6e6634d423da99262b1a90f92`.
- Frozen bundle commit:
  `c0b48ab618680783011d1002ed15f4cac05fc1cb`.
- Synchronized public-main commit:
  `2d811bbde248878c8f76b7fa20bd23deb5b382a0`.
- P3E.3 real config SHA-256:
  `2df76ba5ab577894f055244c96c819d704d136dc0e87ce193e3fdb01326fb3d9`.
- Production-code SHA-256:
  `a57bc56feeebde576af7b2204e3c3e752a00e3cffc26fb75db7854727f2c842c`.

`git bundle verify` reports a complete history and the expected bundle HEAD.
The bundle and synchronized public main have identical production code,
configs, tests, paper, and evidence documents. Their tree difference is limited
to `.gitattributes` and the delivery manifest regenerated after LF/EOL pinning.
Both source forms therefore reproduce the same frozen P3E.3 production-code
and real-config identities.

## Verification rerun

On the synchronized public main:

- delivery verification passed;
- the complete suite passed `249/249` tests;
- the P3E.3/leakage-focused subset passed `17/17` tests;
- syntax parsing passed;
- the production static audit checked 66 Python files and 16,300 lines with
  zero failures; and
- the deterministic P3E.3 fixture passed all five registered decisions with
  `heldout_opened=false`, `acquisition_comparison_performed=false`, and
  `acquisition_authorized=false`.

The rerun is correctness evidence only. It reads no CCPP, validation, or
held-out responses and supplies no calibration, adequacy, or efficacy claim.

## Synchronization corrections

The recovered P3E.3 tree already updates the manuscript, README, P3 decision
body, and delivery identity through P3E.3. The following post-freeze metadata
debts are corrected on the synchronization branch:

- the paper-completion assessment is advanced through the audited P3E.2 real
  non-rejection result and the pending P3E.3 real audit;
- the claim--code--evidence matrix records P3E.2 as protocol-valid
  non-rejection rather than a pending real audit;
- the P3 decision header is aligned with its detailed P3E.2/P3E.3 sections;
- install metadata declares the dependencies required by the formal runtime
  snapshot and the P3E.2/P3E.3 command entry points; and
- the manifest generator no longer states that the P3E.2 real result is
  unreported.

These post-freeze corrections do not alter the registered P3E.3 statistical
method or config. They also do not replace the experiment identity: the local
formal run remains bound to the supplied `c0b48ab` bundle and its registered
hashes unless a separately frozen replacement bundle is explicitly issued.

## Next Gate

The next admissible evidence is exactly the frozen CCPP validation-role P3E.3
audit. It may use 32 initial-development rows per seed for likelihood-power
selection and 256 registered validation rows per seed for the PIT diagnostic.
It must keep held-out closed and must not execute or compare an acquisition
policy. A threshold crossing rejects the registered predictive-calibration
null for that seed; no crossing is only non-rejection against the fixed betting
family. Neither outcome alone authorizes a new acquisition experiment.
