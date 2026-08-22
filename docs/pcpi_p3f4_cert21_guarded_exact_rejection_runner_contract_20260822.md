# P3F.4-CERT.21 guarded exact-rejection runner contract

Status: frozen response-free runner candidate; identity-bound user Gate pending.

## 1. Purpose

CERT.20 closed the exact source composition but did not provide a complete
selection/confirmation execution transaction. CERT.21 supplies that missing
runner layer while keeping the operational entry point blocked before frozen
`H0`, system entropy or output-path access.

The runner has four responsibilities:

1. consume the exact CERT.20 rejection source under fixed proposal caps;
2. separate candidate selection and fixed-candidate confirmation transcripts;
3. convert every registered numerical, entropy or cap failure into one
   indivisible abstention without partial marks; and
4. publish exactly one hash-verified terminal JSON ledger without overwrite.

No response, random byte, real datum, acquisition value, heldout value or
formal experimental result is materialized by the CERT.21 Gate.

## 2. Identity-bound runner plan

`CertifiedExactRejectionRunnerPlan` binds:

- the complete CERT.20 source-plan hash;
- the actual CERT.18 evaluator and CERT.17 refinement hashes;
- the CERT.14 common target and provider contract;
- the explicit CERT.20 ideal-byte premise; and
- distinct logical selection and confirmation coordinate domains.

Crossing any identity fails before a runner can be built. The operational
runner repeats this complete identity check before constructing an entropy
source. Its frozen plan itself carries `operational_execution_authorized=false`.

## 3. Coordinate separation

Both phases inherit CERT.20's external iid-byte product-law premise. A
`CoordinateBoundIdealByteSource` records the logical domain, byte count and
request count of every source use. The selection and confirmation domain names
must differ and are validated before the first draw.

Logical domain binding is not claimed to prove physical independence. The
mathematical independence remains exactly the explicit external premise
accepted in CERT.20.

## 4. Fixed-cap batch semantics

For a required accepted count `n` and cap `T`, the batch engine consumes at
most `T` exact rejection decisions. Every proposal contributes a hash-chained
audit record containing:

- raw-state ID;
- operational class ID;
- proposal atom and core/tail role;
- exact accept/reject result;
- Arb refinement-round count; and
- revealed uniform-prefix bit count.

If `n` acceptances arrive, the complete accepted state/class sequence is
returned. If the cap is exhausted, all partial accepted marks are erased and
only the transcript hash, counts and abstention reason remain.

Registered draw failures include validated numerical failures, runtime source
failures, operating-system errors and invalid external values. These also
erase all partial marks and become `draw-failure` abstentions. Programming
`AssertionError`, `KeyboardInterrupt` and `SystemExit` are not masked as
scientific abstentions.

## 5. Selection and sequential confirmation

Selection first requests exactly the frozen 8192 accepted samples from its
own coordinate domain. A cap or draw failure ends the entire procedure. Only
a complete selection batch may enter the frozen empirical-mode selector.

The resulting candidate is fixed once. Confirmation then uses a fresh logical
coordinate domain and checks the CERT.20 exact binomial boundaries at accepted
sample counts `512, 2048, 8192, 32768`. It terminates at the first crossed
boundary. If the final stage is reached without crossing, it abstains. There
is no second candidate, stream replacement, cap extension or rerun.

An overall abstention never exposes the candidate, selection state IDs,
selection class IDs or partial confirmation state IDs. This prevents an
abstention ledger from becoming a favourable-retry selection channel.

## 6. Indivisible terminal evidence ledger

The terminal ledger is canonical JSON and contains one of these statuses:

- `confirmed`;
- `abstained-selection-cap`;
- `abstained-confirmation-cap`;
- `abstained-selection-failure`;
- `abstained-confirmation-failure`; or
- `abstained-no-boundary`.

A confirmed ledger contains both transcript hashes, the fixed candidate, the
complete selection sequence and the complete confirmation sequence used at
the crossed boundary. An abstention contains transcript hashes and proposal
counts only.

The writer serializes the complete payload to a unique sibling staging file,
calls `fsync`, and publishes it by a same-filesystem hard link. Hard-link
creation is atomic and fails if the target already exists. The staging link is
then removed. Therefore a reader sees either no terminal path or the complete
payload; an earlier terminal ledger cannot be overwritten by a rerun.

The ledger carries SHA-256 over its complete payload. Loading reconstructs the
strict field set and recomputes the digest. Added, removed or changed fields
fail verification.

Filesystem or process failure can still prevent publication; no software can
guarantee durable evidence when its storage device is unavailable. Such a
failure does not authorize a scientific result or retry.

## 7. Guard boundary

`GuardedOperationalExactRejectionRunner.run` checks three independent false
flags before inspecting any supplied object:

- operational execution;
- operational frozen-`H0` access; and
- system-entropy access.

The response-free test passes objects that raise on every attribute access and
proves the guard fires first. Real data, acquisition and heldout flags remain
separately false.

## 8. Deterministic Gate evidence

The CERT.21 checks cover:

1. complete runner-plan identity binding;
2. byte accounting and disjoint coordinate domains;
3. complete fixed-cap success and transcript identity;
4. cap erasure of partial accepted states;
5. numerical/draw failure erasure without masking programming assertions;
6. selection followed by first-stage fixed-candidate confirmation;
7. no-boundary and phase-failure ledgers with no candidate leakage;
8. atomic no-overwrite publication and verified reload;
9. tamper detection;
10. pre-access operational guards; and
11. equality with the frozen CERT.21 configuration.

All draws used by these state-machine checks are finite deterministic fixtures.
They are not simulated observations, Monte Carlo experiments, power evidence,
real-data evidence or efficacy results.

## 9. Remaining boundary

A passing CERT.21 Gate proves that the execution transaction and evidence
boundary are source-correct. It does not authorize an operational run.

Before giving the user a formal execution command, the next Gate must:

1. bind the actual registered `H0` artifact and output location identities;
2. perform a response-free cost preflight for the complete `J=17` core table,
   selection cap and confirmation cap;
3. decide explicitly whether the resulting certified wall-clock/storage
   budget is operationally feasible without lowering sample counts or changing
   the target; and
4. authorize exactly one no-overwrite run identity if and only if the preflight
   passes.
