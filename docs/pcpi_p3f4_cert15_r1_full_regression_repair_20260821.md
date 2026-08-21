# P3F.4-CERT.15-R1 full-regression repair record

Status: **RESPONSE-FREE SOURCE AND ENVIRONMENT REPAIR; NO EXPERIMENT**

Parent source: `a5d0681204d9a08407315c773c9bdb59910ebf61`

CERT.15-R1 resolves all five failures inherited from the `00ecdf` baseline
before CERT.16 is allowed to begin.  It does not read or generate responses,
change a statistical result, relax a numerical tolerance, or authorize any
resident execution.

## Root-cause repairs

1. The fixed virtual environment contained a stale editable
   `hypothesis-mvp==1.0.0` distribution while the project and local egg metadata
   declared `1.4.0`.  Reinstalling the same checkout with `--no-deps --editable`
   removes the stale distribution and records one version identity.
2. The P3F.3 validation test expected an error string from before
   `post-bridge-always` became a registered finite-N schedule.  The test now
   verifies the complete registered schedule set; production validation is
   unchanged.
3. CERT.8 had modified the byte-frozen historical confirmatory
   `certification.py` to accept response prefixes.  The frozen file is restored
   to SHA-256
   `73b268c5a998be65c4a8ebc245471c2b1c3d1b54faef0fa06faa86ac749d3e8d`.
   Prefix padding now occurs only in the resident Feynman--Kac adapter before
   calling the frozen full-grid interface.
4. The final-source audit's universal 80/100-line threshold conflicted with
   eleven long functions that predated the audit.  The threshold is unchanged.
   Each legacy body is accepted only when its complete normalized source
   segment matches a registered SHA-256.  Any edit that remains over the limit,
   or any newly over-limit function, fails closed; shortening or deleting a
   legacy body is allowed without updating the registry.

## Verification boundary

The following response-free checks passed in the fixed Python 3.11 environment:

- the four originally failing contract/freeze/dependency tests plus the
  complete CERT.8 resident Feynman--Kac test module;
- the final-source audit with zero failures and eleven verified legacy source
  identities;
- the complete CERT.15 inherited Gate: 138 of 138 checks, 232 Git-tracked
  Python files, `python-flint==0.8.0`, 512-bit Arb and 256-bit threshold cells;
- the complete repository test suite: 518 of 518 tests with exit code zero.

No simulated experiment, formal experiment, real-data access, held-out access,
confirmatory materialization, resident SMC execution or island execution took
place.  The next admissible scientific phase remains the response-free CERT.16
integration theorem after the repaired source identity is independently
checked.
