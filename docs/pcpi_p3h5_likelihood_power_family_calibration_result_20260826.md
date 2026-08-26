# PCPI P3H.5 likelihood-power family calibration result

## Frozen execution identity

The user executed the unique P3H.5 real calibration-only Gate on 2026-08-26.
The immutable execution identity was:

- source commit `53c081f66222496aed55376904c595107d41e1d0`;
- source tree `aa7d44ed7d0f98197fad867ca6c446cb5d731534`;
- config SHA-256
  `d1f945a899871fc70e3e1cebfaf5534fd4733c9388aed5f11cab146401710f97`;
- runtime dependency SHA-256
  `b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6`.

The recorded runner, posterior, residual-family, predictive-calibration and
semiparametric-residual source hashes match the frozen sources. P3H.1, P3H.3
and P3H.4 prerequisite evaluations all passed before registered data access,
and every P3H.5 deterministic self-check passed.

The output manifest validated exactly:

- `family_calibration_runs.csv`:
  `697419e28722add620ea2da1958cacf5cb9d24d0e3e6e97e37c81de37d539e24`;
- `summary.json`:
  `eb6b421c722da196367d5b9895135f00fef3e98e33c48e72876f4dc084abff3a`.

## Registered result

All 96 registered coordinates completed: three real datasets, eight frozen
seeds and four likelihood powers. Every coordinate used 256 validation
responses and the fixed per-coordinate e-process boundary 9600. There were no
rejections. The registered terminal status is therefore

`FAMILY_CALIBRATION_COMPATIBLE_ACQUISITION_BLOCKED`.

The maximum e-value over the complete family was `164.9035448844772`, about
58.2 times below the registered rejection boundary. Candidate-response access,
candidate selection, acquisition execution, held-out access, simulation and
formal efficacy evidence were all false.

The row audit found 96 distinct coordinate keys and 24 distinct final family
states. Initial, validation and candidate row commitments each had 16 distinct
values because the CO and NOx tasks share the same gas-turbine source rows and
split while retaining different response targets. This is expected and is not
a duplicated coordinate.

## Descriptive signals retained without changing the Gate

Twenty coordinates reached maximum e-value at least 10, eight reached at least
40, and three reached at least 100. None reached 1000 or 9600. Only one
coordinate ended above 10:

- Gas Turbine CO, seed `2026080707`, likelihood power `1.0`:
  maximum and final e-value `137.59472900466108`, mean PIT `0.5027295935`, PIT
  variance `0.0640975655`, lower-decile rate `0.0625`, upper-decile rate
  `0.07421875`.

Its mean is centered, while its variance and tail rates indicate a locally
under-dispersed corrected-PIT sequence. Two other Gas CO coordinates had larger
transient maxima but small terminal values: seed `2026080703`, power `0.25`
reached `164.9035` and ended at `0.04757`; seed `2026080701`, power `0.125`
reached `116.4674` and ended at `0.02561`.

These are mandatory descriptive audit findings, not new testing boundaries.
They do not justify changing the registered decision, removing a power,
selecting a power after observing results, tuning the residual tree or rerunning
seeds. In particular, the persistent Gas CO signal makes the frozen maximin
aggregation over the complete likelihood-power family more important; it does
not license a post-hoc eta choice.

## Decision and next authorization boundary

P3H.5 removes the family-calibration blocker only in its preregistered narrow
sense. It does not yet authorize a real acquisition experiment because the
existing operational runner does not construct and advance the conditioned
likelihood-power residual family.

P3H.6 must be a correctness-only operational-lifecycle Gate. Before any new
real acquisition response is opened, it must establish all of the following:

1. reconstruct the four-model family only from the frozen 16-observation base
   warmup and 16-observation strict-prefix residual-training history;
2. discard every P3H.5 validation response and validation-updated state;
3. pass the exact bound family into the P3H transformed maximin joint utility
   for every candidate scoring decision, with no Student-t fallback;
4. either produce a certified ranking or terminally abstain when numerical
   utility intervals overlap;
5. open only the selected candidate response, then advance every model's base
   posterior and residual state exactly once from the common opened history;
6. retain all four frozen likelihood powers throughout the run, with no
   result-dependent eta selection, retry, replacement or state sharing;
7. keep candidate targets sealed before selection and held-out sealed for the
   complete acquisition phase.

Until that lifecycle Gate passes and a separate formal protocol is frozen,
`operational_execution_authorized=false` remains the only valid state.

## Claim boundary

The P3H.5 outcome supports the statement that the four registered transformed
predictive families were not rejected by the fixed simultaneous corrected-PIT
audit on the registered real development data. It does not prove conditional
calibration, model correctness, acquisition quality, superiority, held-out
improvement, scientific discovery or an AISTATS contribution.
