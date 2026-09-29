# BOQD family-calibration replay result

Date recorded: 2026-09-29

Status: **PASSED response-free development replay**

Paper synchronization status: **DEFERRED**

This record preserves the completed BOQD family-calibration replay as an
independent experiment artifact.  It must not be interpreted as realized
efficacy, held-out confirmation, or universal superiority.  The paper text,
figures, abstract, and conclusions are intentionally not updated from this
record until the remaining registered experiments are complete.

## Frozen identity

- Result:
  `D:\01\666\outputs\scientific_operational_qd_family_calibration_20260929\OPERATIONAL_QD_REPLAY_RESULT.json`
- Result SHA-256:
  `d2109bbb90b4dccc630e0b246ce126296edcf4c76b90e6290963ff1050383056`
- Freeze:
  `D:\01\666\registrations\scientific_operational_qd_family_calibration_20260929\OPERATIONAL_QD_REPLAY_FREEZE.json`
- Freeze SHA-256:
  `da2d9578f2e57278bf11e457d0995345d06f9becc55f2c5bc2ad8b5dc1a80fad`
- Mainline source commit:
  `5841b81bf5c74071c70dd5dc64d635e7bd8c3295`
- Benchmark source commit:
  `2bb099a0594a10b6e25073814662270b1dc7e6be`
- Source candidate artifact:
  `D:\01\666\outputs\scientific_aistats_synthesis_only_pilot_v2_20260928`
- Prefixes: `8, 16, 32`
- Seeds: `71, 72`

## Registered coordinates

- `bio_pop_growth/BPG9`
- `chem_react/CRK20`
- `lsr_transform/II.36.38_3_0`
- `matsci/MatSci12`

Both `full_scientist_v6` and `no_llm_v6` were replayed for every
task/seed coordinate, giving 16 accounted rows.

## Result

- Mean BOQD effect over the four registered families:
  `+0.04082346601706559`
- Population growth:
  `+0.029776041213343457`
- Chemical reactions:
  `0.0`
- Transformed laws:
  `+0.09623482505698067`
- Materials science:
  `+0.03728299779793824`
- Positive families: 3/4
- Tied families: 1/4
- Negative-transfer families: 0/4
- Failed rows: 0/16
- LLM novel-niche families:
  `bio_pop_growth`, `chem_react`, `matsci`

All registered decisions passed:

- all 16 rows accounted for;
- at least two LLM novel-niche families;
- at least two positive families;
- strictly positive mean boost;
- no negative-transfer family;
- zero failures.

## Response and claim boundary

- Candidate response accessed: `false`
- Held-out opened: `false`
- Test or OOD accessed: `false`
- `efficacy_demonstrated`: `false`

The admissible claim is:

> On the frozen synthesis-v2 candidate artifacts, conservative BOQD
> handover improves response-free fixed-prefix normalized decision-risk AULC
> in three of four registered families, ties in the fourth, and introduces no
> family-level negative transfer.

The result does **not** establish:

- realized acquisition efficacy;
- held-out or independent confirmation;
- universal superiority of BOQD;
- universal superiority of the LLM scientist;
- permission to change tasks, seeds, prefixes, thresholds, or source
  artifacts and rerun the frozen protocol.

## Relationship to earlier BOQD evidence

This result does not overwrite the earlier raw or conservative BOQD replay
artifacts.  Those remain immutable evidence on their own frozen candidate
sets.  The present positive result uses the prospectively frozen
synthesis-v2 candidate set and is the authoritative result only for the
`scientific_operational_qd_family_calibration_20260929` protocol.

## Deferred paper update

When all remaining experiments are complete, the consolidated paper update
should:

1. replace the current statement that BOQD cross-family benefit is unproved;
2. report the 3-positive/1-tied family result and mean effect `+0.04082`;
3. add a BOQD-versus-legacy fixed-prefix multi-panel figure from
   `BOQD_ROWS.json`;
4. retain the response-free and development-only claim boundary;
5. preserve the negative realized-DRR evidence as a separate evidence layer.

