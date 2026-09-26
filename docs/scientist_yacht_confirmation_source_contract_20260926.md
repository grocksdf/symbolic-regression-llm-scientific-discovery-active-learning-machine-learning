# Yacht Hydrodynamics independent-confirmation source candidate

Yacht Hydrodynamics is the preferred replacement for the ineligible VED Fuel
Rate task. It contains physical resistance measurements from the Delft
systematic yacht hull series, six continuous inputs and one continuous target.
The frozen PCPI closed basis can consume the variables without a new primitive.

The no-download registration is supported by:

- the official UCI dataset identity, DOI and CC BY 4.0 license;
- a pinned TorchUncertainty implementation containing the official ZIP URL,
  filename, parser and MD5;
- a pinned GPflow benchmark definition classifying the task as real-data
  regression;
- a pinned independent seven-column semantic mapping.

The first five geometry variables define a hull group. Froude number varies
within a hull and is not part of the group identity. Groups, rather than rows,
are hash-partitioned before any response access:

- 8 development groups;
- 4 validation groups;
- 2 acquisition-pool groups;
- 3 unused open groups;
- 5 reserved confirmation groups.

The source candidate Gate performs no download. The next Gate may download the
official ZIP and verify bytes, seven numeric columns, 308 rows, 22 geometry
groups and 14 rows per group. It may not publish values or open confirmation
responses.

During byte/schema inspection, all six input columns may be decoded to assign
hull groups. The target column is numerically decoded only for development,
validation and acquisition-pool groups. Target tokens belonging to unused-open
or reserved-confirmation groups are not numerically decoded or summarized.
