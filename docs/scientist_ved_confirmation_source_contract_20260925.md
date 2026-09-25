# VED independent-confirmation source contract

VED is an untouched real-data family for typed-synthesis confirmation. The
registered task predicts instantaneous fuel rate from five raw continuous
vehicle signals. No reciprocal, shifted-power or dataset-specific primitive is
required, so the frozen PCPI closed basis remains unchanged.

The source Gate binds:

- both original VED dynamic-data archives;
- the exact 7-Zip executable used only to list or stream registered members;
- three chronologically distinct open weeks for development, validation and
  acquisition-pool roles;
- a reserved confirmation week represented only by its member-name SHA-256;
- the exact five feature columns and fuel-rate target.

The source Gate lists archive metadata only. It does not extract a CSV, count
rows, inspect missingness, observe a feature or response, or reveal the
reserved confirmation member name.

Passing this Gate authorizes only implementation of a target-blind streaming
loader on controlled fixtures. Real VED execution remains blocked until that
loader proves exact column binding, deterministic row selection, disjoint
roles, archive/member identity, and non-access to the reserved confirmation
member.
