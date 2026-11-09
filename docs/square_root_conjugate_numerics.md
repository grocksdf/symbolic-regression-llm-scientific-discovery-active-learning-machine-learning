# Square-root conjugate regression numerical repair

This is a numerical correctness change, not a new statistical method or an
efficacy result. No task, seed, candidate identity, score threshold, prior,
likelihood power, admission rule or experimental response enters its policy.

## Failure mechanisms

The former reference implementation formed `lambda I + w X'X`, solved that
normal equation and computed the noise scale by subtracting the posterior
quadratic from `w y'y + lambda m0'm0`. This has three distinct weaknesses:

* A finite feature can have an unrepresentable square. For example, a symbolic
  exponential feature of order `1e215` is finite but its normal equation is not.
* Normal equations square the conditioning of the augmented design.
* Near an exact fit, subtraction of nearly equal large quadratics can destroy
  the positive residual and prior penalty. The result can be negative or NaN.

Expanding `x' covariance x` can additionally cancel large opposing covariance
entries for collinear columns. Testing only whether each expression returns a
finite value cannot detect these downstream failures.

## Equivalent computation

For the existing Normal--Inverse-Gamma model, form the augmented matrix

    A = [sqrt(lambda) I, sqrt(lambda) m0;
         sqrt(w) X,      sqrt(w) y].

Let `D` contain the maximum absolute value of each column (exact zero columns
use one). Compute `A D^-1 = Q R`. With `D_x` and `d_y` the predictor and response
scales, partition `R` into coefficient block `R_x`, response column `r_xy` and
final diagonal `r_yy`. Then

    mean = D_x^-1 solve(R_x, r_xy) d_y
    covariance = D_x^-1 R_x^-1 R_x^-T D_x^-1
    alpha = a0 + w n / 2
    beta = b0 + (r_yy d_y)^2 / 2
    logdet(precision) = 2 sum(log(diag(R_x))) + 2 sum(log(diag(D_x))).

The prior rows are scaled along with the data; this does not install a new
prior in standardized coordinates. `beta` is a residual norm, not a difference
of energy estimates. There is no jitter, ridge change, clipping, pseudoinverse
or singular-value truncation.

Prediction uses `(x / D_x) solve(R_x, r_xy) d_y` and the squared norm
`||solve(R_x', x / D_x)||^2`. It does not need a representable unscaled dense
covariance. Online updates append the new observation to the compressed QR
rows **in scaled coordinates**; they do not reconstruct a potentially
overflowing unscaled R. SMC keeps its own sequential likelihood weights but
shares this numerical primitive. Its agreement with the reference alone is
therefore not an independent numerical oracle.

## Representation and failure boundaries

The QR factor and column scales are authoritative. Legacy unscaled precision,
information and response-square moments remain available when finite. An
unrepresentable compatibility moment is explicitly `None`, not infinity or a
clipped number. The factor is hash-bound both to these metadata and to its own
array bytes. State hashes used by class-conditional checkpoints include it.

`predictive_location_scale` returns Student-t locations, squared scales (not
variances), and degrees of freedom. Bank feedback and the regional structural
adapter use this route. Requests for an unrepresentable dense coefficient
covariance fail explicitly; legacy acquisition consumers requiring raw moments
also fail explicitly. Old states without QR factors retain a guarded legacy
path; already-lost residual information cannot be recovered from rounded
normal equations. Reconstruct such states from the original observations.

An unresolved normalized triangular system, nonfinite formula value,
unrepresentable noise scale or predictive distribution still fails closed.
The condition check is a floating-point resolution safeguard, not a certified
bound on all posterior errors. This patch does not claim arbitrary-precision
inference, universal numerical safety, or repairs to every historical
open-target/discrepancy implementation.

## Validation and experimental provenance

`tests/test_reference_square_root_stability.py` uses an independent 100-digit
mpmath oracle in the original coordinates, not the production QR code. It
covers large near-exact fits, duplicate columns and their prior-supported null
direction, nonzero prior means, fractional powers, every sequential prefix,
predictive densities, factor mutation, legacy ambiguity and feature scales up
to `1e308`. Tests also cover collapsed SMC and an actual symbolic crossfit
adapter on controlled inputs. These are correctness fixtures only.

A read-only diagnostic reconstructed the next unlogged C-arm edit of historical
coordinate 028 from frozen contexts and recorded paired feedback. On previously
opened fitting rows alone, its feature magnitude was approximately `7.16e215`:
the old computation produced NaN, whereas scaled QR produced finite evidence
and fitting predictive densities. No admission/reporting/confirmation/test/OOD
responses were accessed. This diagnostic neither reruns nor changes that
historical coordinate's failure result, and is not an efficacy comparison.

Numerical trajectories, posterior hashes and random coefficient draws can
change. Existing experiment freezes must not silently point at this source.
Use a new source identity and explicit continuation/new-run protocol for any
subsequent experiment; retain original successes, failures and artifacts.

References: LAPACK's [least-squares guide](https://www.netlib.org/lapack/lug/node27.html)
and [Working Note 149](https://www.netlib.org/lapack/lawnspdf/lawn149.pdf) discuss
QR and the accuracy limits of normal equations; the implementation uses
[triangular solves](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html).
