"""Response-free CERT.11 product-source and operational-projector proofs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import inspect

from hypothesis_mvp.pcpi.open_target import (
    P3F4_CERT11_CERTIFIED_CDF_INTERVAL_ORACLE_IMPLEMENTATION_AUTHORIZED,
    P3F4_CERT11_KEY_MANIFEST_SCHEMA,
    P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA,
    P3F4_CERT11_PRODUCT_SOURCE_SCHEMA,
    P3F4_CERT11_PRODUCT_STREAM_MATERIALIZATION_AUTHORIZED,
    P3F4_CERT11_PROJECTOR_RESULT_ACCESS_AUTHORIZED,
    P3F4_CERT11_SYSTEM_ENTROPY_CAPTURE_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED,
    P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED,
    AuditablePhiloxProductRandomSource,
    CertifiedOperationalStateRecord,
    CertifiedProbabilityInterval,
    FullSupportOperationalClassProjector,
    ResidentOperationalEstimandSpec,
    ResidentPhiloxKeyManifest,
    ResidentPhiloxProductSourceContract,
    build_resident_island_stream_coordinates,
    operational_projector_hash,
    project_certified_operational_records,
)
import hypothesis_mvp.pcpi.open_target.resident_product_projector as implementation


ORACLE_HASH = "certified-student-t-cdf-outward-interval-oracle-contract-v1"


@dataclass(frozen=True)
class _PlanStub:
    stable_hash: str
    product_law_hash: str
    island_count: int
    contract_hash: str = "posterior-contract"
    operational_estimand_hash: str = "operational-estimand"
    class_projector_hash: str = "class-projector"
    class_ids: tuple[str, ...] = ("class-a", "class-b")


def _source_plan(identity: str = "plan-a") -> _PlanStub:
    return _PlanStub(
        stable_hash=identity,
        product_law_hash="external-independent-product-law-v1",
        island_count=3,
    )


def _source_contract_and_manifest(plan: _PlanStub):
    contract = ResidentPhiloxProductSourceContract.from_island_plan(plan)
    manifest = ResidentPhiloxKeyManifest(
        schema=P3F4_CERT11_KEY_MANIFEST_SCHEMA,
        source_contract_hash=contract.stable_hash,
        plan_hash=plan.stable_hash,
        coordinate_hashes=contract.coordinate_hashes,
        key_hex_by_coordinate=(
            "00000000000000000000000000000001",
            "00000000000000000000000000000002",
            "ffffffffffffffffffffffffffffffff",
        ),
    )
    return contract, manifest


def _spec(action_grid: tuple[tuple[float, ...], ...] = ((0.0,),)):
    return ResidentOperationalEstimandSpec(
        schema=P3F4_CERT11_OPERATIONAL_ESTIMAND_SCHEMA,
        initial_history_hash="frozen-initial-history-h0",
        initial_standardizer_hash="frozen-center-scale-standardizer",
        action_grid=action_grid,
        response_threshold_grid=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    )


def _constant_intervals(
    spec: ResidentOperationalEstimandSpec,
    value: Fraction,
) -> tuple[CertifiedProbabilityInterval, ...]:
    return tuple(
        CertifiedProbabilityInterval(value, value)
        for _ in range(spec.coordinate_count)
    )


class _OracleStub:
    full_open_support = True
    certified_outward_intervals = True
    future_response_access = False

    def __init__(self, spec: ResidentOperationalEstimandSpec) -> None:
        self.oracle_contract_hash = ORACLE_HASH
        self.operational_estimand_hash = spec.stable_hash
        self.initial_history_hash = spec.initial_history_hash

    def cdf_intervals(self, particle):
        raise AssertionError("CERT.11 response-free checks must not access particles")


class _AccessBomb:
    def __getattribute__(self, name):
        raise AssertionError(f"guard failed before forbidden access: {name}")


def test_cert11_retains_all_cert10_and_new_execution_guards() -> None:
    assert P3F4_RESIDENT_ISLAND_EXECUTOR_RUN_AUTHORIZED is False
    assert P3F4_RESIDENT_ISLAND_PRODUCT_SOURCE_AUTHORIZED is False
    assert P3F4_RESIDENT_ISLAND_PROJECTOR_AUTHORIZED is False
    assert P3F4_CERT11_SYSTEM_ENTROPY_CAPTURE_AUTHORIZED is False
    assert P3F4_CERT11_PRODUCT_STREAM_MATERIALIZATION_AUTHORIZED is False
    assert P3F4_CERT11_CERTIFIED_CDF_INTERVAL_ORACLE_IMPLEMENTATION_AUTHORIZED is False
    assert P3F4_CERT11_PROJECTOR_RESULT_ACCESS_AUTHORIZED is False


def test_product_source_contract_binds_every_ordered_coordinate_directly() -> None:
    plan = _source_plan()
    contract = ResidentPhiloxProductSourceContract.from_island_plan(plan)
    coordinates = build_resident_island_stream_coordinates(plan)
    assert contract.schema == P3F4_CERT11_PRODUCT_SOURCE_SCHEMA
    assert contract.plan_hash == plan.stable_hash
    assert contract.product_law_hash == plan.product_law_hash
    assert contract.coordinate_hashes == tuple(item.stable_hash for item in coordinates)
    assert len(set(contract.coordinate_hashes)) == plan.island_count
    assert contract.bit_generator == "numpy.random.Philox"
    assert contract.key_bits == 128
    assert contract.initial_counter == 0
    assert contract.root_key_derivation_used is False
    assert contract.seedsequence_spawn_used is False
    assert contract.jumped_streams_used is False
    assert contract.collision_retry_authorized is False
    assert contract.favourable_key_selection_authorized is False


def test_key_manifest_is_one_key_per_coordinate_auditable_and_fail_closed() -> None:
    plan = _source_plan()
    contract, manifest = _source_contract_and_manifest(plan)
    assert len(manifest.key_commitments) == plan.island_count
    assert len(set(manifest.key_commitments)) == plan.island_count
    assert manifest.key_for_coordinate(contract.coordinate_hashes[0]) == 1
    assert manifest.key_for_coordinate(contract.coordinate_hashes[2]) == 2**128 - 1
    assert "00000000000000000000000000000001" not in repr(manifest)
    assert manifest.audit_record()["raw_keys_exposed"] is False
    assert manifest.audit_record()["retry_count"] == 0

    invalid_changes = (
        {"key_hex_by_coordinate": (manifest.key_hex_by_coordinate[0],) * 3},
        {"key_hex_by_coordinate": ("0",) * 3},
        {"coordinate_hashes": manifest.coordinate_hashes[:-1]},
    )
    for changes in invalid_changes:
        try:
            replace(manifest, **changes)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid CERT.11 key manifest must fail closed")

    source = AuditablePhiloxProductRandomSource(plan, contract, manifest)
    assert source.audit_record["external_independence_premise_proved_by_source"] is False
    crossed_plan = _source_plan("plan-b")
    try:
        AuditablePhiloxProductRandomSource(crossed_plan, contract, manifest)
    except ValueError:
        pass
    else:
        raise AssertionError("cross-plan CERT.11 product source must fail")


def test_actual_product_source_is_guarded_before_entropy_or_coordinate_access() -> None:
    plan = _source_plan()
    contract, manifest = _source_contract_and_manifest(plan)
    source = AuditablePhiloxProductRandomSource(plan, contract, manifest)
    capture_source = inspect.getsource(
        AuditablePhiloxProductRandomSource.capture_from_system_entropy
    )
    materialize_source = inspect.getsource(
        AuditablePhiloxProductRandomSource.materialize_coordinate
    )
    assert capture_source.index("if (") < capture_source.index("source_contract =")
    assert capture_source.count("secrets.token_bytes(16)") == 1
    assert "while " not in capture_source
    assert "SeedSequence" not in capture_source
    assert ".spawn(" not in capture_source
    assert ".jumped(" not in capture_source
    assert materialize_source.index("if (") < materialize_source.index(
        "coordinate_hash = coordinate.stable_hash"
    )
    assert "np.random.Philox(key=key, counter=0)" in materialize_source
    assert "default_rng" not in materialize_source
    assert "seed=" not in materialize_source
    try:
        AuditablePhiloxProductRandomSource.capture_from_system_entropy(_AccessBomb())
    except RuntimeError:
        pass
    else:
        raise AssertionError("system entropy capture guard must remain closed")
    try:
        source.materialize_coordinate(_AccessBomb())
    except RuntimeError:
        pass
    else:
        raise AssertionError("coordinate materialization guard must remain closed")


def test_operational_estimand_freezes_h0_grid_budget_and_claim_domain() -> None:
    spec = _spec(((-1.0,), (0.0,), (1.0,)))
    assert spec.future_budget == 32
    assert spec.bin_count == 6
    assert spec.coordinate_count == 21
    assert spec.class_space_size == 6**21
    assert spec.response_probability_levels == (
        Fraction(1, 20),
        Fraction(3, 20),
        Fraction(3, 10),
        Fraction(1, 2),
        Fraction(7, 10),
        Fraction(17, 20),
        Fraction(19, 20),
    )
    assert spec.claim_domain == "registered-action-threshold-grid-only"
    assert spec.exact_polynomial_classes_used is False
    assert spec.result_derived_grid_used is False
    assert spec.future_response_access is False

    for changed in (
        {"future_budget": 31},
        {"action_grid": ((1.0,), (0.0,))},
        {"response_threshold_grid": (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.0)},
        {"future_response_access": True},
    ):
        try:
            replace(spec, **changed)
        except ValueError:
            pass
        else:
            raise AssertionError("changed CERT.11 estimand identity must fail")


def test_implicit_operational_class_rank_is_a_complete_bijection() -> None:
    spec = _spec()
    observed_signatures = set()
    for rank in range(spec.class_space_size):
        signature = spec.signature_from_rank(rank)
        assert spec.class_rank(signature) == rank
        observed_signatures.add(signature)
    assert len(observed_signatures) == 6**7
    for rank in (0, 1, 6, 6**7 - 1):
        identifier = spec.class_id(spec.signature_from_rank(rank))
        assert spec.signature_from_class_id(identifier) == spec.signature_from_rank(rank)


def test_class_identity_is_support_extension_and_population_order_invariant() -> None:
    spec = _spec()
    signature = (0, 1, 2, 3, 4, 5, 0)
    identifier = spec.class_id(signature)
    discovered_a = ("state-z", "state-a", "state-new")
    discovered_b = tuple(reversed(discovered_a)) + ("unseen-future-state",)
    labels_a = {state: spec.class_id(signature) for state in discovered_a}
    labels_b = {state: spec.class_id(signature) for state in discovered_b}
    assert labels_a["state-z"] == labels_b["state-z"] == identifier
    assert labels_a["state-a"] == labels_b["state-a"] == identifier
    assert "state" not in identifier


def test_exact_interval_binning_obeys_boundaries_without_nearest_rounding() -> None:
    bin_count = 6
    assert CertifiedProbabilityInterval(Fraction(0), Fraction(0)).possible_bins(bin_count) == (0,)
    assert CertifiedProbabilityInterval(Fraction(1), Fraction(1)).possible_bins(bin_count) == (5,)
    assert CertifiedProbabilityInterval(Fraction(1, 6), Fraction(1, 6)).possible_bins(bin_count) == (1,)
    assert CertifiedProbabilityInterval(Fraction(5, 6), Fraction(5, 6)).possible_bins(bin_count) == (5,)
    assert CertifiedProbabilityInterval(Fraction(1, 7), Fraction(1, 5)).possible_bins(bin_count) == (0, 1)
    assert CertifiedProbabilityInterval(Fraction(1, 6), Fraction(1, 3)).possible_bins(bin_count) == (1, 2)
    try:
        CertifiedProbabilityInterval(Fraction(-1, 10), Fraction(1, 10))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid CDF enclosure must fail")


def test_sparse_exact_pushforward_preserves_mass_order_and_support_splits() -> None:
    spec = _spec()
    low = _constant_intervals(spec, Fraction(1, 12))
    high = _constant_intervals(spec, Fraction(3, 4))
    records = (
        CertifiedOperationalStateRecord("state-a", Fraction(1, 3), low),
        CertifiedOperationalStateRecord("state-b", Fraction(2, 3), high),
    )
    projected = project_certified_operational_records(spec, ORACLE_HASH, records)
    reversed_projection = project_certified_operational_records(
        spec,
        ORACLE_HASH,
        tuple(reversed(records)),
    )
    extended = project_certified_operational_records(
        spec,
        ORACLE_HASH,
        (
            CertifiedOperationalStateRecord("state-a1", Fraction(1, 6), low),
            CertifiedOperationalStateRecord("state-a2", Fraction(1, 6), low),
            records[1],
        ),
    )
    assert projected.exact_mass_by_rank == reversed_projection.exact_mass_by_rank
    assert projected.exact_mass_by_rank == extended.exact_mass_by_rank
    assert projected.boundary_uncertain == ()
    assert projected.normalization_applied is False
    assert len(projected.exact_mass_by_rank) == 2
    assert len(projected.exact_mass_by_rank) < spec.class_space_size


def test_boundary_uncertainty_propagates_sparse_exact_class_mass_bounds() -> None:
    spec = _spec()
    exact = _constant_intervals(spec, Fraction(1, 12))
    crossing = list(exact)
    crossing[0] = CertifiedProbabilityInterval(Fraction(1, 7), Fraction(1, 5))
    projected = project_certified_operational_records(
        spec,
        ORACLE_HASH,
        (
            CertifiedOperationalStateRecord("exact", Fraction(1, 3), exact),
            CertifiedOperationalStateRecord(
                "crossing",
                Fraction(2, 3),
                tuple(crossing),
            ),
        ),
    )
    all_zero = (0,) * spec.coordinate_count
    first_one = (1,) + (0,) * (spec.coordinate_count - 1)
    first_two = (2,) + (0,) * (spec.coordinate_count - 1)
    assert projected.boundary_uncertain_mass == Fraction(2, 3)
    assert projected.class_mass_bounds(spec, all_zero) == (Fraction(1, 3), Fraction(1))
    assert projected.class_mass_bounds(spec, first_one) == (Fraction(0), Fraction(2, 3))
    assert projected.class_mass_bounds(spec, first_two) == (Fraction(0), Fraction(0))
    assert len(projected.boundary_uncertain) == 1
    assert len(projected.boundary_uncertain[0].possible_bins_by_coordinate) == 7


def test_fixed_vector_adapter_requires_exact_and_every_occupied_class_registered() -> None:
    spec = _spec()
    low = _constant_intervals(spec, Fraction(1, 12))
    high = _constant_intervals(spec, Fraction(3, 4))
    exact_projection = project_certified_operational_records(
        spec,
        ORACLE_HASH,
        (
            CertifiedOperationalStateRecord("low", Fraction(1, 4), low),
            CertifiedOperationalStateRecord("high", Fraction(3, 4), high),
        ),
    )
    low_id = spec.class_id((0,) * spec.coordinate_count)
    high_id = spec.class_id((4,) * spec.coordinate_count)
    assert exact_projection.exact_registered_vector(spec, (low_id, high_id)) == (
        Fraction(1, 4),
        Fraction(3, 4),
    )
    try:
        exact_projection.exact_registered_vector(spec, (low_id,))
    except RuntimeError:
        pass
    else:
        raise AssertionError("unregistered occupied class must fail closed")

    crossing = list(low)
    crossing[0] = CertifiedProbabilityInterval(Fraction(1, 7), Fraction(1, 5))
    uncertain_projection = project_certified_operational_records(
        spec,
        ORACLE_HASH,
        (CertifiedOperationalStateRecord("crossing", Fraction(1), tuple(crossing)),),
    )
    try:
        uncertain_projection.exact_registered_vector(spec, (low_id, high_id))
    except RuntimeError:
        pass
    else:
        raise AssertionError("boundary uncertainty must fail the fixed-vector adapter")


def test_actual_projector_is_plan_bound_and_guarded_before_result_or_oracle_access() -> None:
    spec = _spec()
    oracle = _OracleStub(spec)
    class_ids = (
        spec.class_id((0,) * spec.coordinate_count),
        spec.class_id((1,) * spec.coordinate_count),
    )
    plan = _PlanStub(
        stable_hash="island-plan",
        product_law_hash="product-law",
        island_count=3,
        operational_estimand_hash=spec.stable_hash,
        class_projector_hash=operational_projector_hash(spec, ORACLE_HASH),
        class_ids=class_ids,
    )
    projector = FullSupportOperationalClassProjector(plan, spec, oracle)
    assert projector.plan_hash == plan.stable_hash
    assert projector.operational_estimand_hash == spec.stable_hash
    assert projector.class_projector_hash == plan.class_projector_hash
    source = inspect.getsource(FullSupportOperationalClassProjector.project)
    assert source.index("if (") < source.index("result.contract.stable_hash")
    assert source.index("if (") < source.index("result.posterior_particles")
    assert "result.targets" not in source
    assert "result.actions" not in source
    assert "equivalence_class" not in source
    assert "normalize" not in source
    try:
        projector.project(_AccessBomb())
    except RuntimeError:
        pass
    else:
        raise AssertionError("projector result-access guard must remain closed")


def test_no_uncertified_cdf_or_cartesian_class_enumeration_is_smuggled_in() -> None:
    module_source = inspect.getsource(implementation)
    projection_source = inspect.getsource(project_certified_operational_records)
    assert "student_t.cdf" not in module_source
    assert "scipy.stats" not in module_source
    assert "nextafter" not in module_source
    assert "nearest" not in projection_source
    assert "itertools.product" not in module_source
    assert "class_space_size" not in projection_source
    assert P3F4_CERT11_CERTIFIED_CDF_INTERVAL_ORACLE_IMPLEMENTATION_AUTHORIZED is False
