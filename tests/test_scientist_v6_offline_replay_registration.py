"""Static registration-builder identity tests."""

from scripts.build_scientist_v6_offline_replay_registration import (
    SPECS, _artifact_names,
)


def test_v6_replay_registration_covers_three_fixed_families():
    assert [row[0] for row in SPECS] == ["yacht", "airfoil", "energy"]
    for _, _, _, coordinate in SPECS:
        names = _artifact_names(coordinate)
        assert len(names) == 13
        assert len(set(names)) == 13
