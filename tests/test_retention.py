from axiom.benchmarks.retention import run_retention


def test_distribution_shift_probe_keeps_all_conditions_and_state_evidence(tmp_path):
    report = run_retention(tmp_path / "retention", seeds=[0], tasks=3)
    assert len(report["aggregate"]) == 6
    assert len(report["states"]) == 3
    assert all(row["n"] == 3 for row in report["aggregate"].values())
    assert len({row["digest"] for row in report["states"]}) > 1
