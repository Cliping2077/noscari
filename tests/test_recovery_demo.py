from axiom.examples.recovery import run_demo


def test_recovery_demo_separates_recording_and_learning(tmp_path):
    result = run_demo(tmp_path / "demo")
    assert result["initial_choice"] == "traverse"
    assert result["recorded_choice"] == "traverse"
    assert result["trained_choice"] == "ask"
    assert result["initial_digest"] == result["recorded_digest"]
    assert result["initial_digest"] != result["trained_digest"]
    assert result["rollback_digest"] == result["initial_digest"]
