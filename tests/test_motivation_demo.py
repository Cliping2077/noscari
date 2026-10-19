from axiom.examples.motivation import run_demo


def test_motivation_demo_has_real_change_and_persistent_goal(tmp_path, monkeypatch):
    import socket
    def no_network(*args, **kwargs):
        raise AssertionError("motivation demo must stay offline")
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "socket", no_network)
    result = run_demo(tmp_path / "demo")
    assert result["initial_choice"] == "alpha"
    assert result["trained_choice"] == "beta"
    assert result["recording_unchanged"]
    assert result["restored_same_parameters"] and result["restored_same_goals"]
    assert result["pause_blocks_suggestions"]
    assert result["learning"]["parameter_delta"] > 0
