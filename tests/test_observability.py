from pathlib import Path

from slai_mi.observability.session import classify_incident


def test_classifies_ur5_connection_failure(tmp_path: Path):
    (tmp_path / "ur5.log").write_text("RTDE receive did not connect", encoding="utf-8")

    incident = classify_incident(tmp_path, reason="ur5_failed", exit_code=1)

    assert incident is not None
    assert incident["code"] == "ur5_unreachable"


def test_success_has_no_incident(tmp_path: Path):
    assert classify_incident(tmp_path, reason="completed", exit_code=0) is None
