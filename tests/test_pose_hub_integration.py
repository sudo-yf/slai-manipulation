import json
import socket
import threading

from slai_mi.apps import pose_hub, pose_hub_bridge
from slai_mi.devices.iphone import PoseHubBridge, RobotStateHandoff


def test_robot_state_handoff_emits_valid_packet() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    received = {}

    def accept() -> None:
        connection, _ = listener.accept()
        with connection:
            received.update(json.loads(connection.makefile("rb").readline()))

    thread = threading.Thread(target=accept)
    thread.start()
    try:
        with RobotStateHandoff(port=listener.getsockname()[1]) as handoff:
            handoff.publish(
                ["shoulder_pan_joint", "wrist_fe_joint"], [0.1, -0.2], timestamp_s=123.0
            )
    finally:
        thread.join(timeout=2)
        listener.close()

    assert received == {
        "type": "robot_state",
        "timestamp_s": 123.0,
        "joint_names": ["shoulder_pan_joint", "wrist_fe_joint"],
        "joint_positions_rad": [0.1, -0.2],
    }


def test_pose_hub_bridge_builds_authenticated_stream_path() -> None:
    bridge = PoseHubBridge("https://6d.leai.me", "session id", "token", 5005, 5006, 0.25)
    assert bridge.websocket_url() == "wss://6d.leai.me/ws/bridge/session%20id"


def test_pose_hub_bridge_cli_defaults_to_dry_run(capsys) -> None:
    assert pose_hub_bridge.main(["--session", "example"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["app"] == "pose_hub_bridge"
    assert plan["mode"] == "dry-run"
    assert plan["pose_port"] == 5005
    assert plan["robot_state_port"] == 5006


def test_pose_hub_cli_defaults_to_dry_run(capsys) -> None:
    assert pose_hub.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan == {
        "app": "pose_hub",
        "mode": "dry-run",
        "host": "127.0.0.1",
        "port": 8706,
    }
