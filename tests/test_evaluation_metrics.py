import numpy as np

from slai_mi.evaluation import action_metrics, select_horizon_anchors


def test_action_metrics_and_episode_anchors() -> None:
    anchors = select_horizon_anchors([0, 0, 0, 1, 1], [0, 1, 2, 0, 1], count=5, horizon_span=1)
    assert anchors.tolist() == [0, 1, 3]
    metrics = action_metrics(np.ones((2, 8)), np.zeros((2, 8)), np.ones(8))
    assert metrics["mae"] == 1.0
    assert metrics["arm_mae"] == 1.0
    assert metrics["hand_mae"] == 1.0
