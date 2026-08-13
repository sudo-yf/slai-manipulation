import numpy as np

from slai_mi.training import CompleteActionHorizonDataset


def test_complete_action_horizon_filters_episode_tails() -> None:
    class Dataset:
        def __init__(self) -> None:
            self.episode_data_index = {"from": [0, 4], "to": [4, 7]}

        def __getitem__(self, index):
            return {"index": index, "actions_is_pad": np.zeros(3, dtype=bool)}

    filtered = CompleteActionHorizonDataset(Dataset(), 3, ("actions",))
    assert len(filtered) == 3
    assert [filtered[index]["index"] for index in range(3)] == [0, 1, 4]
