"""Write staged PI0.5 episodes with the intentionally separate LeRobot v2.1 runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_dataset(staging: Path, root: Path, repo_id: str, *, fps: int = 15) -> None:
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"PI0.5 writer requires LeRobot v2.1, found {CODEBASE_VERSION}")
    if root.exists():
        raise FileExistsError(f"refusing to overwrite PI0.5 dataset: {root}")
    episodes = sorted(staging.glob("episode-*.npz"))
    if not episodes:
        raise ValueError(f"no staged episodes found in {staging}")
    image = {"dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channel"]}
    features = {
        "primary_rgb": dict(image),
        "secondary_rgb": dict(image),
        "state": {"dtype": "float32", "shape": (32,), "names": ["pi05_state"]},
        "actions": {"dtype": "float32", "shape": (26,), "names": ["pi05_actions"]},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="ur5_wujihand_pi05",
        fps=fps,
        features=features,
        use_videos=True,
        tolerance_s=1e-6,
        image_writer_threads=4,
    )
    try:
        for path in episodes:
            with np.load(path, allow_pickle=False) as episode:
                arrays = [episode[name] for name in ("primary_rgb", "secondary_rgb", "state", "actions")]
                if len({len(value) for value in arrays}) != 1 or not len(arrays[0]):
                    raise ValueError(f"inconsistent or empty episode: {path}")
                for index in range(len(arrays[0])):
                    dataset.add_frame(
                        {
                            "primary_rgb": np.ascontiguousarray(arrays[0][index]),
                            "secondary_rgb": np.ascontiguousarray(arrays[1][index]),
                            "state": np.ascontiguousarray(arrays[2][index], dtype=np.float32),
                            "actions": np.ascontiguousarray(arrays[3][index], dtype=np.float32),
                            "task": str(episode["task"].item()),
                        }
                    )
            dataset.save_episode()
    finally:
        dataset.stop_image_writer()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    write_dataset(args.staging.resolve(), args.root.resolve(), args.repo_id)


if __name__ == "__main__":
    main()
