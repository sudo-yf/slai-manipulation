"""Zero-episode collection roots are discarded during finalization."""

from __future__ import annotations

from types import SimpleNamespace

from slai_mi.datasets.lerobot_v3.configured import ConfiguredDatasetWriter
from slai_mi.datasets.lerobot_v3.writer import ContractDatasetWriter


class _EmptyDataset:
    def __init__(self) -> None:
        self.meta = SimpleNamespace(total_episodes=0)
        self.finalized = False

    def finalize(self) -> None:
        self.finalized = True


class _Contract:
    def write_manifest(self, root) -> None:
        (root / "contract.json").write_text("{}", encoding="utf-8")


def test_contract_writer_removes_zero_episode_root(tmp_path) -> None:
    root = tmp_path / "empty-contract"
    root.mkdir()
    dataset = _EmptyDataset()
    writer = ContractDatasetWriter(dataset, root)

    writer.finalize()

    assert dataset.finalized is True
    assert not root.exists()


def test_configured_writer_removes_zero_episode_root(tmp_path) -> None:
    root = tmp_path / "empty-configured"
    root.mkdir()
    dataset = _EmptyDataset()
    writer = ConfiguredDatasetWriter(dataset, root, _Contract())

    writer.finalize()

    assert dataset.finalized is True
    assert not root.exists()
