import sys

from slai_mi.apps import collect_sim, teleop_sim


def test_sim_dry_runs_do_not_import_isaac(capsys) -> None:
    before = {name for name in sys.modules if name.startswith(("isaaclab", "omni"))}
    assert teleop_sim.main([]) == 0
    assert collect_sim.main([]) == 0
    capsys.readouterr()
    after = {name for name in sys.modules if name.startswith(("isaaclab", "omni"))}
    assert after == before
