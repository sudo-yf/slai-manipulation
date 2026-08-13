from slai_mi import __version__
from slai_mi.apps import collect_real, collect_sim, teleop_real, teleop_sim
from slai_mi.datasets import lerobot_v3
from slai_mi.devices import (
    cameras,
    iphone,
    spacemouse,
    stereo_camera,
    ur5,
    wrist_sensor,
    wujihand,
)
from slai_mi.ui import collection_frontend


def test_package_version():
    assert __version__ == "0.1.0"


def test_interface_modules_import():
    assert cameras.__doc__
    assert stereo_camera.__doc__
    assert iphone.__doc__
    assert spacemouse.__doc__
    assert ur5.__doc__
    assert wrist_sensor.__doc__
    assert wujihand.__doc__


def test_app_modules_import():
    assert teleop_real.__doc__
    assert teleop_sim.__doc__
    assert collect_real.__doc__
    assert collect_sim.__doc__
    assert collection_frontend.__doc__
    assert lerobot_v3.__doc__
