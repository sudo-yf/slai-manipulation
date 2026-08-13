from slai_mi.collection.operator_control import EpisodeAction, SpaceMouseEpisodeControls


def test_episode_buttons_are_rising_edge_triggered() -> None:
    controls = SpaceMouseEpisodeControls()
    assert controls.update({0: True}) is EpisodeAction.START
    assert controls.update({0: True}) is None
    assert controls.update({}) is None
    assert controls.update({1: True}) is EpisodeAction.SAVE


def test_escape_discards_an_active_episode() -> None:
    controls = SpaceMouseEpisodeControls()
    assert controls.update({0: True}) is EpisodeAction.START
    assert controls.update({}) is None
    assert controls.update({22: True}) is EpisodeAction.DISCARD
    assert not controls.recording
