from __future__ import annotations

from switchdoor import (
    ActionFamily,
    GenerationConfig,
    LayoutFamily,
    build_scenario_group,
    generate_scenario_groups,
)


def test_seeded_generator_matches_registered_golden_root() -> None:
    group = build_scenario_group(GenerationConfig(seed=7), candidate_index=0)
    assert (
        group.shell.layout_sha256
        == "ab996b594d39fabe9fc2b9d3819982ff0500c412a20ba2055b2220f2d5e32e15"
    )
    assert (
        group.shell.shell_sha256
        == "6b4cbbb8ad0273aa301d868966307972c7f5de98cca0d2d651c7812b9face9af"
    )
    assert [action.value for action in group.shell.actions] == [
        "INTERACT",
        "MOVE_LEFT",
        "MOVE_UP",
        "INTERACT",
        "INTERACT",
        "INTERACT",
        "MOVE_LEFT",
    ]


def test_group_is_counterfactually_paired_without_endpoint_shortcut() -> None:
    group = build_scenario_group(GenerationConfig(seed=19), candidate_index=3)
    assert len({episode.spec for episode in group.episodes}) == 1
    assert len({episode.actions for episode in group.episodes}) == 1
    assert len({episode.states for episode in group.episodes}) == 4
    assert len({episode.states[-1] for episode in group.episodes}) == 1
    assert group.episodes[0].states[-1].red_door_open
    assert group.episodes[0].states[-1].blue_door_open


def test_generation_is_deterministic_and_unique() -> None:
    config = GenerationConfig(seed=101, id_seed=202)
    first = generate_scenario_groups(config, 12)
    second = generate_scenario_groups(config, 12)
    assert first == second
    assert len({group.shell.layout_sha256 for group in first}) == 12
    assert len({group.shell.root_family_id for group in first}) == 12
    other = generate_scenario_groups(GenerationConfig(seed=102, id_seed=202), 1)
    assert other[0].shell.layout_sha256 != first[0].shell.layout_sha256


def test_far_and_five_families_keep_registered_history_contract() -> None:
    config = GenerationConfig(
        seed=5,
        layout_family=LayoutFamily.FAR,
        action_family=ActionFamily.FIVE,
    )
    groups = generate_scenario_groups(config, 8)
    assert all(group.shell.layout_family is LayoutFamily.FAR for group in groups)
    assert all(group.shell.action_family is ActionFamily.FIVE for group in groups)
    assert all(8 <= len(group.shell.actions) <= 10 for group in groups)
