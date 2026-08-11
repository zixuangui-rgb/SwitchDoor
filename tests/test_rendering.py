from __future__ import annotations

import hashlib

import pytest

from switchdoor import GenerationConfig, build_scenario_group
from switchdoor.rendering import (
    changed_pixel_positions,
    png_dimensions,
    positions_within_cell,
    render_png,
    visibility_report,
)


def test_render_is_deterministic_and_matches_golden_png() -> None:
    group = build_scenario_group(GenerationConfig(seed=7), candidate_index=0)
    spec = group.shell.spec
    state = group.episodes[0].states[0]
    rgb, png = render_png(spec, state, renderer="salient", resolution=448)
    assert png_dimensions(png) == (448, 448)
    assert (
        hashlib.sha256(png).hexdigest()
        == "06c83c2f97e87f776d8b3a49357ae9c6a3742a72c7e1f365a92d0a03713766e4"
    )
    assert render_png(spec, state, renderer="salient", resolution=448) == (
        rgb,
        png,
    )
    assert visibility_report(spec, state, rgb, renderer="salient", resolution=448)[
        "pass"
    ]


@pytest.mark.parametrize("renderer", ["legacy", "salient"])
@pytest.mark.parametrize("resolution", [448, 896])
def test_all_registered_render_contracts_are_visually_auditable(
    renderer: str,
    resolution: int,
) -> None:
    group = build_scenario_group(GenerationConfig(seed=8), candidate_index=1)
    spec = group.shell.spec
    for state in (group.episodes[0].states[0], group.episodes[0].states[-1]):
        rgb, png = render_png(
            spec,
            state,
            renderer=renderer,
            resolution=resolution,
        )
        assert png_dimensions(png) == (resolution, resolution)
        assert visibility_report(
            spec,
            state,
            rgb,
            renderer=renderer,
            resolution=resolution,
        )["pass"]


def test_door_change_is_localized_to_the_registered_door_cell() -> None:
    group = build_scenario_group(GenerationConfig(seed=7), candidate_index=0)
    episode = group.episodes[0]
    before = render_png(
        episode.spec, episode.states[0], renderer="salient", resolution=448
    )[0]
    after = render_png(
        episode.spec, episode.states[1], renderer="salient", resolution=448
    )[0]
    changed = changed_pixel_positions(before, after, resolution=448)
    assert changed
    assert positions_within_cell(
        changed,
        cell=episode.spec.door_map[
            episode.rule.target_door(group.shell.first_switch_color)
        ],
        resolution=448,
    )
