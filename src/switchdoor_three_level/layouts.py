"""Registered templates and seeded D4 spatial randomization."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .model import WorldSpec

TRANSFORMS: tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "mirror",
    "mirror_rot90",
    "mirror_rot180",
    "mirror_rot270",
)


@dataclass(frozen=True, slots=True)
class MaterializedLevel:
    level_id: str
    purpose: str
    oracle_policy: str
    transform: str
    rows: tuple[str, ...]
    spec: WorldSpec


def transform_position(
    position: tuple[int, int],
    size: int,
    transform: str,
) -> tuple[int, int]:
    if transform not in TRANSFORMS:
        raise ValueError(f"unknown D4 transform: {transform}")
    row, col = position
    reflected = transform.startswith("mirror")
    if reflected:
        col = size - 1 - col
    rotation_name = transform.removeprefix("mirror_") if reflected else transform
    rotations = {
        "identity": 0,
        "mirror": 0,
        "rot90": 1,
        "rot180": 2,
        "rot270": 3,
    }[rotation_name]
    for _ in range(rotations):
        row, col = col, size - 1 - row
    return row, col


def transform_rows(rows: list[str] | tuple[str, ...], transform: str) -> tuple[str, ...]:
    size = len(rows)
    if size == 0 or any(len(row) != size for row in rows):
        raise ValueError("template must be square")
    result = [["" for _ in range(size)] for _ in range(size)]
    for row_index, row in enumerate(rows):
        for col_index, token in enumerate(row):
            target_row, target_col = transform_position(
                (row_index, col_index),
                size,
                transform,
            )
            result[target_row][target_col] = token
    return tuple("".join(row) for row in result)


def materialize_level(level_config: dict[str, Any], transform: str) -> MaterializedLevel:
    rows = transform_rows(level_config["template"], transform)
    spec = WorldSpec.from_grid(list(rows))
    if spec.grid_size != level_config["grid_size"]:
        raise ValueError("materialized grid size differs from level config")
    return MaterializedLevel(
        level_id=level_config["level_id"],
        purpose=level_config["purpose"],
        oracle_policy=level_config["oracle_policy"],
        transform=transform,
        rows=rows,
        spec=spec,
    )


def materialize_levels(config: dict[str, Any], root_seed: int) -> tuple[MaterializedLevel, ...]:
    rng = random.Random(root_seed)
    return tuple(
        materialize_level(level_config, rng.choice(TRANSFORMS)) for level_config in config["levels"]
    )
