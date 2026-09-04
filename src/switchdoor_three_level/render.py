"""Deterministic dependency-free RGB and PNG renderer for 7x7 and 11x11 grids."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping

from .model import Color, Position, WorldSpec, WorldState, validate_state

RGB = tuple[int, int, int]


class Raster:
    def __init__(self, width: int, height: int, background: RGB) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(background * (width * height))

    def rectangle(self, x0: int, y0: int, x1: int, y1: int, color: RGB) -> None:
        left = max(0, min(x0, x1))
        right = min(self.width - 1, max(x0, x1))
        top = max(0, min(y0, y1))
        bottom = min(self.height - 1, max(y0, y1))
        if left > right or top > bottom:
            return
        row_payload = bytes(color) * (right - left + 1)
        for y in range(top, bottom + 1):
            start = (y * self.width + left) * 3
            self.pixels[start : start + len(row_payload)] = row_payload

    def circle(self, center_x: int, center_y: int, radius: int, color: RGB) -> None:
        squared = radius * radius
        for y in range(center_y - radius, center_y + radius + 1):
            if not 0 <= y < self.height:
                continue
            dy = y - center_y
            width = int(max(0, squared - dy * dy) ** 0.5)
            self.rectangle(center_x - width, y, center_x + width, y, color)

    def diamond(self, center_x: int, center_y: int, radius: int, color: RGB) -> None:
        for y in range(center_y - radius, center_y + radius + 1):
            width = radius - abs(y - center_y)
            self.rectangle(center_x - width, y, center_x + width, y, color)

    def frame(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        width: int,
        color: RGB,
    ) -> None:
        self.rectangle(x0, y0, x1, y0 + width - 1, color)
        self.rectangle(x0, y1 - width + 1, x1, y1, color)
        self.rectangle(x0, y0, x0 + width - 1, y1, color)
        self.rectangle(x1 - width + 1, y0, x1, y1, color)


def _palette(render_config: Mapping[str, object]) -> dict[str, RGB]:
    value = render_config["palette"]
    if not isinstance(value, Mapping):
        raise TypeError("render palette must be an object")
    result: dict[str, RGB] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("render palette entry is invalid")
        result[name] = (int(raw[0]), int(raw[1]), int(raw[2]))
    return result


def _cell_box(position: Position, grid_size: int, resolution: int) -> tuple[int, int, int, int]:
    row, col = position
    x0 = col * resolution // grid_size
    x1 = (col + 1) * resolution // grid_size - 1
    y0 = row * resolution // grid_size
    y1 = (row + 1) * resolution // grid_size - 1
    return x0, y0, x1, y1


def _center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    return (x0 + x1) // 2, (y0 + y1) // 2


def render_rgb(
    spec: WorldSpec,
    state: WorldState,
    render_config: Mapping[str, object],
) -> bytes:
    validate_state(spec, state)
    resolution = int(render_config["resolution"])
    palette = _palette(render_config)
    raster = Raster(resolution, resolution, palette["background"])

    for row in range(spec.grid_size):
        for col in range(spec.grid_size):
            position = (row, col)
            x0, y0, x1, y1 = _cell_box(position, spec.grid_size, resolution)
            raster.rectangle(
                x0,
                y0,
                x1,
                y1,
                palette["wall"] if position in spec.walls else palette["floor"],
            )

    goal_box = _cell_box(spec.goal, spec.grid_size, resolution)
    goal_cell = min(goal_box[2] - goal_box[0] + 1, goal_box[3] - goal_box[1] + 1)
    goal_inset = max(4, goal_cell // 4)
    raster.rectangle(
        goal_box[0] + goal_inset,
        goal_box[1] + goal_inset,
        goal_box[2] - goal_inset,
        goal_box[3] - goal_inset,
        palette["goal"],
    )

    for color, position in (
        (Color.RED, spec.red_switch),
        (Color.BLUE, spec.blue_switch),
    ):
        box = _cell_box(position, spec.grid_size, resolution)
        center_x, center_y = _center(box)
        radius = max(5, min(box[2] - box[0] + 1, box[3] - box[1] + 1) // 5)
        raster.circle(center_x, center_y, radius + 2, palette["dark"])
        raster.circle(center_x, center_y, radius, palette[color.value])

    for color, position in (
        (Color.RED, spec.red_door),
        (Color.BLUE, spec.blue_door),
    ):
        box = _cell_box(position, spec.grid_size, resolution)
        cell = min(box[2] - box[0] + 1, box[3] - box[1] + 1)
        inset = max(3, cell // 7)
        door_box = (
            box[0] + inset,
            box[1] + inset,
            box[2] - inset,
            box[3] - inset,
        )
        if state.door_open(color):
            raster.rectangle(*door_box, palette["floor"])
            raster.frame(*door_box, max(2, cell // 16), palette[color.value])
        else:
            raster.rectangle(*door_box, palette[color.value])
            door_width = door_box[2] - door_box[0] + 1
            bar_width = max(1, cell // 18)
            for fraction in (1, 2, 3):
                x = door_box[0] + fraction * door_width // 4
                raster.rectangle(
                    x - bar_width // 2,
                    door_box[1] + 2,
                    x + bar_width // 2,
                    door_box[3] - 2,
                    palette["dark"],
                )

    agent_box = _cell_box(state.agent, spec.grid_size, resolution)
    agent_x, agent_y = _center(agent_box)
    agent_radius = max(
        6,
        min(agent_box[2] - agent_box[0] + 1, agent_box[3] - agent_box[1] + 1) // 5,
    )
    raster.diamond(agent_x, agent_y, agent_radius + 2, palette["dark"])
    raster.diamond(agent_x, agent_y, agent_radius, palette["agent"])

    line_width = max(1, resolution // 448)
    for index in range(spec.grid_size + 1):
        coordinate = min(index * resolution // spec.grid_size, resolution - 1)
        raster.rectangle(
            coordinate,
            0,
            min(resolution - 1, coordinate + line_width - 1),
            resolution - 1,
            palette["grid_line"],
        )
        raster.rectangle(
            0,
            coordinate,
            resolution - 1,
            min(resolution - 1, coordinate + line_width - 1),
            palette["grid_line"],
        )
    return bytes(raster.pixels)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def encode_png(rgb: bytes, resolution: int, compression_level: int) -> bytes:
    if len(rgb) != resolution * resolution * 3:
        raise ValueError("RGB payload has the wrong size")
    stride = resolution * 3
    scanlines = b"".join(
        b"\x00" + rgb[row * stride : (row + 1) * stride] for row in range(resolution)
    )
    header = struct.pack(">IIBBBBB", resolution, resolution, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, compression_level))
        + _png_chunk(b"IEND", b"")
    )


def render_png(
    spec: WorldSpec,
    state: WorldState,
    render_config: Mapping[str, object],
) -> tuple[bytes, bytes]:
    resolution = int(render_config["resolution"])
    rgb = render_rgb(spec, state, render_config)
    png = encode_png(rgb, resolution, int(render_config["png_compression_level"]))
    return rgb, png
