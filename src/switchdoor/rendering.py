"""Dependency-free RGB renderer and pixel audits for SwitchDoor.

The renderer accepts only :mod:`switchdoor.core` objects. Both supported
profiles use the same object semantics:

* switches are drawn before the smaller agent diamond, leaving a color halo;
* an open door has a light center and a colored frame;
* a closed door is color-filled with dark bars.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import zlib
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .core import (
    GRID_SIZE,
    Color,
    Position,
    WorldSpec,
    WorldState,
    validate_state_for_spec,
)

RGB = tuple[int, int, int]
SUPPORTED_RESOLUTIONS = (448, 896)
SUPPORTED_PROFILES = ("legacy", "salient")


DEFAULT_RENDERER_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "grid_size": GRID_SIZE,
    "base_resolution": 448,
    "resolutions": [448, 896],
    "native_symbolic_rerender": True,
    "png_compression_level": 9,
    "text_overlay": False,
    "palette": {
        "floor": [246, 246, 240],
        "grid_line": [188, 190, 194],
        "wall": [55, 60, 70],
        "goal": [72, 190, 92],
        "red": [220, 55, 62],
        "blue": [48, 105, 220],
        "agent": [246, 196, 55],
        "dark": [25, 25, 28],
        "legacy_open_center": [250, 250, 247],
        "background": [235, 236, 232],
    },
    "shared_geometry_at_448": {
        "line_width": 2,
        "switch_outline_radius": 28,
        "switch_color_radius": 25,
        "agent_radius": 16,
        "agent_outline_width": 4,
        "goal_inset": 15,
    },
    "profiles": {
        "legacy": {
            "renderer_id": "switchdoor-legacy-native-v1",
            "door_inset": 7,
            "open_border_width": 7,
            "open_center_color": "legacy_open_center",
            "closed_bar_width": 5,
            "closed_bar_offsets": [18, 30, 42],
        },
        "salient": {
            "renderer_id": "switchdoor-salient-native-v1",
            "door_inset": 3,
            "open_border_width": 10,
            "open_center_color": "floor",
            "closed_bar_width": 7,
            "closed_bar_offsets": [15, 29, 43],
        },
    },
}


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _plain_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _rgb(value: Any, *, label: str) -> RGB:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(channel, bool) or not isinstance(channel, int)
            for channel in value
        )
        or any(not 0 <= channel <= 255 for channel in value)
    ):
        raise ValueError(f"{label} must contain exactly three RGB bytes")
    return value[0], value[1], value[2]


def validate_renderer_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "grid_size",
        "base_resolution",
        "resolutions",
        "native_symbolic_rerender",
        "png_compression_level",
        "text_overlay",
        "palette",
        "shared_geometry_at_448",
        "profiles",
    }
    if set(value) != expected:
        raise ValueError("renderer config top-level fields differ")
    if (
        value["schema_version"] != 1
        or value["grid_size"] != GRID_SIZE
        or value["base_resolution"] != 448
        or value["resolutions"] != list(SUPPORTED_RESOLUTIONS)
        or value["native_symbolic_rerender"] is not True
        or value["text_overlay"] is not False
    ):
        raise ValueError("renderer config top-level contract differs")
    compression = _plain_int(
        value["png_compression_level"],
        label="png_compression_level",
    )
    if not 0 <= compression <= 9:
        raise ValueError("png_compression_level must lie in [0, 9]")

    palette = value["palette"]
    expected_palette = {
        "floor",
        "grid_line",
        "wall",
        "goal",
        "red",
        "blue",
        "agent",
        "dark",
        "legacy_open_center",
        "background",
    }
    if not isinstance(palette, Mapping) or set(palette) != expected_palette:
        raise ValueError("renderer palette fields differ")
    parsed_palette = {
        name: _rgb(color, label=f"palette.{name}") for name, color in palette.items()
    }
    if len(set(parsed_palette.values())) != len(parsed_palette):
        raise ValueError("renderer palette colors must be pairwise distinct")

    shared = value["shared_geometry_at_448"]
    expected_shared = {
        "line_width",
        "switch_outline_radius",
        "switch_color_radius",
        "agent_radius",
        "agent_outline_width",
        "goal_inset",
    }
    if not isinstance(shared, Mapping) or set(shared) != expected_shared:
        raise ValueError("shared renderer geometry fields differ")
    parsed_shared = {
        name: _plain_int(item, label=f"shared_geometry_at_448.{name}")
        for name, item in shared.items()
    }
    if any(item <= 0 for item in parsed_shared.values()):
        raise ValueError("shared renderer geometry must be positive")
    if not (
        parsed_shared["switch_outline_radius"]
        > parsed_shared["switch_color_radius"]
        > parsed_shared["agent_radius"]
        > parsed_shared["agent_outline_width"]
    ):
        raise ValueError(
            "switch and agent radii do not guarantee a visible switch halo"
        )

    profiles = value["profiles"]
    if not isinstance(profiles, Mapping) or set(profiles) != set(SUPPORTED_PROFILES):
        raise ValueError("renderer profiles must be legacy and salient")
    expected_profile = {
        "renderer_id",
        "door_inset",
        "open_border_width",
        "open_center_color",
        "closed_bar_width",
        "closed_bar_offsets",
    }
    for name, profile in profiles.items():
        if not isinstance(profile, Mapping) or set(profile) != expected_profile:
            raise ValueError(f"renderer profile {name} fields differ")
        if not isinstance(profile["renderer_id"], str) or not profile["renderer_id"]:
            raise ValueError(f"renderer profile {name} has invalid renderer_id")
        if profile["open_center_color"] not in palette:
            raise ValueError(f"renderer profile {name} has unknown open center color")
        offsets = profile["closed_bar_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) < 1
            or any(
                _plain_int(
                    offset,
                    label=f"profiles.{name}.closed_bar_offsets",
                )
                <= 0
                for offset in offsets
            )
        ):
            raise ValueError(f"renderer profile {name} closed bar offsets are invalid")
        for field in (
            "door_inset",
            "open_border_width",
            "closed_bar_width",
        ):
            if (
                _plain_int(
                    profile[field],
                    label=f"profiles.{name}.{field}",
                )
                <= 0
            ):
                raise ValueError(f"renderer profile {name} geometry must be positive")
    if profiles["salient"]["open_center_color"] != "floor":
        raise ValueError("salient open door center must use exact floor color")


def default_renderer_config() -> dict[str, Any]:
    """Return a mutable copy of the built-in, validated configuration."""

    copied = json.loads(json.dumps(DEFAULT_RENDERER_CONFIG))
    validate_renderer_config(copied)
    return copied


def load_renderer_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a strict renderer JSON, or return the built-in configuration."""

    if path is None:
        return default_renderer_config()
    config_path = Path(path)
    try:
        value = json.loads(
            config_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load renderer config: {config_path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("renderer config must be a JSON object")
    result = dict(value)
    validate_renderer_config(result)
    return result


def _active_config(
    config: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    active = DEFAULT_RENDERER_CONFIG if config is None else config
    validate_renderer_config(active)
    return active


class Raster:
    def __init__(self, width: int, height: int, background: RGB) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("raster dimensions must be positive")
        self.width = width
        self.height = height
        self.pixels = bytearray(background * (width * height))

    def set(self, x: int, y: int, color: RGB) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        offset = (y * self.width + x) * 3
        self.pixels[offset : offset + 3] = bytes(color)

    def get(self, x: int, y: int) -> RGB:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("pixel coordinate is outside raster")
        offset = (y * self.width + x) * 3
        red, green, blue = self.pixels[offset : offset + 3]
        return red, green, blue

    def rectangle(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: RGB,
    ) -> None:
        left = max(0, min(x0, x1))
        right = min(self.width - 1, max(x0, x1))
        top = max(0, min(y0, y1))
        bottom = min(self.height - 1, max(y0, y1))
        if left > right or top > bottom:
            return
        row_payload = bytes(color) * (right - left + 1)
        for y in range(top, bottom + 1):
            offset = (y * self.width + left) * 3
            self.pixels[offset : offset + len(row_payload)] = row_payload

    def frame(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        width: int,
        color: RGB,
    ) -> None:
        if width <= 0:
            raise ValueError("frame width must be positive")
        self.rectangle(x0, y0, x1, y0 + width - 1, color)
        self.rectangle(x0, y1 - width + 1, x1, y1, color)
        self.rectangle(x0, y0, x0 + width - 1, y1, color)
        self.rectangle(x1 - width + 1, y0, x1, y1, color)

    def circle(self, cx: int, cy: int, radius: int, color: RGB) -> None:
        if radius <= 0:
            raise ValueError("circle radius must be positive")
        radius_squared = radius * radius
        for y in range(cy - radius, cy + radius + 1):
            y_squared = (y - cy) * (y - cy)
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) * (x - cx) + y_squared <= radius_squared:
                    self.set(x, y, color)

    def diamond(
        self,
        cx: int,
        cy: int,
        radius: int,
        outline: RGB,
        fill: RGB,
        outline_width: int,
    ) -> None:
        if not 0 < outline_width < radius:
            raise ValueError("diamond outline_width must be smaller than radius")
        for y_offset in range(-radius, radius + 1):
            extent = radius - abs(y_offset)
            for x_offset in range(-extent, extent + 1):
                self.set(cx + x_offset, cy + y_offset, outline)
        inner_radius = radius - outline_width
        for y_offset in range(-inner_radius, inner_radius + 1):
            extent = inner_radius - abs(y_offset)
            for x_offset in range(-extent, extent + 1):
                self.set(cx + x_offset, cy + y_offset, fill)

    def paste(
        self,
        rgb: bytes,
        width: int,
        height: int,
        x0: int,
        y0: int,
    ) -> None:
        if len(rgb) != width * height * 3:
            raise ValueError("pasted RGB payload size differs")
        if x0 < 0 or y0 < 0 or x0 + width > self.width or y0 + height > self.height:
            raise ValueError("pasted image does not fit inside raster")
        stride = width * 3
        for row in range(height):
            source = row * stride
            destination = ((y0 + row) * self.width + x0) * 3
            self.pixels[destination : destination + stride] = rgb[
                source : source + stride
            ]


def _scale(resolution: int) -> int:
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {list(SUPPORTED_RESOLUTIONS)}")
    return resolution // 448


def cell_size(resolution: int) -> int:
    _scale(resolution)
    if resolution % GRID_SIZE:
        raise ValueError("resolution must be divisible by grid size")
    return resolution // GRID_SIZE


def cell_box(position: Position, resolution: int) -> tuple[int, int, int, int]:
    if not isinstance(position, Position):
        raise TypeError("position must be a Position")
    size = cell_size(resolution)
    x0 = position.col * size
    y0 = position.row * size
    return x0, y0, x0 + size - 1, y0 + size - 1


def cell_center(position: Position, resolution: int) -> tuple[int, int]:
    size = cell_size(resolution)
    return (
        position.col * size + size // 2,
        position.row * size + size // 2,
    )


def _active_values(
    renderer: str,
    resolution: int,
    config: Mapping[str, Any] | None,
) -> tuple[dict[str, RGB], Mapping[str, Any], Mapping[str, Any], int]:
    active = _active_config(config)
    scale = _scale(resolution)
    if renderer not in SUPPORTED_PROFILES:
        raise ValueError(f"unknown renderer profile: {renderer!r}")
    palette_value = active["palette"]
    profiles_value = active["profiles"]
    shared_value = active["shared_geometry_at_448"]
    if (
        not isinstance(palette_value, Mapping)
        or not isinstance(profiles_value, Mapping)
        or not isinstance(shared_value, Mapping)
    ):
        raise ValueError("validated renderer config has invalid nested values")
    palette = {
        name: _rgb(value, label=f"palette.{name}")
        for name, value in palette_value.items()
    }
    selected = profiles_value[renderer]
    if not isinstance(selected, Mapping):
        raise ValueError("validated renderer profile is invalid")
    return palette, shared_value, selected, scale


def render_rgb(
    spec: WorldSpec,
    state: WorldState,
    *,
    renderer: str,
    resolution: int = 448,
    config: Mapping[str, Any] | None = None,
) -> bytes:
    """Render one native-resolution RGB frame from a symbolic state."""

    if not isinstance(spec, WorldSpec) or not isinstance(state, WorldState):
        raise TypeError("render_rgb requires WorldSpec and WorldState")
    validate_state_for_spec(spec, state)
    palette, shared, selected, scale = _active_values(
        renderer,
        resolution,
        config,
    )
    size = cell_size(resolution)
    raster = Raster(resolution, resolution, palette["background"])

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            position = Position(row, col)
            x0, y0, x1, y1 = cell_box(position, resolution)
            raster.rectangle(x0, y0, x1, y1, palette["floor"])
            if position in spec.walls:
                raster.rectangle(x0, y0, x1, y1, palette["wall"])

    goal_x0, goal_y0, goal_x1, goal_y1 = cell_box(spec.goal, resolution)
    goal_inset = int(shared["goal_inset"]) * scale
    raster.rectangle(
        goal_x0 + goal_inset,
        goal_y0 + goal_inset,
        goal_x1 - goal_inset,
        goal_y1 - goal_inset,
        palette["goal"],
    )

    # Draw switches before the smaller agent diamond.  The remaining annulus is
    # the explicit visual evidence that fixes the prior occlusion bug.
    for color in (Color.RED, Color.BLUE):
        position = spec.switch_map[color]
        center_x, center_y = cell_center(position, resolution)
        raster.circle(
            center_x,
            center_y,
            int(shared["switch_outline_radius"]) * scale,
            palette["dark"],
        )
        raster.circle(
            center_x,
            center_y,
            int(shared["switch_color_radius"]) * scale,
            palette[color.value],
        )

    for color in (Color.RED, Color.BLUE):
        position = spec.door_map[color]
        is_open = state.door_open(color)
        x0, y0, x1, y1 = cell_box(position, resolution)
        inset = int(selected["door_inset"]) * scale
        if is_open:
            center_color_name = selected["open_center_color"]
            if not isinstance(center_color_name, str):
                raise ValueError("open center color name must be text")
            raster.rectangle(
                x0 + inset,
                y0 + inset,
                x1 - inset,
                y1 - inset,
                palette[center_color_name],
            )
            raster.frame(
                x0 + inset,
                y0 + inset,
                x1 - inset,
                y1 - inset,
                int(selected["open_border_width"]) * scale,
                palette[color.value],
            )
        else:
            raster.rectangle(
                x0 + inset,
                y0 + inset,
                x1 - inset,
                y1 - inset,
                palette[color.value],
            )
            bar_width = int(selected["closed_bar_width"]) * scale
            offsets = selected["closed_bar_offsets"]
            if not isinstance(offsets, list):
                raise ValueError("closed bar offsets must be a list")
            for offset in offsets:
                left = x0 + int(offset) * scale
                raster.rectangle(
                    left,
                    y0 + inset + scale,
                    left + bar_width - 1,
                    y1 - inset - scale,
                    palette["dark"],
                )

    agent_x, agent_y = cell_center(state.agent, resolution)
    raster.diamond(
        agent_x,
        agent_y,
        int(shared["agent_radius"]) * scale,
        palette["dark"],
        palette["agent"],
        int(shared["agent_outline_width"]) * scale,
    )

    line_width = int(shared["line_width"]) * scale
    for index in range(GRID_SIZE + 1):
        coordinate = min(index * size, resolution - 1)
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
    if len(kind) != 4:
        raise ValueError("PNG chunk type must contain four bytes")
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )


def encode_png_rgb(
    rgb: bytes,
    resolution: int,
    *,
    compression_level: int = 9,
) -> bytes:
    """Encode one square frame as deterministic 8-bit RGB/filter-0 PNG."""

    resolution = _plain_int(resolution, label="PNG resolution")
    compression_level = _plain_int(
        compression_level,
        label="PNG compression_level",
    )
    _scale(resolution)
    if len(rgb) != resolution * resolution * 3:
        raise ValueError("RGB payload size differs from PNG resolution")
    if not 0 <= compression_level <= 9:
        raise ValueError("PNG compression_level must lie in [0, 9]")
    stride = resolution * 3
    raw = b"".join(
        b"\x00" + rgb[row * stride : (row + 1) * stride] for row in range(resolution)
    )
    header = struct.pack(
        ">IIBBBBB",
        resolution,
        resolution,
        8,
        2,
        0,
        0,
        0,
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, compression_level))
        + _png_chunk(b"IEND", b"")
    )


def render_png(
    spec: WorldSpec,
    state: WorldState,
    *,
    renderer: str,
    resolution: int = 448,
    config: Mapping[str, Any] | None = None,
) -> tuple[bytes, bytes]:
    """Return ``(raw_rgb, deterministic_png)`` for one state."""

    active = _active_config(config)
    rgb = render_rgb(
        spec,
        state,
        renderer=renderer,
        resolution=resolution,
        config=active,
    )
    png = encode_png_rgb(
        rgb,
        resolution,
        compression_level=int(active["png_compression_level"]),
    )
    return rgb, png


def write_rendered_png(
    path: str | Path,
    spec: WorldSpec,
    state: WorldState,
    *,
    renderer: str,
    resolution: int = 448,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write a frame and return its auditable identity."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgb, png = render_png(
        spec,
        state,
        renderer=renderer,
        resolution=resolution,
        config=config,
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(png)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return {
        "width": resolution,
        "height": resolution,
        "renderer": renderer,
        "pixel_sha256": hashlib.sha256(rgb).hexdigest(),
        "png_sha256": hashlib.sha256(png).hexdigest(),
        "native_symbolic_render": True,
    }


def png_chunks(payload: bytes) -> tuple[str, ...]:
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("payload is not a PNG")
    cursor = 8
    chunks: list[str] = []
    while cursor < len(payload):
        if cursor + 12 > len(payload):
            raise ValueError("PNG contains a truncated chunk")
        length = struct.unpack(">I", payload[cursor : cursor + 4])[0]
        kind = payload[cursor + 4 : cursor + 8]
        end = cursor + 12 + length
        if end > len(payload):
            raise ValueError("PNG contains a truncated chunk payload")
        try:
            chunks.append(kind.decode("ascii"))
        except UnicodeDecodeError as exc:
            raise ValueError("PNG chunk name is not ASCII") from exc
        cursor = end
        if kind == b"IEND":
            break
    if cursor != len(payload) or chunks[:1] != ["IHDR"] or chunks[-1:] != ["IEND"]:
        raise ValueError("PNG chunk structure differs")
    return tuple(chunks)


def png_dimensions(payload: bytes) -> tuple[int, int]:
    if png_chunks(payload)[:1] != ("IHDR",):
        raise ValueError("PNG is missing IHDR")
    return struct.unpack(">II", payload[16:24])


def decode_own_png_rgb(payload: bytes) -> tuple[bytes, int, int]:
    """Decode only the narrow PNG format emitted by :func:`encode_png_rgb`."""

    if png_chunks(payload) != ("IHDR", "IDAT", "IEND"):
        raise ValueError("PNG chunk sequence differs from the renderer contract")
    width, height = png_dimensions(payload)
    cursor = 8
    compressed_parts: list[bytes] = []
    png_fields: tuple[int, int, int, int, int] | None = None
    while cursor < len(payload):
        length = struct.unpack(">I", payload[cursor : cursor + 4])[0]
        kind = payload[cursor + 4 : cursor + 8]
        content = payload[cursor + 8 : cursor + 8 + length]
        checksum = struct.unpack(
            ">I",
            payload[cursor + 8 + length : cursor + 12 + length],
        )[0]
        expected = zlib.crc32(kind)
        expected = zlib.crc32(content, expected) & 0xFFFFFFFF
        if checksum != expected:
            raise ValueError("PNG chunk CRC differs")
        if kind == b"IHDR":
            (
                parsed_width,
                parsed_height,
                bit_depth,
                color_type,
                compression,
                filtering,
                interlace,
            ) = struct.unpack(">IIBBBBB", content)
            if (parsed_width, parsed_height) != (width, height):
                raise ValueError("PNG IHDR dimensions disagree")
            png_fields = (
                bit_depth,
                color_type,
                compression,
                filtering,
                interlace,
            )
        elif kind == b"IDAT":
            compressed_parts.append(content)
        cursor += 12 + length
    if png_fields != (8, 2, 0, 0, 0) or len(compressed_parts) != 1:
        raise ValueError("PNG encoding differs from RGB/filter-0 contract")
    raw = zlib.decompress(b"".join(compressed_parts))
    stride = width * 3
    if len(raw) != height * (stride + 1):
        raise ValueError("decompressed PNG size differs")
    rows: list[bytes] = []
    for row in range(height):
        start = row * (stride + 1)
        if raw[start] != 0:
            raise ValueError("renderer PNG must use filter type zero")
        rows.append(raw[start + 1 : start + stride + 1])
    return b"".join(rows), width, height


def exact_color_count_in_cell(
    rgb: bytes,
    *,
    position: Position,
    resolution: int,
    color: RGB,
) -> int:
    if len(rgb) != resolution * resolution * 3:
        raise ValueError("RGB payload size differs from resolution")
    expected = bytes(_rgb(color, label="counted color"))
    x0, y0, x1, y1 = cell_box(position, resolution)
    count = 0
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            offset = (y * resolution + x) * 3
            count += int(rgb[offset : offset + 3] == expected)
    return count


def door_region_report(
    rgb: bytes,
    *,
    position: Position,
    color: Color | str,
    renderer: str,
    resolution: int = 448,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recover open/closed from a robust center region, not one pixel."""

    if len(rgb) != resolution * resolution * 3:
        raise ValueError("RGB payload size differs from resolution")
    try:
        active_color = Color(color)
    except (TypeError, ValueError) as exc:
        raise ValueError("door color must be red or blue") from exc
    palette, _, selected, scale = _active_values(
        renderer,
        resolution,
        config,
    )
    center_x, center_y = cell_center(position, resolution)
    radius = cell_size(resolution) // 8
    counts: dict[RGB, int] = {}
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            offset = (y * resolution + x) * 3
            pixel_bytes = rgb[offset : offset + 3]
            pixel = pixel_bytes[0], pixel_bytes[1], pixel_bytes[2]
            counts[pixel] = counts.get(pixel, 0) + 1
    center_area = (2 * radius + 1) ** 2
    center_color_name = selected["open_center_color"]
    if not isinstance(center_color_name, str):
        raise ValueError("open center color name must be text")
    light_fraction = counts.get(palette[center_color_name], 0) / center_area
    dark_fraction = counts.get(palette["dark"], 0) / center_area
    door_fraction = counts.get(palette[active_color.value], 0) / center_area
    door_color_pixels = exact_color_count_in_cell(
        rgb,
        position=position,
        resolution=resolution,
        color=palette[active_color.value],
    )
    dark_pixels = exact_color_count_in_cell(
        rgb,
        position=position,
        resolution=resolution,
        color=palette["dark"],
    )
    if light_fraction >= 0.95 and door_color_pixels >= 400 * scale**2:
        state = "open"
    elif (
        light_fraction <= 0.05
        and door_fraction + dark_fraction >= 0.95
        and dark_fraction >= 0.15
        and door_color_pixels >= 1000 * scale**2
        and dark_pixels >= 500 * scale**2
    ):
        state = "closed"
    else:
        raise ValueError(
            "door region is visually ambiguous: "
            f"light={light_fraction:.3f}, "
            f"door={door_fraction:.3f}, dark={dark_fraction:.3f}"
        )
    return {
        "state": state,
        "center_region_area": center_area,
        "center_light_fraction": light_fraction,
        "center_door_fraction": door_fraction,
        "center_dark_fraction": dark_fraction,
        "door_color_pixels": door_color_pixels,
        "dark_pixels": dark_pixels,
    }


def agent_position_report(
    rgb: bytes,
    *,
    resolution: int = 448,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Locate the unique agent-colored cell over all 49 legal cells."""

    if len(rgb) != resolution * resolution * 3:
        raise ValueError("RGB payload size differs from resolution")
    palette, _, _, _ = _active_values("legacy", resolution, config)
    counts: dict[Position, int] = {}
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            position = Position(row, col)
            count = exact_color_count_in_cell(
                rgb,
                position=position,
                resolution=resolution,
                color=palette["agent"],
            )
            if count:
                counts[position] = count
    if len(counts) != 1:
        readable = {
            f"{position.row},{position.col}": count
            for position, count in counts.items()
        }
        raise ValueError(f"cannot locate exactly one agent cell: {readable}")
    position, count = next(iter(counts.items()))
    return {
        "position": position,
        "agent_color_pixels": count,
    }


def _neutral_floor_position(
    spec: WorldSpec,
    *,
    exclude: Position,
) -> Position:
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            candidate = Position(row, col)
            if (
                candidate != exclude
                and candidate not in spec.walls
                and spec.door_at(candidate) is None
                and candidate != spec.goal
            ):
                return candidate
    raise ValueError("world has no neutral floor cell for visibility audit")


def switch_visibility_report(
    spec: WorldSpec,
    *,
    color: Color | str,
    renderer: str,
    resolution: int = 448,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare the same switch with and without an agent on top of it."""

    try:
        active_color = Color(color)
    except (TypeError, ValueError) as exc:
        raise ValueError("switch color must be red or blue") from exc
    active = _active_config(config)
    palette, _, _, scale = _active_values(renderer, resolution, active)
    switch_position = spec.switch_map[active_color]
    neutral = _neutral_floor_position(spec, exclude=switch_position)
    empty_state = WorldState(agent=neutral)
    occupied_state = WorldState(agent=switch_position)
    empty_rgb = render_rgb(
        spec,
        empty_state,
        renderer=renderer,
        resolution=resolution,
        config=active,
    )
    occupied_rgb = render_rgb(
        spec,
        occupied_state,
        renderer=renderer,
        resolution=resolution,
        config=active,
    )
    unoccupied_count = exact_color_count_in_cell(
        empty_rgb,
        position=switch_position,
        resolution=resolution,
        color=palette[active_color.value],
    )
    occupied_count = exact_color_count_in_cell(
        occupied_rgb,
        position=switch_position,
        resolution=resolution,
        color=palette[active_color.value],
    )
    if unoccupied_count <= 0:
        raise ValueError("unoccupied switch has no visible color pixels")
    visible_fraction = occupied_count / unoccupied_count
    passed = occupied_count >= 1000 * scale**2 and visible_fraction >= 0.55
    return {
        "color": active_color.value,
        "unoccupied_color_pixels": unoccupied_count,
        "occupied_color_pixels": occupied_count,
        "occupied_visible_fraction": visible_fraction,
        "minimum_occupied_color_pixels": 1000 * scale**2,
        "minimum_visible_fraction": 0.55,
        "pass": passed,
    }


# Backwards-friendly descriptive alias; both names use the new dynamic API.
visible_switch_report = switch_visibility_report


def audit_rendered_state(
    spec: WorldSpec,
    state: WorldState,
    rgb: bytes,
    *,
    renderer: str,
    resolution: int = 448,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Model-free proof that one frame preserves its symbolic semantics."""

    if len(rgb) != resolution * resolution * 3:
        raise ValueError("RGB payload size differs from resolution")
    validate_state_for_spec(spec, state)
    agent = agent_position_report(
        rgb,
        resolution=resolution,
        config=config,
    )
    if agent["position"] != state.agent:
        raise ValueError("rendered agent position disagrees with symbolic state")
    red = door_region_report(
        rgb,
        position=spec.door_map[Color.RED],
        color=Color.RED,
        renderer=renderer,
        resolution=resolution,
        config=config,
    )
    blue = door_region_report(
        rgb,
        position=spec.door_map[Color.BLUE],
        color=Color.BLUE,
        renderer=renderer,
        resolution=resolution,
        config=config,
    )
    if red["state"] != ("open" if state.red_door_open else "closed"):
        raise ValueError("rendered red door disagrees with symbolic state")
    if blue["state"] != ("open" if state.blue_door_open else "closed"):
        raise ValueError("rendered blue door disagrees with symbolic state")

    switch_color = spec.switch_at(state.agent)
    switch_report: dict[str, Any] | None = None
    if switch_color is not None:
        switch_report = switch_visibility_report(
            spec,
            color=switch_color,
            renderer=renderer,
            resolution=resolution,
            config=config,
        )
        if not switch_report["pass"]:
            raise ValueError("agent-occupied switch is not visibly recoverable")
    return {
        "agent_position": state.agent.to_dict(),
        "agent_color_pixels": agent["agent_color_pixels"],
        "red_door": red,
        "blue_door": blue,
        "occupied_switch": switch_report,
        "pass": True,
    }


def visibility_report(
    spec: WorldSpec,
    state: WorldState,
    rgb: bytes | None = None,
    *,
    renderer: str,
    resolution: int = 448,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit one frame using the stable dataset-facing interface.

    When ``rgb`` is supplied, it must be byte-identical to a fresh deterministic
    render.  Omitting it renders the frame internally.  In either case, the
    returned report proves recoverable agent position and door states, and also
    proves switch visibility whenever the agent occupies a switch.
    """

    expected = render_rgb(
        spec,
        state,
        renderer=renderer,
        resolution=resolution,
        config=config,
    )
    observed = expected if rgb is None else rgb
    if observed != expected:
        raise ValueError("frame bytes differ from deterministic symbolic render")
    report = audit_rendered_state(
        spec,
        state,
        observed,
        renderer=renderer,
        resolution=resolution,
        config=config,
    )
    return {
        **report,
        "renderer": renderer,
        "resolution": resolution,
        "pixel_sha256": hashlib.sha256(observed).hexdigest(),
        "exact_symbolic_rerender": True,
    }


def changed_pixel_positions(
    first: bytes,
    second: bytes,
    *,
    resolution: int,
) -> frozenset[tuple[int, int]]:
    if len(first) != resolution * resolution * 3 or len(second) != len(first):
        raise ValueError("RGB payloads have incompatible dimensions")
    changed: set[tuple[int, int]] = set()
    for index in range(resolution * resolution):
        start = index * 3
        if first[start : start + 3] != second[start : start + 3]:
            changed.add((index // resolution, index % resolution))
    return frozenset(changed)


def positions_within_cell(
    positions: Iterable[tuple[int, int]],
    *,
    cell: Position,
    resolution: int,
) -> bool:
    x0, y0, x1, y1 = cell_box(cell, resolution)
    return all(y0 <= row <= y1 and x0 <= col <= x1 for row, col in positions)


def contact_sheet_rgb(
    images: Sequence[bytes],
    *,
    image_size: int = 448,
    columns: int = 4,
) -> tuple[bytes, int, int]:
    if not images or columns <= 0 or len(images) % columns:
        raise ValueError("contact sheet requires complete, non-empty rows")
    if any(len(image) != image_size * image_size * 3 for image in images):
        raise ValueError("contact sheet image dimensions differ")
    rows = len(images) // columns
    canvas = Raster(
        columns * image_size,
        rows * image_size,
        (235, 236, 232),
    )
    for index, image in enumerate(images):
        canvas.paste(
            image,
            image_size,
            image_size,
            (index % columns) * image_size,
            (index // columns) * image_size,
        )
    return bytes(canvas.pixels), canvas.width, canvas.height
