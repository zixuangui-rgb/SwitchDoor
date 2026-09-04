from __future__ import annotations

import copy
import http.client
import json
from collections.abc import Iterator
from threading import Thread

import pytest

from switchdoor_three_level.config import load_config, validate_config
from switchdoor_three_level.layouts import materialize_levels
from switchdoor_three_level.model import Color, Rule, SwitchMapping
from switchdoor_three_level.solver import shortest_plan
from switchdoor_three_level.web_app import SwitchDoorWebServer, create_server


@pytest.fixture(scope="module")
def fast_config() -> dict:
    value = copy.deepcopy(load_config())
    value["render"]["resolution"] = 128
    value["render"]["png_compression_level"] = 1
    return validate_config(value)


@pytest.fixture()
def web_server(fast_config: dict) -> Iterator[SwitchDoorWebServer]:
    server = create_server("127.0.0.1", 0, config=fast_config)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(
    server: SwitchDoorWebServer,
    method: str,
    path: str,
    payload: dict | None = None,
) -> tuple[int, dict[str, object] | bytes, str]:
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=5)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    content_type = response.getheader("Content-Type", "")
    connection.close()
    if content_type.startswith("application/json"):
        return response.status, json.loads(raw), content_type
    return response.status, raw, content_type


def _post_ok(
    server: SwitchDoorWebServer,
    path: str,
    payload: dict,
) -> dict[str, object]:
    status, body, _ = _request(server, "POST", path, payload)
    assert status == 200
    assert isinstance(body, dict)
    return body


def test_web_assets_and_security_headers_are_served(web_server: SwitchDoorWebServer) -> None:
    for path, marker in (
        ("/", "MODEL-VISIBLE FEED"),
        ("/favicon.svg", "<svg"),
        ("/styles.css", "--phosphor"),
        ("/app.js", 'request("/api/session"'),
    ):
        status, body, _ = _request(web_server, "GET", path)
        assert status == 200
        assert isinstance(body, bytes)
        assert marker.encode() in body

    host, port = web_server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=5)
    connection.request("GET", "/")
    response = connection.getresponse()
    response.read()
    assert response.getheader("X-Content-Type-Options") == "nosniff"
    assert "default-src 'self'" in response.getheader("Content-Security-Policy", "")
    connection.close()


def test_state_probe_is_clean_before_session_start(web_server: SwitchDoorWebServer) -> None:
    status, body, content_type = _request(web_server, "GET", "/api/state")
    assert status == 200
    assert content_type.startswith("application/json")
    assert body == {"session_active": False}


def test_session_api_separates_public_observation_from_runner_state(
    web_server: SwitchDoorWebServer,
) -> None:
    payload = _post_ok(
        web_server,
        "/api/session",
        {"root_seed": 4107, "mapping": "same_color"},
    )
    assert set(payload) == {"observation", "runner"}
    observation = payload["observation"]
    runner = payload["runner"]
    assert isinstance(observation, dict)
    assert isinstance(runner, dict)
    assert set(observation) == {
        "image_url",
        "available_actions",
        "status",
        "previous_action",
    }
    assert set(runner) == {
        "level",
        "level_id",
        "level_done",
        "episode_done",
        "can_advance",
        "session_mode",
        "root_seed",
        "action_count",
    }
    encoded = json.dumps(payload)
    assert "same_color" not in encoded
    assert "cross_color" not in encoded
    assert "rgb_frame" not in encoded
    assert runner["session_mode"] == "practice"
    assert runner["root_seed"] == 4107

    status, frame, content_type = _request(web_server, "GET", observation["image_url"])
    assert status == 200
    assert content_type == "image/png"
    assert isinstance(frame, bytes)
    assert frame.startswith(b"\x89PNG\r\n\x1a\n")

    stepped = _post_ok(web_server, "/api/action", {"action": "INTERACT"})
    assert stepped["observation"]["previous_action"] == "INTERACT"
    assert stepped["runner"]["action_count"] == 1
    assert stepped["observation"]["image_url"] != observation["image_url"]


def test_api_rejects_invalid_actions_and_early_advance(
    web_server: SwitchDoorWebServer,
) -> None:
    _post_ok(web_server, "/api/session", {"root_seed": 9, "mapping": "random"})
    status, body, _ = _request(web_server, "POST", "/api/action", {"action": "JUMP"})
    assert status == 400
    assert isinstance(body, dict) and "invalid action" in body["error"]

    status, body, _ = _request(web_server, "POST", "/api/advance", {})
    assert status == 409
    assert isinstance(body, dict) and body["error"] == "the next level is not available"


def test_three_level_web_flow_advances_then_reveals_rule(
    web_server: SwitchDoorWebServer,
    fast_config: dict,
) -> None:
    root_seed = 20260826
    mapping = SwitchMapping.CROSS_COLOR
    levels = materialize_levels(fast_config, root_seed)
    payload = _post_ok(
        web_server,
        "/api/session",
        {"root_seed": root_seed, "mapping": mapping.value},
    )

    for level_index, level in enumerate(levels):
        required_switch = Color.RED if level_index == 0 else None
        plan = shortest_plan(level.spec, Rule(mapping), required_first_switch=required_switch)
        assert plan is not None
        for action in plan:
            payload = _post_ok(web_server, "/api/action", {"action": action.value})

        runner = payload["runner"]
        assert runner["level"] == level_index + 1
        assert runner["level_done"] is True
        if level_index < 2:
            assert runner["episode_done"] is False
            assert runner["can_advance"] is True
            assert "revealed_mapping" not in runner
            payload = _post_ok(web_server, "/api/advance", {})
            assert payload["runner"]["level"] == level_index + 2
            assert payload["runner"]["level_done"] is False
            assert payload["observation"]["previous_action"] is None
        else:
            assert runner["episode_done"] is True
            assert runner["can_advance"] is False
            assert runner["revealed_mapping"] == mapping.value
