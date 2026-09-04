"""Dependency-free local Web console for the live three-level environment."""

from __future__ import annotations

import json
import secrets
import sysconfig
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlsplit

from .config import load_config, validate_config
from .live_env import LiveEnvError, PublicObservation, ThreeLevelSwitchDoorEnv
from .model import SwitchMapping
from .render import encode_png

MAX_REQUEST_BYTES = 4096
ASSET_TYPES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class WebRequestError(RuntimeError):
    """A client-visible request failure with an explicit HTTP status."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def _default_web_root() -> Path:
    """Locate assets in an editable checkout or an installed data-files tree."""

    source_root = Path(__file__).resolve().parents[2] / "web"
    if source_root.is_dir():
        return source_root
    installed_root = Path(sysconfig.get_path("data")) / "share" / "switchdoor-three-level" / "web"
    return installed_root


class WebSession:
    """Own private runner state and export an explicitly partitioned Web view."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = validate_config(config)
        self._lock = RLock()
        self._env: ThreeLevelSwitchDoorEnv | None = None
        self._observation: PublicObservation | None = None
        self._mapping: SwitchMapping | None = None
        self._root_seed: int | None = None
        self._session_mode: str | None = None
        self._level_index = 0
        self._level_done = False
        self._episode_done = False
        self._action_count = 0
        self._frame_png: bytes | None = None
        self._frame_version = 0

    def start(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        extra = set(payload) - {"mapping", "root_seed"}
        if extra:
            raise WebRequestError(
                HTTPStatus.BAD_REQUEST,
                f"unsupported session fields: {', '.join(sorted(extra))}",
            )

        mapping_choice = payload.get("mapping", "random")
        if mapping_choice not in {"random", "same_color", "cross_color"}:
            raise WebRequestError(
                HTTPStatus.BAD_REQUEST,
                "mapping must be random, same_color, or cross_color",
            )
        raw_seed = payload.get("root_seed")
        if raw_seed is None:
            root_seed = secrets.randbits(63)
        elif isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise WebRequestError(HTTPStatus.BAD_REQUEST, "root_seed must be an integer")
        else:
            root_seed = raw_seed

        if mapping_choice == "random":
            mapping = secrets.choice(tuple(SwitchMapping))
            session_mode = "blind"
        else:
            mapping = SwitchMapping(mapping_choice)
            session_mode = "practice"

        with self._lock:
            env = ThreeLevelSwitchDoorEnv(self._config)
            observation = env.reset(root_seed=root_seed, mapping=mapping)
            self._env = env
            self._mapping = mapping
            self._root_seed = root_seed
            self._session_mode = session_mode
            self._level_index = 0
            self._level_done = False
            self._episode_done = False
            self._action_count = 0
            self._set_observation(observation)
            return self._snapshot()

    def act(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if set(payload) != {"action"} or not isinstance(payload.get("action"), str):
            raise WebRequestError(
                HTTPStatus.BAD_REQUEST,
                "action request must contain exactly one string field named action",
            )
        with self._lock:
            env = self._require_env()
            if self._level_done:
                raise WebRequestError(
                    HTTPStatus.CONFLICT,
                    "current level is complete; advance before sending another action",
                )
            try:
                result = env.step(payload["action"])
            except ValueError as exc:
                raise WebRequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
            except LiveEnvError as exc:
                raise WebRequestError(HTTPStatus.CONFLICT, str(exc)) from exc
            self._level_done = result.level_done
            self._episode_done = result.episode_done
            self._action_count += 1
            self._set_observation(result.observation)
            return self._snapshot()

    def advance(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if payload:
            raise WebRequestError(
                HTTPStatus.BAD_REQUEST,
                "advance request must use an empty JSON object",
            )
        with self._lock:
            env = self._require_env()
            if not self._level_done or self._episode_done:
                raise WebRequestError(
                    HTTPStatus.CONFLICT,
                    "the next level is not available",
                )
            try:
                observation = env.advance_level()
            except LiveEnvError as exc:
                raise WebRequestError(HTTPStatus.CONFLICT, str(exc)) from exc
            self._level_index += 1
            self._level_done = False
            self._set_observation(observation)
            return self._snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._require_env()
            return self._snapshot()

    def optional_snapshot(self) -> dict[str, Any] | None:
        """Return the current session, or None before the first reset."""

        with self._lock:
            return None if self._env is None else self._snapshot()

    def frame_png(self) -> bytes:
        with self._lock:
            self._require_env()
            assert self._frame_png is not None
            return self._frame_png

    def _require_env(self) -> ThreeLevelSwitchDoorEnv:
        if self._env is None:
            raise WebRequestError(HTTPStatus.CONFLICT, "start a session first")
        return self._env

    def _set_observation(self, observation: PublicObservation) -> None:
        self._observation = observation
        render = self._config["render"]
        self._frame_png = encode_png(
            observation.rgb_frame,
            int(render["resolution"]),
            int(render["png_compression_level"]),
        )
        self._frame_version += 1

    def _snapshot(self) -> dict[str, Any]:
        assert self._observation is not None
        assert self._root_seed is not None
        assert self._session_mode is not None

        # Deliberately construct the model-visible namespace from its exact
        # allowlist instead of serializing the environment object wholesale.
        observation = {
            "image_url": f"/api/frame.png?v={self._frame_version}",
            "available_actions": list(self._observation.available_actions),
            "status": self._observation.status,
            "previous_action": self._observation.previous_action,
        }
        runner: dict[str, Any] = {
            "level": self._level_index + 1,
            "level_id": f"L{self._level_index + 1}",
            "level_done": self._level_done,
            "episode_done": self._episode_done,
            "can_advance": self._level_done and not self._episode_done,
            "session_mode": self._session_mode,
            "root_seed": self._root_seed,
            "action_count": self._action_count,
        }
        if self._episode_done:
            assert self._mapping is not None
            runner["revealed_mapping"] = self._mapping.value
        return {"observation": observation, "runner": runner}


class SwitchDoorWebServer(ThreadingHTTPServer):
    """HTTP server carrying the application and an explicit static root."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        application: WebSession,
        web_root: Path,
    ) -> None:
        self.application = application
        self.web_root = web_root
        super().__init__(server_address, SwitchDoorRequestHandler)


class SwitchDoorRequestHandler(BaseHTTPRequestHandler):
    """Serve the console assets and the small same-origin JSON API."""

    server: SwitchDoorWebServer

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        try:
            if path in ASSET_TYPES:
                filename, content_type = ASSET_TYPES[path]
                self._send_bytes(
                    HTTPStatus.OK,
                    (self.server.web_root / filename).read_bytes(),
                    content_type,
                    cache_control="no-cache",
                )
            elif path == "/api/state":
                snapshot = self.server.application.optional_snapshot()
                self._send_json(
                    HTTPStatus.OK,
                    {"session_active": False} if snapshot is None else snapshot,
                )
            elif path == "/api/frame.png":
                self._send_bytes(
                    HTTPStatus.OK,
                    self.server.application.frame_png(),
                    "image/png",
                    cache_control="no-store",
                )
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except WebRequestError as exc:
            self._send_json(exc.status, {"error": str(exc)})
        except OSError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "asset not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            payload = self._read_json_object()
            if path == "/api/session":
                result = self.server.application.start(payload)
            elif path == "/api/action":
                result = self.server.application.act(payload)
            elif path == "/api/advance":
                result = self.server.application.advance(payload)
            else:
                raise WebRequestError(HTTPStatus.NOT_FOUND, "not found")
            self._send_json(HTTPStatus.OK, result)
        except WebRequestError as exc:
            self._send_json(exc.status, {"error": str(exc)})
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "request body must be JSON"})

    def _read_json_object(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise WebRequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise WebRequestError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from exc
        if not 0 <= length <= MAX_REQUEST_BYTES:
            raise WebRequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large")
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(value, dict):
            raise WebRequestError(HTTPStatus.BAD_REQUEST, "request JSON must be an object")
        return value

    def _send_json(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(status, payload, "application/json; charset=utf-8", "no-store")

    def _send_bytes(
        self,
        status: HTTPStatus,
        payload: bytes,
        content_type: str,
        cache_control: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(payload)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    config: Mapping[str, Any] | None = None,
    web_root: str | Path | None = None,
) -> SwitchDoorWebServer:
    """Build a server without starting its blocking loop, useful for tests."""

    resolved_config = load_config() if config is None else validate_config(config)
    resolved_web_root = _default_web_root() if web_root is None else Path(web_root)
    missing = [name for name, _ in ASSET_TYPES.values() if not (resolved_web_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing Web assets in {resolved_web_root}: {sorted(set(missing))}"
        )
    return SwitchDoorWebServer(
        (host, port),
        application=WebSession(resolved_config),
        web_root=resolved_web_root,
    )


def run_web_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    config: Mapping[str, Any] | None = None,
) -> int:
    """Run the local Web console until interrupted."""

    server = create_server(host, port, config=config)
    actual_host, actual_port = server.server_address[:2]
    display_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    print(f"SwitchDoor Web console: http://{display_host}:{actual_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSwitchDoor Web console stopped.")
    finally:
        server.server_close()
    return 0
