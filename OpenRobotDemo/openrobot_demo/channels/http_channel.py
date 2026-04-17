"""HTTP input channel for OpenRobotDemo (zero-dependency, stdlib only)."""

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable
from urllib.parse import urlparse, parse_qs

from .base import Channel

logger = logging.getLogger(__name__)


class _HealthHandler(BaseHTTPRequestHandler):
    """Request handler injected with queue access."""

    # These will be set by HTTPChannel before server starts
    on_message: Callable[[str], str] | None = None
    get_status: Callable[[str], dict | None] | None = None

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/health":
            self._send_json({"status": "ok", "channel": "http"})
            return

        if path == "/status":
            episode_id = qs.get("episode_id", [None])[0]
            if not episode_id:
                self._send_json({"error": "Missing episode_id"}, status=400)
                return
            handler_cls = self.__class__
            status = handler_cls.get_status(episode_id) if handler_cls.get_status else None
            if status is None:
                self._send_json({"error": "Episode not found"}, status=404)
                return
            self._send_json(status)
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        if self.path != "/task":
            self._send_json({"error": "Not found"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json({"error": "Empty body"}, status=400)
            return

        try:
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, status=400)
            return

        instruction = data.get("instruction")
        if not instruction or not isinstance(instruction, str):
            self._send_json({"error": "Missing or invalid 'instruction' field"}, status=400)
            return

        try:
            handler_cls = self.__class__
            response_text = handler_cls.on_message(instruction) if handler_cls.on_message else "No handler"
            # Try to parse response as JSON, otherwise wrap it
            try:
                payload = json.loads(response_text)
            except json.JSONDecodeError:
                payload = {"response": response_text}
            self._send_json(payload)
        except Exception as exc:
            logger.exception("[HTTPChannel] Error handling task request")
            self._send_json({"error": str(exc)}, status=500)


class HTTPChannel(Channel):
    """Receive instructions via HTTP POST /task and expose /status and /health."""

    name = "http"

    def __init__(self, port: int = 8080, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._on_message: Callable[[str], str] | None = None

    def start(self, on_message: Callable[[str], str]) -> None:
        self._on_message = on_message

        # Inject dependencies into the handler class
        _HealthHandler.on_message = on_message

        self._server = HTTPServer((self.host, self.port), _HealthHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("[HTTPChannel] Server started on http://%s:%d", self.host, self.port)
        print(f"\n🌐 HTTP channel listening on http://{self.host}:{self.port}")
        print("   POST /task    -> {\"instruction\": \"...\"}")
        print("   GET  /status  -> ?episode_id=xxx")
        print("   GET  /health  -> health check")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        logger.info("[HTTPChannel] Server stopped.")

    def bind_status_lookup(self, get_status: Callable[[str], dict | None]):
        """Allow the HTTP handler to query task status by episode_id."""
        _HealthHandler.get_status = get_status
