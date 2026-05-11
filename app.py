"""Local development server — mimics the serverless deployment locally.

Serves static files from public/ and handles POST /api/move as a stateless endpoint.
No server-side game state — all state lives in the client.
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import GameState
from policy import get_policy


class GomokuHandler(SimpleHTTPRequestHandler):
    """Serves static files from public/ and handles the /api/move endpoint."""

    def __init__(self, *args, **kwargs):
        # Serve files from public/ directory
        super().__init__(*args, directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "public"), **kwargs)

    def do_POST(self):
        if self.path == "/api/move":
            self._handle_ai_move()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handle_ai_move(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length else b"{}"

        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON"})
            return

        board = body.get("board")
        current_player = body.get("current_player")
        policy_name = body.get("policy", "smart")

        if board is None or current_player is None:
            self._respond(400, {"error": "Missing 'board' or 'current_player'"})
            return

        state = GameState(
            board=np.array(board, dtype=np.int8),
            current_player=int(current_player),
            winner=None,
            last_move=None,
        )

        try:
            policy = get_policy(policy_name)
        except KeyError:
            self._respond(400, {"error": f"Unknown policy: {policy_name}"})
            return

        row, col = policy.select_move(state)
        self._respond(200, {"row": int(row), "col": int(col)})

    def _respond(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Cleaner logging
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    port = int(os.environ.get("PORT", 5000))
    server = HTTPServer(("0.0.0.0", port), GomokuHandler)
    print(f"🎮 Gomoku dev server running at http://localhost:{port}")
    print(f"   Serving static files from public/")
    print(f"   AI endpoint at POST /api/move")
    print(f"   Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
