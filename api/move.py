"""Serverless AI endpoint — stateless: receives board, returns move.

Vercel/Netlify Python serverless function.
Delegates to policy.py for move computation.
"""

import json
import sys
import os

# Add parent directory so we can import policy/game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState
from policy import get_policy
import numpy as np


def handler(event, context=None):
    """Netlify function handler."""
    try:
        body = json.loads(event.get("body", "{}"))
    except (json.JSONDecodeError, TypeError):
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Invalid JSON body"}),
        }

    # Handle CORS preflight
    method = event.get("httpMethod", "POST")
    if method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
            "body": "",
        }

    board = body.get("board")
    current_player = body.get("current_player")
    policy_name = body.get("policy", "random")

    if board is None or current_player is None:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Missing 'board' or 'current_player'"}),
        }

    # Reconstruct game state from client data
    state = GameState(
        board=np.array(board, dtype=np.int8),
        current_player=int(current_player),
        winner=None,
        last_move=None,
    )

    # Get AI policy and compute move
    try:
        policy = get_policy(policy_name)
    except KeyError:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": f"Unknown policy: {policy_name}"}),
        }

    row, col = policy.select_move(state)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps({"row": int(row), "col": int(col)}),
    }


# --- Vercel handler (uses ASGI/WSGI-like interface via Flask) ---
# For Vercel, we expose a minimal Flask/ASGI app that routes POST /api/move

from http.server import BaseHTTPRequestHandler


class Handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler (Python runtime)."""

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length else b"{}"

        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON"})
            return

        board = body.get("board")
        current_player = body.get("current_player")
        policy_name = body.get("policy", "random")

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
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
