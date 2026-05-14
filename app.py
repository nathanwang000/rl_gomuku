"""Local development server — mimics the serverless deployment locally.

Serves static files from public/ and handles POST /api/move as a stateless endpoint.
No server-side game state — all state lives in the client.
"""

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import BOARD_SIZE, GameState
from policy import get_policy

# Cache loaded neural nets so we don't reload weights on every request
_neural_net_cache: dict = {}


def _get_neural_net(checkpoint_name: str):
    """Load (or return cached) PolicyValueNet for a given checkpoint filename."""
    if checkpoint_name in _neural_net_cache:
        return _neural_net_cache[checkpoint_name]

    import torch
    from rl.network import PolicyValueNet

    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "checkpoints", checkpoint_name)
    net = PolicyValueNet(board_size=BOARD_SIZE)
    # cpu mapping as server may not have GPU
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    net.eval()
    _neural_net_cache[checkpoint_name] = net
    return net


def _make_neural_policy(checkpoint_name: str, temperature: float,
                        mcts_simulations: int):
    """Build a NeuralPolicy or MCTSPolicy around the cached network."""
    net = _get_neural_net(checkpoint_name)
    if mcts_simulations > 0:
        from rl.mcts import MCTSPolicy
        return MCTSPolicy(net,
                          simulations=mcts_simulations,
                          temperature=temperature,
                          device="cpu")
    else:
        from rl.selfplay import NeuralPolicy
        return NeuralPolicy(net, temperature=temperature, device="cpu")


def _list_checkpoints() -> list:
    """Return sorted list of .pt filenames in the checkpoints/ directory."""
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))


class GomokuHandler(SimpleHTTPRequestHandler):
    """Serves static files from public/ and handles the /api/move endpoint."""

    def __init__(self, *args, **kwargs):
        # Serve files from public/ directory
        super().__init__(*args,
                         directory=os.path.join(
                             os.path.dirname(os.path.abspath(__file__)),
                             "public"),
                         **kwargs)

    def do_GET(self):
        if self.path == "/api/checkpoints":
            self._respond(200, {"checkpoints": _list_checkpoints()})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/move":
            self._handle_ai_move()
        elif self.path == "/api/policy":
            self._handle_policy_probs()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handle_policy_probs(self):
        """Return action_probs for a board position without making a move."""
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(
            content_length) if content_length else b"{}"
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON"})
            return

        board = body.get("board")
        current_player = body.get("current_player")
        policy_name = body.get("policy", "smart")
        temperature = float(body.get("temperature", 0.0))
        mcts_simulations = int(body.get("mcts_simulations", 0))

        if board is None or current_player is None:
            self._respond(400,
                          {"error": "Missing 'board' or 'current_player'"})
            return

        state = GameState(
            board=np.array(board, dtype=np.int8),
            current_player=int(current_player),
            winner=None,
            last_move=None,
        )

        try:
            if policy_name.startswith("checkpoint:"):
                checkpoint_name = policy_name[len("checkpoint:"):]
                policy = _make_neural_policy(checkpoint_name, temperature,
                                             mcts_simulations)
            else:
                policy = get_policy(policy_name)
        except (KeyError, FileNotFoundError) as e:
            self._respond(
                400,
                {"error": f"Unknown or missing policy: {policy_name} ({e})"})
            return

        probs = policy.action_probs(state)
        probs_list = [[int(r), int(c), round(p, 5)]
                      for (r, c), p in probs.items() if p > 1e-4]
        self._respond(200, {"probs": probs_list})

    def _handle_ai_move(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(
            content_length) if content_length else b"{}"

        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON"})
            return

        board = body.get("board")
        current_player = body.get("current_player")
        policy_name = body.get("policy", "smart")
        temperature = float(body.get("temperature", 0.0))
        mcts_simulations = int(body.get("mcts_simulations", 0))
        print(
            f"[move] policy={policy_name!r}  temperature={temperature}  mcts_sims={mcts_simulations}"
        )

        if board is None or current_player is None:
            self._respond(400,
                          {"error": "Missing 'board' or 'current_player'"})
            return

        state = GameState(
            board=np.array(board, dtype=np.int8),
            current_player=int(current_player),
            winner=None,
            last_move=None,
        )

        try:
            if policy_name.startswith("checkpoint:"):
                checkpoint_name = policy_name[len("checkpoint:"):]
                policy = _make_neural_policy(checkpoint_name, temperature,
                                             mcts_simulations)
            else:
                policy = get_policy(policy_name)
        except (KeyError, FileNotFoundError) as e:
            self._respond(
                400,
                {"error": f"Unknown or missing policy: {policy_name} ({e})"})
            return

        move, probs = policy.select_move_with_probs(state)
        row, col = move
        probs_list = [[int(r), int(c), round(p, 5)]
                      for (r, c), p in probs.items() if p > 1e-4]
        self._respond(200, {
            "row": int(row),
            "col": int(col),
            "probs": probs_list
        })

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
