"""Gomoku webapp — Flask backend serving a clean single-page UI."""

from flask import Flask, jsonify, request, send_from_directory
from game import GameState
from policy import get_policy, HumanPolicy

app = Flask(__name__, static_folder="static")

# --- Game session (single game for now, extend to multi-session later) ---
session = {
    "state": None,
    "policies": {1: None, 2: None},  # Policy instances per player
}


def new_game(black="human", white="human"):
    session["state"] = GameState.new()
    session["policies"] = {1: get_policy(black), 2: get_policy(white)}


def ai_move_if_needed() -> bool:
    """If current player is AI, make a move. Returns True if move was made."""
    state = session["state"]
    if state.winner is not None:
        return False
    policy = session["policies"][state.current_player]
    if isinstance(policy, HumanPolicy):
        return False
    row, col = policy.select_move(state)
    state.make_move(row, col)
    return True


# --- Routes ---

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/new", methods=["POST"])
def api_new():
    data = request.json or {}
    black = data.get("black", "human")
    white = data.get("white", "human")
    new_game(black, white)
    # If black is AI, make first move(s)
    while ai_move_if_needed():
        pass
    return jsonify(session["state"].to_dict())


@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.json
    row, col = data["row"], data["col"]
    state = session["state"]
    if state is None:
        return jsonify({"error": "No game in progress"}), 400
    policy = session["policies"][state.current_player]
    if not isinstance(policy, HumanPolicy):
        return jsonify({"error": "Not human's turn"}), 400
    if not state.make_move(row, col):
        return jsonify({"error": "Invalid move"}), 400
    # After human moves, let AI respond
    while ai_move_if_needed():
        pass
    return jsonify(state.to_dict())


@app.route("/api/state")
def api_state():
    if session["state"] is None:
        return jsonify({"error": "No game"}), 400
    return jsonify(session["state"].to_dict())


if __name__ == "__main__":
    new_game()
    app.run(host="0.0.0.0", port=5000, debug=True)
