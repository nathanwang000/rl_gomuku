"""Netlify serverless function wrapper for the AI move endpoint."""

import json
import sys
import os

# Add project root so we can import game/policy modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from game import GameState
from policy import get_policy
import numpy as np


def handler(event, context):
    """Netlify function entry point."""
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    # CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": headers, "body": ""}

    try:
        body = json.loads(event.get("body", "{}"))
    except (json.JSONDecodeError, TypeError):
        return {"statusCode": 400, "headers": headers, "body": json.dumps({"error": "Invalid JSON"})}

    board = body.get("board")
    current_player = body.get("current_player")
    policy_name = body.get("policy", "random")

    if board is None or current_player is None:
        return {"statusCode": 400, "headers": headers, "body": json.dumps({"error": "Missing 'board' or 'current_player'"})}

    # Reconstruct state
    state = GameState(
        board=np.array(board, dtype=np.int8),
        current_player=int(current_player),
        winner=None,
        last_move=None,
    )

    try:
        policy = get_policy(policy_name)
    except KeyError:
        return {"statusCode": 400, "headers": headers, "body": json.dumps({"error": f"Unknown policy: {policy_name}"})}

    row, col = policy.select_move(state)
    return {"statusCode": 200, "headers": headers, "body": json.dumps({"row": int(row), "col": int(col)})}
