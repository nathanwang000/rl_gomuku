"""Serverless policy-probs endpoint — returns action_probs without making a move."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from game import GameState
from policy import get_policy

HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*"
}


def handler(event, context=None):
    method = event.get("httpMethod", "POST")
    if method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                **HEADERS, "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": "",
        }

    try:
        body = json.loads(event.get("body", "{}"))
    except (json.JSONDecodeError, TypeError):
        return {
            "statusCode": 400,
            "headers": HEADERS,
            "body": json.dumps({"error": "Invalid JSON body"})
        }

    board = body.get("board")
    current_player = body.get("current_player")
    policy_name = body.get("policy", "smart")
    temperature = float(body.get("temperature", 0.0))
    mcts_simulations = int(body.get("mcts_simulations", 0))

    if board is None or current_player is None:
        return {
            "statusCode": 400,
            "headers": HEADERS,
            "body":
            json.dumps({"error": "Missing 'board' or 'current_player'"})
        }

    state = GameState(
        board=np.array(board, dtype=np.int8),
        current_player=int(current_player),
        winner=None,
        last_move=None,
    )

    try:
        if policy_name.startswith("checkpoint:"):
            # Neural/MCTS policies require the full app-server setup; not available serverless
            return {
                "statusCode":
                400,
                "headers":
                HEADERS,
                "body":
                json.dumps({
                    "error":
                    "Neural policies not supported in serverless policy endpoint"
                })
            }
        policy = get_policy(policy_name)
    except KeyError:
        return {
            "statusCode": 400,
            "headers": HEADERS,
            "body": json.dumps({"error": f"Unknown policy: {policy_name}"})
        }

    probs = policy.action_probs(state)
    probs_list = [[int(r), int(c), round(p, 5)] for (r, c), p in probs.items()
                  if p > 1e-4]

    return {
        "statusCode": 200,
        "headers": HEADERS,
        "body": json.dumps({"probs": probs_list})
    }
