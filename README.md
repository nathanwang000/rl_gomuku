# Gomoku Web App

A clean, extensible Gomoku (Five in a Row) game designed as a playground for RL experiments.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser.

## Architecture

```
game.py      — Pure game logic (GameState), no IO
policy.py    — Player policy interface + implementations
app.py       — Flask server, manages sessions and routes
static/      — Single-page frontend (vanilla JS + Canvas)
```

## Adding a New AI Policy

1. Subclass `Policy` in `policy.py`
2. Implement `select_move(state: GameState) -> (row, col)`
3. Register it in the `POLICIES` dict
4. It will automatically appear as a player option (add to frontend dropdown)

## Player Options

- **Human** — clicks on the board
- **Random AI** — picks a random valid move (baseline)

## Designed for RL

- `GameState` exposes `board` as a numpy array, `valid_moves()`, and `make_move()`
- Policies are stateless functions of the game state — easy to wrap with neural nets
- The game loop cleanly separates logic from IO for headless self-play
