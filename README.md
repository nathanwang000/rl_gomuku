# Gomoku Web App (Serverless)

A clean, extensible Gomoku (Five in a Row) game. Game state lives entirely in the
browser; the AI is a **stateless serverless function** that receives a board and returns
a move. Ready to deploy on **Vercel** or **Netlify** with zero server infrastructure.

## Quick Start (Local Dev)

```bash
uv run python app.py
```

Open http://localhost:5000 — that's it. No database, no sessions.

## Architecture

```
public/index.html   — Single-page frontend (vanilla JS + Canvas, owns all game state)
api/move.py         — Vercel serverless function (stateless AI endpoint)
netlify/functions/  — Netlify function wrapper (same logic)
game.py             — Pure game engine (board state, win detection)
policy.py           — AI policy interface + implementations (the serverless backend brain)
app.py              — Local dev server (mimics serverless routing)
```

### How It Works

1. **Client** manages the full game state (board, current player, win detection)
2. When it's an AI's turn, client sends `POST /api/move` with:
   ```json
   { "board": [[0,0,...], ...], "current_player": 1, "policy": "random" }
   ```
3. **Serverless function** reconstructs state, runs the policy, returns:
   ```json
   { "row": 7, "col": 7 }
   ```
4. Client applies the move locally and continues

No server-side sessions. No database. Fully stateless.

## Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

The `vercel.json` routes everything correctly:
- `POST /api/move` → Python serverless function
- `GET /*` → static files from `public/`

## Deploy to Netlify

```bash
# Install Netlify CLI
npm i -g netlify-cli

# Deploy
netlify deploy --prod
```

The `netlify.toml` handles:
- Static site from `public/`
- Function redirect `/api/move` → `/.netlify/functions/move`

## Adding a New AI Policy

1. Subclass `Policy` in `policy.py`
2. Implement `select_move(state: GameState) -> (row, col)`
3. Register it in the `POLICIES` dict
4. Add the option to the frontend dropdown in `public/index.html`

The function is truly stateless — each call receives the full board.
This makes it trivial to add neural-net policies, MCTS, etc.

## API Reference

### `POST /api/move`

**Request:**
```json
{
  "board": [[0,0,0,...], ...],   // 15x15 array, 0=empty, 1=black, 2=white
  "current_player": 1,           // whose turn it is
  "policy": "random"             // which AI to use
}
```

**Response:**
```json
{
  "row": 7,
  "col": 7
}
```

## Designed for RL

- `GameState` exposes board as numpy array + `valid_moves()` — ready for neural nets
- Policies are pure functions of state — no hidden state, easy to test
- The serverless model means you can scale AI inference independently
- Swap `random` for a trained model by updating `policy.py`
