"""RL self-play training entry point.

Run:
    python train.py

The training loop is intentionally linear and readable:
  1. Collect  — play N games using the current policy against itself
  2. Train    — sample mini-batches from the replay buffer and update
  3. Evaluate — pit the new policy against the old one to track progress
  4. Save     — checkpoint the network weights

Hyperparameters live at the top of this file for easy tuning.
"""

import argparse
import copy
import time
from pathlib import Path

import torch
from game import BOARD_SIZE
from rl.network import PolicyValueNet
from rl.replay import ReplayBuffer
from rl.selfplay import NeuralPolicy, run_episode
from rl.trainer import Trainer

# ── Hyperparameters ────────────────────────────────────────────────────────────

ITERATIONS = 50  # outer training iterations
GAMES_PER_ITER = 20  # self-play games collected each iteration
TRAIN_STEPS_PER_ITER = 50  # gradient steps taken each iteration
BATCH_SIZE = 256
BUFFER_CAPACITY = 50_000

EVAL_GAMES = 20  # games played to compare new vs old policy
WIN_RATE_THRESHOLD = 0.55  # replace old policy only if new wins > this fraction

LR = 1e-3
NUM_BLOCKS = 4  # ResNet depth
CHANNELS = 64  # ResNet width
TEMPERATURE = 1.0  # exploration temperature during self-play
EVAL_TEMPERATURE = 0.0  # deterministic during evaluation

CHECKPOINT_DIR = Path("checkpoints")

# ──────────────────────────────────────────────────────────────────────────────


def evaluate(new_policy: NeuralPolicy, old_policy: NeuralPolicy,
             n_games: int) -> float:
    """Return win rate of new_policy against old_policy (draws count as 0.5)."""
    wins = 0.0
    for i in range(n_games):
        # Alternate colors so neither side has a systematic advantage
        if i % 2 == 0:
            winner, _ = run_episode(new_policy, old_policy, record=False)
            new_player = 1
        else:
            winner, _ = run_episode(old_policy, new_policy, record=False)
            new_player = 2

        if winner == new_player:
            wins += 1.0
        elif winner == 0:
            wins += 0.5

    return wins / n_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",
                        type=str,
                        default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training on {device}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    net = PolicyValueNet(board_size=BOARD_SIZE,
                         num_blocks=NUM_BLOCKS,
                         channels=CHANNELS)
    trainer = Trainer(net, lr=LR, device=device)

    if args.resume:
        trainer.load(args.resume)
        # Infer starting iteration from filename (e.g. "checkpoints/iter_0012.pt" → 12)
        match = __import__("re").search(r"iter_(\d+)", args.resume)
        start_iter = int(match.group(1)) if match else 0
        print(
            f"Resumed from {args.resume} (continuing from iter {start_iter})")
    else:
        start_iter = 0

    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

    # The "best" policy starts as a copy of the current network
    best_net = copy.deepcopy(net)

    for iteration in range(start_iter + 1, start_iter + ITERATIONS + 1):
        t0 = time.time()

        # ── 1. Collect ──────────────────────────────────────────────────────
        current_policy = NeuralPolicy(net,
                                      temperature=TEMPERATURE,
                                      device=device)

        game_lengths = []
        for _ in range(GAMES_PER_ITER):
            _, transitions = run_episode(current_policy,
                                         current_policy,
                                         record=True)
            buffer.add(transitions)
            game_lengths.append(len(transitions))

        avg_length = sum(game_lengths) / len(game_lengths)

        # ── 2. Train ────────────────────────────────────────────────────────
        if len(buffer) < BATCH_SIZE:
            print(
                f"[iter {iteration:3d}] buffer too small ({len(buffer)}), skipping training"
            )
            continue

        total_p_loss = total_v_loss = 0.0
        for _ in range(TRAIN_STEPS_PER_ITER):
            states, policy_targets, value_targets, _ = buffer.sample(
                BATCH_SIZE)
            p_loss, v_loss = trainer.train_step(states, policy_targets,
                                                value_targets)
            total_p_loss += p_loss
            total_v_loss += v_loss

        avg_p = total_p_loss / TRAIN_STEPS_PER_ITER
        avg_v = total_v_loss / TRAIN_STEPS_PER_ITER

        # ── 3. Evaluate ─────────────────────────────────────────────────────
        new_policy = NeuralPolicy(net,
                                  temperature=EVAL_TEMPERATURE,
                                  device=device)
        old_policy = NeuralPolicy(best_net,
                                  temperature=EVAL_TEMPERATURE,
                                  device=device)
        win_rate = evaluate(new_policy, old_policy, EVAL_GAMES)

        if win_rate >= WIN_RATE_THRESHOLD:
            best_net = copy.deepcopy(net)
            updated = "✓ updated best"
        else:
            updated = "  (kept old best)"

        # ── 4. Save ─────────────────────────────────────────────────────────
        ckpt = CHECKPOINT_DIR / f"iter_{iteration:04d}.pt"
        trainer.save(str(ckpt))

        elapsed = time.time() - t0
        print(f"[iter {iteration:4d}] "
              f"p_loss={avg_p:.4f}  v_loss={avg_v:.4f}  "
              f"win_rate={win_rate:.2f}  "
              f"buf={len(buffer)}  avg_game={avg_length:.0f}  "
              f"t={elapsed:.1f}s  {updated}")

    print(
        "Training complete. Best weights are in the last checkpoint where '✓ updated best' appeared."
    )


if __name__ == "__main__":
    main()
