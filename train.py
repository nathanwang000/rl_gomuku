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

ON_POLICY = True  # whether to clear replay buffer each iteration (True = on-policy, False = off-policy)
ITERATIONS = 50  # outer training iterations
GAMES_PER_ITER = 20  # self-play games collected each iteration
TRAIN_STEPS_PER_ITER = 300  # gradient steps taken each iteration
BATCH_SIZE = 256

EVAL_GAMES = 2  # games played to compare new vs old policy
WIN_RATE_THRESHOLD = 0.55  # replace old policy only if new wins > this fraction

LR = 1e-3
NUM_BLOCKS = 4  # ResNet depth
CHANNELS = 64  # ResNet width
TEMPERATURE = 1.0  # exploration temperature during self-play
EVAL_TEMPERATURE = 0.0  # deterministic during evaluation
GAMMA = 1.0 # 0.95  # discount factor: rewards winning fast; 1.0 = flat outcome
LAMBDA = 1.0  # TD(λ) mixing: 1.0 = pure MC, 0.0 = pure TD(0)
BC_COEFF = 1  # 1.0      # behavioural cloning weight: 0.0 = pure RL, higher = more imitation of teacher
POLICY_COEFF = 0  # 1.0  # set to 0.0 to disable REINFORCE loss (e.g. to debug BC or value alone)
VALUE_COEFF = 1.0  # 1.0   # set to 0.0 to disable value loss
# Teacher policy for BC supervision. Swap to MCTSPolicy(net, ...) for AlphaZero-style distillation.
TEACHER = None  # None → defaults to SmartPolicy() inside run_episode

# ── Debug flags ───────────────────────────────────────────────────────────────
USE_VALUE_BASELINE = False  # True  → advantage = value_targets - V(s) (standard actor-critic)
# False → advantage = value_targets (pure REINFORCE, no baseline)
DEBUG_TARGETS = True  # print value_target mean/std/min/max each iter (sanity-check λ=1 → ±1)
DEBUG_ENTROPY = True  # print mean policy entropy each iter (should decrease as policy learns)
DEBUG_GRADNORM = True  # print gradient norm after each train step (detect vanishing/exploding)
DEBUG_VALUE_CORR = True  # print Pearson r(value_pred, value_target) — VALUE_COEFF>0 needed to see this rise

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
    trainer = Trainer(net,
                      lr=LR,
                      device=device,
                      policy_coeff=POLICY_COEFF,
                      value_coeff=VALUE_COEFF,
                      bc_coeff=BC_COEFF,
                      use_value_baseline=USE_VALUE_BASELINE)

    if args.resume:
        trainer.load(args.resume)
        # Infer starting iteration from filename (e.g. "checkpoints/iter_0012.pt" → 12)
        match = __import__("re").search(r"iter_(\d+)", args.resume)
        start_iter = int(match.group(1)) if match else 0
        print(
            f"Resumed from {args.resume} (continuing from iter {start_iter})")
    else:
        start_iter = 0

    buffer = ReplayBuffer()

    # The "best" policy starts as a copy of the current network
    best_net = copy.deepcopy(net)

    for iteration in range(start_iter + 1, start_iter + ITERATIONS + 1):
        t0 = time.time()

        # ── 1. Collect ──────────────────────────────────────────────────────
        # Clear buffer so we only train on data from the current policy (on-policy)
        if ON_POLICY:
            buffer.clear()
        current_policy = NeuralPolicy(net,
                                      temperature=TEMPERATURE,
                                      device=device)

        game_lengths = []
        all_value_targets: list[float] = []
        for _ in range(GAMES_PER_ITER):
            _, transitions = run_episode(current_policy,
                                         current_policy,
                                         record=True,
                                         gamma=GAMMA,
                                         lam=LAMBDA,
                                         teacher=TEACHER)
            buffer.add(transitions)
            game_lengths.append(len(transitions))
            all_value_targets.extend(t.value_target for t in transitions)

        avg_length = sum(game_lengths) / len(game_lengths)

        # ── 2. Train ────────────────────────────────────────────────────────
        if len(buffer) < BATCH_SIZE:
            print(
                f"[iter {iteration:3d}] buffer too small ({len(buffer)}), skipping training"
            )
            continue

        total_p_loss = total_v_loss = total_bc_loss = 0.0
        total_grad_norm = 0.0
        first_p, first_v, first_bc = None, None, None
        last_p, last_v, last_bc = None, None, None
        for step in range(TRAIN_STEPS_PER_ITER):
            states, _, value_targets, move_indices, teacher_move_indices = buffer.sample(
                BATCH_SIZE)
            p_loss, v_loss, bc_loss, extras = trainer.train_step(
                states, move_indices, value_targets, teacher_move_indices)
            total_p_loss += p_loss
            total_v_loss += v_loss
            total_bc_loss += bc_loss
            total_grad_norm += extras["grad_norm"]
            if step == 0:
                first_p, first_v, first_bc = p_loss, v_loss, bc_loss
            last_p, last_v, last_bc = p_loss, v_loss, bc_loss

        avg_p = total_p_loss / TRAIN_STEPS_PER_ITER
        avg_v = total_v_loss / TRAIN_STEPS_PER_ITER
        avg_bc = total_bc_loss / TRAIN_STEPS_PER_ITER
        avg_grad_norm = total_grad_norm / TRAIN_STEPS_PER_ITER

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
        p_arrow = '↓' if last_p < first_p else '↑'
        v_arrow = '↓' if last_v < first_v else '↑'
        bc_arrow = '↓' if last_bc < first_bc else '↑'
        print(f"[iter {iteration:4d}] "
              f"p={avg_p:.2f}({p_arrow}{first_p:.2f}→{last_p:.2f})  "
              f"v={avg_v:.4f}({v_arrow}{first_v:.4f}→{last_v:.4f})  "
              f"bc={avg_bc:.2f}({bc_arrow}{first_bc:.2f}→{last_bc:.2f})  "
              f"win_rate={win_rate:.2f}  "
              f"buf={len(buffer)}  avg_game={avg_length:.0f}  "
              f"t={elapsed:.1f}s  {updated}")

        if DEBUG_TARGETS and all_value_targets:
            import statistics
            vt = all_value_targets
            print(f"  [dbg targets ] mean={statistics.mean(vt):+.3f}  "
                  f"std={statistics.stdev(vt):.3f}  "
                  f"min={min(vt):+.3f}  max={max(vt):+.3f}  "
                  f"n={len(vt)}")
            # |G^λ_t| ≤ 1 always (discounted sum of rewards ∈ {-1,0,+1} with γ≤1)
            bad = [v for v in vt if abs(v) > 1.0 + 1e-5]
            if bad:
                print(
                    f"  [dbg targets ] WARNING: {len(bad)}/{len(vt)} targets outside [-1,1] "
                    f"(max={max(abs(v) for v in bad):.6f}) — likely a bug in finalise()"
                )
            else:
                print(f"  [dbg targets ] OK: all {len(vt)} targets in [-1,1]")
            if LAMBDA == 1.0 and GAMMA == 1.0:
                bad01 = [
                    v for v in vt if abs(v) > 1e-5 and abs(abs(v) - 1.0) > 1e-5
                ]
                if bad01:
                    print(
                        f"  [dbg targets ] WARNING: {len(bad01)}/{len(vt)} targets not in {{-1,0,+1}} "
                        f"with λ=γ=1 — every target should be exactly ±1 or 0")
                else:
                    print(
                        f"  [dbg targets ] OK: all {len(vt)} targets are exactly ±1 or 0 with λ=γ=1")

        if DEBUG_ENTROPY:
            # sample one batch and compute mean entropy of the masked policy
            s_sample, *_ = buffer.sample(BATCH_SIZE)
            s_sample = s_sample.to(device)
            net.eval()
            with torch.no_grad():
                logits_s, _ = net(s_sample)
                probs = torch.softmax(logits_s, dim=-1)
                entropy = -(probs *
                            torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            net.train()
            max_entropy = torch.log(
                torch.tensor(BOARD_SIZE * BOARD_SIZE,
                             dtype=torch.float)).item()
            print(
                f"  [dbg entropy ] mean_H={entropy:.3f}  (uniform={max_entropy:.3f})"
            )

        if DEBUG_GRADNORM:
            print(f"  [dbg gradnorm] avg_grad_norm={avg_grad_norm:.4f}")

        if DEBUG_VALUE_CORR:
            s_sample, _, vt_sample, *_ = buffer.sample(min(1024, len(buffer)))
            s_sample = s_sample.to(device)
            net.eval()
            with torch.no_grad():
                _, vpred = net(s_sample)
            net.train()
            vpred_np = vpred.squeeze(-1).cpu().numpy()
            vtgt_np = vt_sample.numpy()
            n = len(vpred_np)
            if n > 1:
                vp_mean, vt_mean = vpred_np.mean(), vtgt_np.mean()
                cov = ((vpred_np - vp_mean) * (vtgt_np - vt_mean)).mean()
                denom = vpred_np.std() * vtgt_np.std()
                r = cov / denom if denom > 1e-8 else 0.0
                print(
                    f"  [dbg val_corr] Pearson r(V_pred, V_target)={r:+.3f}  "
                    f"V_pred_mean={vp_mean:+.3f}")

    print(
        "Training complete. Best weights are in the last checkpoint where '✓ updated best' appeared."
    )


if __name__ == "__main__":
    main()
