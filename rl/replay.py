"""Episode storage and replay buffer.

An Episode records every (state_tensor, move_index) pair from one game.
After the game ends, value targets are back-filled based on the outcome:
  winner's turns  → +γ^(T-1-t)   (1.0 for the final move, decaying backwards)
  loser's turns   → -γ^(T-1-t)
  draw            →  0

Discounting with γ<1 rewards winning fast: the winning move always gets ±1
and earlier moves get progressively smaller targets, naturally down-weighting
moves where causal attribution is weakest.

ReplayBuffer keeps the last `capacity` transitions and serves random mini-batches.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class Transition:
    """One (state, move) pair from a game, ready for training."""
    state: torch.Tensor  # (3, H, W) float
    move_index: int  # flat index = row * W + col
    policy_target: torch.Tensor  # (H*W,) — soft label from MCTS or one-hot
    value_target: float  # +1 / -1 / 0 filled in after game ends
    player: int  # 1 or 2 — who made this move


class Episode:
    """Accumulates transitions during a game, then finalises value targets."""

    def __init__(self):
        self._transitions: List[Transition] = []

    def record(
        self,
        state: torch.Tensor,
        move_index: int,
        policy_target: torch.Tensor,
        player: int,
    ) -> None:
        self._transitions.append(
            Transition(
                state=state,
                move_index=move_index,
                policy_target=policy_target,
                value_target=0.0,  # filled later
                player=player,
            ))

    def finalise(self, winner: Optional[int], gamma: float = 1.0) -> List[Transition]:
        """Back-fill discounted value targets and return the completed transition list.

        winner: 1 or 2 for the winning player, 0 for draw, None for abandoned.
        gamma:  discount factor. 1.0 = flat outcome; <1.0 = rewards winning fast.
                The last move always gets ±1; earlier moves get γ^(T-1-t).
        """
        T = len(self._transitions)
        for i, t in enumerate(self._transitions):
            if winner == 0 or winner is None:
                t.value_target = 0.0
            else:
                sign = 1.0 if winner == t.player else -1.0
                t.value_target = sign * (gamma ** (T - 1 - i))
        return self._transitions


class ReplayBuffer:
    """Fixed-capacity circular buffer of Transitions, sampled uniformly."""

    def __init__(self, capacity: int = 50_000):
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def add(self, transitions: List[Transition]) -> None:
        self._buf.extend(transitions)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (states, policy_targets, value_targets, move_indices) tensors."""
        batch = random.sample(self._buf, min(batch_size, len(self._buf)))

        states = torch.stack([t.state for t in batch])  # (B,3,H,W)
        policy_targets = torch.stack([t.policy_target
                                      for t in batch])  # (B,H*W)
        value_targets = torch.tensor([t.value_target for t in batch],
                                     dtype=torch.float32)  # (B,)
        move_indices = torch.tensor([t.move_index for t in batch],
                                    dtype=torch.long)  # (B,)

        return states, policy_targets, value_targets, move_indices

    def __len__(self) -> int:
        return len(self._buf)
