"""Episode storage and replay buffer.

An Episode records every (state_tensor, move_index) pair from one game.
After the game ends, value targets are back-filled based on the outcome:
  winner's turns  → +1
  loser's turns   → -1
  draw            →  0

ReplayBuffer keeps the last `capacity` episodes and serves random mini-batches.
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

    def finalise(self, winner: Optional[int]) -> List[Transition]:
        """Back-fill value targets and return the completed transition list.

        winner: 1 or 2 for the winning player, 0 for draw, None for abandoned.
        """
        for t in self._transitions:
            if winner == 0 or winner is None:
                t.value_target = 0.0
            elif winner == t.player:
                t.value_target = 1.0
            else:
                t.value_target = -1.0
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
