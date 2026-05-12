"""Episode storage and replay buffer.

An Episode records every (state_tensor, move_index, value_pred) tuple per move.
After the game ends, value targets are computed via the TD(λ) backward pass:

  G^λ_{T-1} = outcome               (terminal move always gets raw ±1 and 0 for draw)
  G^λ_t     = -γ·[(1-λ)·V_{t+1} + λ·G^λ_{t+1}]   for t < T-1
               ^
               negation: V_{t+1} and G^λ_{t+1} are from the OPPONENT's perspective
               (players alternate), so we negate to get the current player's return.

  λ=1  →  pure MC:   G^λ_t = -γ·G^λ_{t+1}   (ignores V entirely, high variance)
  λ=0  →  pure TD(0): G^λ_t = -γ·V_{t+1}    (fully bootstrapped, high bias)
  λ∈(0,1) →  bias-variance trade-off between the two

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
    state:        torch.Tensor   # (2, H, W) float
    move_index:   int            # flat index = row * W + col
    policy_target: torch.Tensor  # (H*W,) — unused placeholder (kept for buffer API)
    value_pred:   float          # network's V estimate at this state (for TD bootstrap)
    value_target: float          # filled in by finalise()
    player:       int            # 1 or 2 — who made this move


class Episode:
    """Accumulates transitions during a game, then finalises value targets."""

    def __init__(self):
        self._transitions: List[Transition] = []

    def record(
        self,
        state: torch.Tensor,
        move_index: int,
        policy_target: torch.Tensor,
        value_pred: float,
        player: int,
    ) -> None:
        self._transitions.append(
            Transition(
                state=state,
                move_index=move_index,
                policy_target=policy_target,
                value_pred=value_pred,
                value_target=0.0,  # filled by finalise()
                player=player,
            ))

    def finalise(
        self,
        winner: Optional[int],
        gamma: float = 1.0,
        lam: float = 1.0,
    ) -> List[Transition]:
        """Compute TD(λ) returns and store them as value_target.

        Values are always encoded from the CURRENT PLAYER's perspective at each
        state: +1 means "I am winning", -1 means "I am losing".  Because players
        alternate every move, V(s_{t+1}) is from the OPPONENT's point of view.
        To express it as a return for the player at step t, we must negate it.

        Backward pass:
          G^λ_{T-1} = outcome            # terminal: +1 winner, -1 loser, 0 draw
          G^λ_t = -γ · [(1-λ)·V_{t+1} + λ·G^λ_{t+1}]   for t < T-1
                   ^
                   negation flips opponent-perspective → current-player-perspective

        λ controls the MC/TD mix applied to every non-terminal step:
          λ=1 → pure MC: G^λ_t = -γ · G^λ_{t+1}  (ignores V entirely)
          λ=0 → pure TD(0): G^λ_t = -γ · V_{t+1}  (fully bootstrapped)
        """
        if not self._transitions:
            return []

        last = self._transitions[-1]
        if winner == 0 or winner is None:
            g = 0.0
        else:
            g = 1.0 if winner == last.player else -1.0
        last.value_target = g

        for t in range(len(self._transitions) - 2, -1, -1):
            tr   = self._transitions[t]
            next_tr = self._transitions[t + 1]
            # next_tr's value_pred and g are from the opponent's perspective → negate
            g = -gamma * ((1.0 - lam) * next_tr.value_pred + lam * g)
            tr.value_target = g

        return self._transitions


class ReplayBuffer:
    """Fixed-capacity circular buffer of Transitions, sampled uniformly."""

    def __init__(self, capacity: int = 50_000):
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def add(self, transitions: List[Transition]) -> None:
        self._buf.extend(transitions)

    def clear(self) -> None:
        self._buf.clear()

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
