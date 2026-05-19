"""Training step logic.

Total loss = policy_coeff * policy_loss + value_coeff * value_loss + bc_coeff * bc_loss

  policy_loss: REINFORCE — -log_prob(chosen_move) * advantage
               advantage = G^λ_t - V(s_t)  (TD(λ) return minus value baseline)
  value_loss:  MSE between V(s_t) and G^λ_t
  bc_loss:     -log_prob(teacher_move)  (cross-entropy vs teacher policy)
               bc_coeff=0.0 → pure RL (identical to no-BC behaviour)
               bc_coeff>0   → nudges policy toward teacher policy's moves
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from rl.network import PolicyValueNet


class Trainer:
    """Owns the network and optimizer; exposes a single train_step()."""

    def __init__(
        self,
        net: PolicyValueNet,
        lr: float = 1e-3,
        device: str = "cpu",
        policy_coeff: float = 1.0,
        value_coeff: float = 1.0,
        bc_coeff: float = 0.0,
        use_value_baseline: bool = False,
    ):
        self.net = net.to(device)
        self.device = device
        self.policy_coeff = policy_coeff
        self.value_coeff  = value_coeff
        self.bc_coeff     = bc_coeff
        self.use_value_baseline = use_value_baseline
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def train_step(
            self,
            states: torch.Tensor,             # (B, 2, H, W)
            move_indices: torch.Tensor,       # (B,) — flat index of chosen move
            value_targets: torch.Tensor,      # (B,) — TD(λ) return
            teacher_move_indices: torch.Tensor,# (B,) — teacher policy's move (BC target)
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """One gradient step. Returns (policy_loss, value_loss, bc_loss, extras).

        extras keys:
          grad_norm  — L2 norm of all parameter gradients (useful for debugging).
        """
        states               = states.to(self.device)
        move_indices         = move_indices.to(self.device)
        value_targets        = value_targets.to(self.device)
        teacher_move_indices = teacher_move_indices.to(self.device)

        self.net.train()
        policy_logits, value = self.net(states)
        log_probs = F.log_softmax(policy_logits, dim=-1)  # (B, H*W)

        # REINFORCE: weight log-prob of chosen move by TD(λ) advantage
        if self.use_value_baseline:
            advantage = value_targets - value.detach()  # actor-critic: reduce variance
        else:
            advantage = value_targets                   # pure REINFORCE: no baseline
        chosen_log_probs = log_probs.gather(1, move_indices.unsqueeze(1)).squeeze(1)  # (B,)
        policy_loss      = -(chosen_log_probs * advantage).mean()

        # Value loss: plain MSE against {-1, 0, +1} discounted by time with TD(\lambda) targets
        value_loss = F.mse_loss(value, value_targets)

        # Behavioural cloning: cross-entropy vs teacher policy's move
        teacher_log_probs = log_probs.gather(1, teacher_move_indices.unsqueeze(1)).squeeze(1)  # (B,)
        bc_loss = -teacher_log_probs.mean()

        loss = self.policy_coeff * policy_loss + self.value_coeff * value_loss + self.bc_coeff * bc_loss
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=float("inf")).item()
        self.optimizer.step()

        extras = {"grad_norm": grad_norm}
        return policy_loss.item(), value_loss.item(), bc_loss.item(), extras

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
