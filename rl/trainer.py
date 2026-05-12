"""Training step logic.

Loss = policy_loss + value_loss
  policy_loss: REINFORCE — -log_prob(chosen_move) * advantage
               advantage = value_target - value.detach() (actor-critic baseline)
  value_loss:  MSE between predicted value and game outcome

The baseline subtracts the network's own value estimate, reducing gradient
variance while keeping the policy gradient unbiased.
"""

from typing import Tuple

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
    ):
        self.net = net.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def train_step(
            self,
            states: torch.Tensor,        # (B, 3, H, W)
            move_indices: torch.Tensor,  # (B,)     — flat index of chosen move
            value_targets: torch.Tensor, # (B,)     — +1 / -1 / 0
    ) -> Tuple[float, float]:
        """One gradient step. Returns (policy_loss, value_loss) as Python floats."""
        states        = states.to(self.device)
        move_indices  = move_indices.to(self.device)
        value_targets = value_targets.to(self.device)

        self.net.train()
        policy_logits, value = self.net(states)

        # REINFORCE policy loss
        # advantage = outcome - baseline; baseline = value estimate (detached so
        # it doesn't pull gradients through the value head via the policy loss)
        # advantage = value_targets - value.detach()                          # (B,)
        advantage = value_targets # vannilla version without baseline --- IGNORE ---
        log_probs = F.log_softmax(policy_logits, dim=-1)                   # (B, H*W)
        chosen_log_probs = log_probs.gather(1, move_indices.unsqueeze(1)).squeeze(1)  # (B,)
        policy_loss = -(chosen_log_probs * advantage).mean()

        # Value loss: plain MSE against {-1, 0, +1} targets
        value_loss = F.mse_loss(value, value_targets)

        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item()

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
