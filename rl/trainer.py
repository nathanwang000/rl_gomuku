"""Training step logic.

Loss = policy_loss + value_loss
  policy_loss: cross-entropy between network policy and target distribution
  value_loss:  mean-squared error between network value and game outcome

Both losses are equally weighted — no hyperparameter needed for balance.
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
            states: torch.Tensor,  # (B, 3, H, W)
            policy_targets: torch.Tensor,  # (B, H*W) — soft distribution
            value_targets: torch.Tensor,  # (B,)     — +1 / -1 / 0
    ) -> Tuple[float, float]:
        """One gradient step. Returns (policy_loss, value_loss) as Python floats."""
        states = states.to(self.device)
        policy_targets = policy_targets.to(self.device)
        value_targets = value_targets.to(self.device)

        self.net.train()
        policy_logits, value = self.net(states)

        # Policy loss: KL-divergence reduces to cross-entropy when targets are soft
        # log_softmax + kl_div gives a clean, numerically stable form
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = F.kl_div(log_probs,
                               policy_targets,
                               reduction="batchmean")

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
