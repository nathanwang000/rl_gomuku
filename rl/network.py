"""Policy-value network.

Input:  (batch, 3, H, W) float tensor
          channel 0 — current player's stones
          channel 1 — opponent's stones
          channel 2 — all-ones (constant "side to move" plane, keeps bias easy)

Outputs:
  policy_logits — (batch, H*W) raw logits over every cell
  value         — (batch,) scalar in [-1, 1], current player's win probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Two conv layers with a residual skip — the standard building block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels,
                               channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PolicyValueNet(nn.Module):
    """Shared ResNet backbone → policy head + value head.

    Deliberately small (num_blocks=4, channels=64) so it trains fast
    on a laptop CPU.  Increase for stronger play.
    """

    def __init__(self,
                 board_size: int,
                 num_blocks: int = 4,
                 channels: int = 64):
        super().__init__()
        self.board_size = board_size
        H = W = board_size

        # Stem: project 3 input channels to `channels`
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Shared residual tower
        self.tower = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)])

        # Policy head: 1×1 conv → flatten → logits over every cell
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * H * W, H * W),
        )

        # Value head: 1×1 conv → fc → tanh scalar
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(H * W, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, W) — returns (policy_logits, value)."""
        features = self.tower(self.stem(x))
        policy_logits = self.policy_head(features)  # (B, H*W)
        value = self.value_head(features).squeeze(-1)  # (B,)
        return policy_logits, value

    # ------------------------------------------------------------------
    # Convenience: encode a single GameState for inference
    # ------------------------------------------------------------------

    def encode_state(self, state) -> torch.Tensor:
        """Turn a GameState into a (1, 3, H, W) float tensor."""
        import numpy as np
        board = state.board  # (H, W) int8
        me = (board == state.current_player).astype(np.float32)
        opp = (board == 3 - state.current_player).astype(np.float32)
        turn = np.ones_like(me)
        tensor = torch.tensor(np.stack([me, opp,
                                        turn])).unsqueeze(0)  # (1,3,H,W)
        return tensor
