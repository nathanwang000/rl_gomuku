"""Self-play data collection.

NeuralPolicy is the bridge between the RL network and the existing Policy
interface — it can be dropped into any game that uses select_move(state).

run_episode() plays one complete game between two policies and, if record=True,
returns a list of Transitions ready for the replay buffer.  The game logic is
untouched — we only call state.valid_moves(), state.make_move(), and read
state.winner / state.current_player.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from game import GameState
from policy import Policy

from rl.network import PolicyValueNet
from rl.replay import Episode, Transition


class NeuralPolicy(Policy):
    """Wraps PolicyValueNet so it can be used anywhere a Policy is expected.

    Move selection:
      - Sample from the softmaxed policy distribution (exploration during training).
      - Set temperature=0 (argmax) for evaluation / deterministic play.
      - Illegal moves are masked to -inf before softmax so the network never
        selects an occupied cell.
    """

    def __init__(self,
                 net: PolicyValueNet,
                 temperature: float = 1.0,
                 device: str = "cpu"):
        self.net = net
        self.temperature = temperature
        self.device = device

    def select_move(self, state: GameState) -> Tuple[int, int]:
        move_index, _ = self._select_with_policy(state)
        H = state.board.shape[0]
        return divmod(move_index, H)

    # ------------------------------------------------------------------
    # Extended interface used by run_episode to also get the policy dist
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _select_with_policy(self,
                            state: GameState) -> Tuple[int, torch.Tensor]:
        """Returns (move_index, policy_distribution) for recording."""
        x = self.net.encode_state(state).to(self.device)  # (1, 3, H, W)
        logits, _ = self.net(x)  # (1, H*W)
        logits = logits.squeeze(0)  # (H*W,)

        # Mask illegal moves
        H, W = state.board.shape
        valid = state.valid_moves()
        mask = torch.full((H * W, ), float("-inf"), device=logits.device)
        for r, c in valid:
            mask[r * W + c] = 0.0
        logits = logits + mask

        if self.temperature == 0:
            move_index = int(logits.argmax())
            dist = F.softmax(logits, dim=0)
        else:
            dist = F.softmax(logits / self.temperature, dim=0)
            move_index = int(torch.multinomial(dist, 1).item())

        return move_index, dist


def run_episode(
    policy1: Policy,
    policy2: Policy,
    record: bool = True,
    gamma: float = 1.0,
) -> Tuple[Optional[int], List[Transition]]:
    """Play one complete game and return (winner, transitions).

    winner: 1, 2, or 0 (draw).
    transitions: empty list when record=False or when policies are not NeuralPolicy.

    The two policies play alternately: policy1 plays as player 1 (black),
    policy2 plays as player 2 (white).
    """
    state = GameState.new()
    episode = Episode()
    policies = {1: policy1, 2: policy2}

    while state.winner is None:
        player = state.current_player
        policy = policies[player]

        if record and isinstance(policy, NeuralPolicy):
            move_index, dist = policy._select_with_policy(state)
            state_tensor = policy.net.encode_state(state).squeeze(0)  # (3,H,W)
            H = state.board.shape[0]
            row, col = divmod(move_index, H)
            episode.record(
                state=state_tensor,
                move_index=move_index,
                policy_target=dist.cpu(),
                player=player,
            )
        else:
            row, col = policy.select_move(state)

        state.make_move(row, col)

    transitions = episode.finalise(state.winner, gamma=gamma) if record else []
    return state.winner, transitions
