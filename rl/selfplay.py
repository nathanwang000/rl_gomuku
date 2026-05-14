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
from game import GameState
from policy import Policy
from policy import SmartPolicy as _SmartPolicy

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

    @torch.no_grad()
    def action_probs(self, state: GameState) -> dict:
        """Return softmaxed policy distribution as {(row, col): prob}."""
        _, dist, _, _ = self._select_with_policy(state)
        return {(r, c): dist[r * state.board.shape[0] + c].item() for r, c in state.valid_moves()}

    # select_move is inherited from Policy base class (samples from action_probs)
    # select_move_with_probs is also inherited

    # ------------------------------------------------------------------
    # Extended interface used by run_episode to also get the policy dist
    # (kept for training use — returns tensor dist + value prediction)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _select_with_policy(
        self, state: GameState
    ) -> Tuple[int, torch.Tensor, float, torch.Tensor]:
        """Returns (move_index, policy_distribution_tensor, value_pred, state_tensor) for training.

        Returns the encoded state tensor so callers don't need to re-encode.
        """
        x = self.net.encode_state(state).to(self.device)
        logits, value = self.net(x)
        logits = logits.squeeze(0)
        value_pred = value.item()

        H, W = state.board.shape
        valid = state.valid_moves()
        mask = torch.full((H * W,), float("-inf"), device=logits.device)
        for r, c in valid:
            mask[r * W + c] = 0.0
        logits = logits + mask

        if self.temperature == 0:
            dist = torch.zeros_like(logits)
            dist[int(logits.argmax())] = 1.0
            move_index = int(dist.argmax())
        else:
            dist = torch.softmax(logits / self.temperature, dim=0)
            move_index = int(torch.multinomial(dist, 1).item())

        return move_index, dist.cpu(), value_pred, x.squeeze(0).cpu()


def run_episode(
    policy1: Policy,
    policy2: Policy,
    record: bool = True,
    gamma: float = 1.0,
    lam: float = 1.0,
    teacher: Optional[Policy] = None,
) -> Tuple[Optional[int], List[Transition]]:
    """Play one complete game and return (winner, transitions).

    winner: 1, 2, or 0 (draw).
    transitions: empty list when record=False or when policies are not NeuralPolicy.

    teacher: policy queried at each step to provide the BC supervision target.
             Defaults to SmartPolicy(). Can be any Policy — e.g. an MCTSPolicy
             for AlphaZero-style distillation of search into the fast policy head.
    """
    state = GameState.new()
    episode = Episode()
    policies = {1: policy1, 2: policy2}
    # Teacher provides BC supervision targets; stateless so safe to instantiate once
    teacher = teacher if teacher is not None else _SmartPolicy()

    while state.winner is None:
        player = state.current_player
        policy = policies[player]

        if record and isinstance(policy, NeuralPolicy):
            move_index, dist, value_pred, state_tensor = policy._select_with_policy(state)
            H = state.board.shape[0]
            row, col = divmod(move_index, H)
            # Query teacher on the same state for the BC supervision target
            tr, tc = teacher.select_move(state)
            teacher_move_index = tr * H + tc
            episode.record(
                state=state_tensor,
                move_index=move_index,
                policy_target=dist,
                value_pred=value_pred,
                player=player,
                teacher_move_index=teacher_move_index,
            )
        else:
            row, col = policy.select_move(state)

        state.make_move(row, col)

    transitions = episode.finalise(state.winner, gamma=gamma, lam=lam) if record else []
    return state.winner, transitions
