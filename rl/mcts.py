"""Monte Carlo Tree Search policy.

A single MCTSNode represents one game position.  From a root node, we run
`simulations` iterations of the standard PUCT loop:

  Select   — walk the tree, picking children by UCB score until a leaf
  Expand   — create all children of the leaf using the network's policy prior
  Evaluate — call the network's value head to estimate the outcome
  Backup   — propagate the value back up, negating at each alternating-player step

After all simulations, move selection uses visit counts (not Q-values), which
gives a smoother distribution and is more robust to outliers.

MCTSPolicy wraps this in the standard Policy interface so it can be used
anywhere a Policy is expected — including as the `teacher` in run_episode().
"""

import copy
import math
from typing import Dict, Optional, Tuple

import torch
from game import GameState
from policy import Policy

from rl.network import PolicyValueNet

# ── PUCT exploration constant — higher = more exploration ─────────────────────
C_PUCT = 1.5


class MCTSNode:
    """One node in the search tree, corresponding to one game state."""

    __slots__ = ("state", "prior", "parent", "children", "visit_count",
                 "value_sum")

    def __init__(self, state: GameState, prior: float,
                 parent: Optional["MCTSNode"]):
        self.state = state
        self.prior = prior  # P(s, a) from the network — set by parent on creation
        self.parent = parent
        self.children: Dict[Tuple[int, int], "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0  # sum of backed-up values (current-player perspective)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def q_value(self) -> float:
        """Mean backed-up value; 0 for unvisited nodes."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb_score(self, parent_visits: int) -> float:
        """PUCT score: Q + C_PUCT * P * sqrt(N_parent) / (1 + N_child)."""
        exploration = C_PUCT * self.prior * math.sqrt(parent_visits) / (
            1 + self.visit_count)
        return self.q_value + exploration


# ── Core MCTS functions ───────────────────────────────────────────────────────


def _select(node: MCTSNode) -> MCTSNode:
    """Walk down the tree, choosing the highest UCB child at each step."""
    while not node.is_leaf and node.state.winner is None:
        parent_visits = node.visit_count
        node = max(node.children.values(),
                   key=lambda c: c.ucb_score(parent_visits))
    return node


def _expand_and_evaluate(node: MCTSNode, net: PolicyValueNet,
                         device: str) -> float:
    """Expand a leaf using the network's priors; return the network's value estimate.

    If the node is terminal, returns the exact outcome (+1 / -1 / 0) instead.
    """
    if node.state.winner is not None:
        if node.state.winner == 0:
            return 0.0
        # The player who just moved won — that's the opponent of current_player
        return -1.0  # bad for the player whose turn it now is

    x = net.encode_state(node.state).to(device)
    with torch.no_grad():
        logits, value = net(x)

    # Convert policy logits to a masked probability distribution
    H, W = node.state.board.shape
    valid = node.state.valid_moves()
    mask = torch.full((H * W, ), float("-inf"), device=logits.device)
    for r, c in valid:
        mask[r * W + c] = 0.0
    priors = torch.softmax(logits.squeeze(0) + mask, dim=0).cpu()

    # Create one child per legal move
    for r, c in valid:
        child_state = copy.deepcopy(node.state)
        child_state.make_move(r, c)
        prior = priors[r * W + c].item()
        node.children[(r, c)] = MCTSNode(child_state, prior=prior, parent=node)

    return value.item()


def _backup(node: MCTSNode, value: float) -> None:
    """Propagate value up, negating at each step (alternating-player convention)."""
    while node is not None:
        node.visit_count += 1
        node.value_sum += value
        value = -value  # flip perspective for the parent
        node = node.parent


# ── Public policy class ───────────────────────────────────────────────────────


class MCTSPolicy(Policy):
    """Wraps PolicyValueNet + MCTS in the standard Policy interface.

    Can be used as:
      - A playing policy (temperature=1.0 for training, 0.0 for eval)
      - A teacher for BC supervision (typically more simulations than the player)

    Args:
        net:         The shared PolicyValueNet.
        simulations: Number of MCTS rollouts per move.
        temperature: Controls how visit counts map to a move distribution.
                     1.0 → proportional to visit count (exploratory)
                     0.0 → argmax (deterministic, for evaluation)
        device:      Torch device string.
    """

    def __init__(
        self,
        net: PolicyValueNet,
        simulations: int = 100,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        self.net = net
        self.simulations = simulations
        self.temperature = temperature
        self.device = device

    def select_move(self, state: GameState) -> Tuple[int, int]:
        visit_counts, moves = self._search(state)

        if self.temperature == 0:
            best = max(range(len(moves)), key=lambda i: visit_counts[i])
            return moves[best]

        # Sample proportional to visit_count^(1/T)
        counts = torch.tensor(visit_counts, dtype=torch.float32)
        probs = (counts**(1.0 / self.temperature))
        probs = probs / probs.sum()
        idx = int(torch.multinomial(probs, 1).item())
        return moves[idx]

    def visit_distribution(self, state: GameState) -> Tuple[list, list]:
        """Return (moves, visit_counts) for use as a policy target (e.g. for BC)."""
        visit_counts, moves = self._search(state)
        return moves, visit_counts

    def _search(self, state: GameState) -> Tuple[list, list]:
        """Run MCTS from `state` and return (visit_counts, moves)."""
        root = MCTSNode(copy.deepcopy(state), prior=1.0, parent=None)
        # Expand root immediately so we always have children to pick from
        _expand_and_evaluate(root, self.net, self.device)
        root.visit_count = 1

        for _ in range(self.simulations):
            leaf = _select(root)
            value = _expand_and_evaluate(leaf, self.net, self.device)
            _backup(leaf, value)

        moves = list(root.children.keys())
        visit_counts = [root.children[m].visit_count for m in moves]
        return visit_counts, moves
