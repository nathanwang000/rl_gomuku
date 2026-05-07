"""Player policies — the interface for AI agents."""

import random
from abc import ABC, abstractmethod
from typing import Tuple
from game import GameState


class Policy(ABC):
    """Base class for all policies (human or AI)."""

    @abstractmethod
    def select_move(self, state: GameState) -> Tuple[int, int]:
        """Given a game state, return (row, col) to play."""
        ...


class RandomPolicy(Policy):
    """Picks a random valid move — dummy baseline."""

    def select_move(self, state: GameState) -> Tuple[int, int]:
        moves = state.valid_moves()
        return random.choice(moves)


class HumanPolicy(Policy):
    """Placeholder — moves come from the frontend, not computed here."""

    def select_move(self, state: GameState) -> Tuple[int, int]:
        raise NotImplementedError("Human moves come from the UI")


# Registry for easy lookup
POLICIES = {
    "human": HumanPolicy,
    "random": RandomPolicy,
}


def get_policy(name: str) -> Policy:
    return POLICIES[name]()
