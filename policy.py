"""Player policies — the interface for AI agents."""

import random
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from game import GameState, BOARD_SIZE, WIN_LENGTH


class Policy(ABC):
    """Base class for all policies (human or AI)."""

    @abstractmethod
    def action_probs(self, state: GameState) -> dict:
        """Return a probability distribution over valid moves as {(row, col): prob}.

        Probabilities must be non-negative and sum to 1.
        Temperature and any other policy-specific shaping should be applied here.
        """
        ...

    def select_move(self, state: GameState) -> Tuple[int, int]:
        """Sample a move from action_probs. Override only for special cases."""
        move, _ = self.select_move_with_probs(state)
        return move

    def select_move_with_probs(self, state: GameState) -> Tuple[Tuple[int, int], dict]:
        """Return (move, action_probs dict) in a single call, avoiding double computation."""
        probs = self.action_probs(state)
        moves = list(probs.keys())
        weights = list(probs.values())
        move = random.choices(moves, weights=weights)[0]
        return move, probs


class RandomPolicy(Policy):
    """Picks a random valid move — dummy baseline."""

    def action_probs(self, state: GameState) -> dict:
        moves = state.valid_moves()
        p = 1.0 / len(moves)
        return {m: p for m in moves}


class HumanPolicy(Policy):
    """Placeholder — moves come from the frontend, not computed here."""

    def action_probs(self, state: GameState) -> dict:
        raise NotImplementedError("Human moves come from the UI")

    def select_move_with_probs(self, state: GameState) -> Tuple[Tuple[int, int], dict]:
        raise NotImplementedError("Human moves come from the UI")


class SmartPolicy(Policy):
    """Heuristic policy that understands blocking and attacking.

    Priority order:
    1. Win immediately (5 in a row)
    2. Block opponent from winning (opponent has open 4)
    3. Create an open 4 (unstoppable next turn)
    4. Block opponent's open 3
    5. Create an open 3
    6. Score all moves by local heuristic and pick the best
    """

    DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

    def action_probs(self, state: GameState) -> dict:
        """Action probability distribution reflecting priority rules.

        - Returns {forced_move: 1.0} when an immediate win or block exists.
        - Otherwise returns normalized heuristic scores over all candidates.
        This means select_move and select_move_with_probs are fully covered
        by the base class — no overrides needed.
        """
        me = state.current_player
        opp = 3 - me
        board = state.board

        scores = self._compute_scores(state)
        candidates = list(scores.keys())

        for r, c in candidates:
            # priority 1: can I win immediately?
            if self._would_win(board, r, c, me):
                return {(r, c): 1.0}
        for r, c in candidates:
            # priority 2: do I need to block opponent's win?
            if self._would_win(board, r, c, opp):
                return {(r, c): 1.0}

        # priority 3+: uniform choose among highest-scoring moves (could be multiple with same score)
        vals = np.array(list(scores.values()), dtype=float)
        best = vals.max()
        uniform = (vals == best).astype(float)
        uniform /= uniform.sum()
        return {m: float(p) for m, p in zip(scores.keys(), uniform)}

    def _compute_scores(self, state: GameState) -> dict:
        """Raw heuristic scores for all candidate moves."""
        me = state.current_player
        opp = 3 - me
        board = state.board

        # Only consider moves near existing stones for efficiency
        candidates = self._get_candidate_moves(board)
        if not candidates:
            # empty board — no heuristics, just pick center
            return {(BOARD_SIZE // 2, BOARD_SIZE // 2): 1}
        return {
            (r, c): self._score_move(board, r, c, me, opp)
            for r, c in candidates
        }

    def _get_candidate_moves(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """Return empty cells within distance 2 of any existing stone."""
        occupied = np.argwhere(board != 0)
        if len(occupied) == 0:
            return []

        candidates = set()
        for r, c in occupied:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == 0:
                        candidates.add((nr, nc))
        return list(candidates)

    def _would_win(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """Check if placing `player` at (row, col) makes 5 in a row."""
        for dr, dc in self.DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == player:
                    count += 1
                    r += dr * sign
                    c += dc * sign
            if count >= WIN_LENGTH:
                return True
        return False

    def _count_line(self, board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int):
        """Count consecutive stones and open ends for `player` through (row, col) in direction (dr, dc).

        Assumes (row, col) is empty and would be filled by player.
        Returns (length, open_ends) where open_ends is 0, 1, or 2.
        """
        count = 1
        open_ends = 0

        # Positive direction
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == player:
            count += 1
            r += dr
            c += dc
        # Check if end is open
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == 0:
            open_ends += 1

        # Negative direction
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == player:
            count += 1
            r -= dr
            c -= dc
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == 0:
            open_ends += 1

        return count, open_ends

    def _score_move(self, board: np.ndarray, row: int, col: int, me: int, opp: int) -> int:
        """Heuristic score for placing `me` at (row, col)."""
        score = 0

        for dr, dc in self.DIRECTIONS:
            # Evaluate for me (attack)
            my_len, my_open = self._count_line(board, row, col, dr, dc, me)
            score += self._pattern_score(my_len, my_open, attack=True)

            # Evaluate for opponent (defense value of this cell)
            opp_len, opp_open = self._count_line(board, row, col, dr, dc, opp)
            score += self._pattern_score(opp_len, opp_open, attack=False)

        return score

    def _pattern_score(self, length: int, open_ends: int, attack: bool) -> int:
        """Score a line pattern. Attack patterns are slightly more valuable than defense."""
        if open_ends == 0 and length < WIN_LENGTH:
            return 0  # Dead line, no value

        # Multiplier: prefer attacking slightly over blocking
        mult = 1.1 if attack else 1.0

        if length >= WIN_LENGTH:
            return int(1000000 * mult)
        if length == 4:
            if open_ends == 2:
                return int(100000 * mult)  # Open four — nearly unstoppable
            else:
                return int(10000 * mult)   # Half-open four
        if length == 3:
            if open_ends == 2:
                return int(5000 * mult)    # Open three — very threatening
            else:
                return int(500 * mult)     # Half-open three
        if length == 2:
            if open_ends == 2:
                return int(200 * mult)     # Open two
            else:
                return int(50 * mult)      # Half-open two
        if length == 1:
            if open_ends == 2:
                return int(10 * mult)
            else:
                return int(3 * mult)

        return 0


# Registry for easy lookup
POLICIES = {
    "human": HumanPolicy,
    "random": RandomPolicy,
    "smart": SmartPolicy,
}


def get_policy(name: str) -> Policy:
    return POLICIES[name]()
