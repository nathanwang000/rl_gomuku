"""Gomoku game engine — pure logic, no IO."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

BOARD_SIZE = 15
WIN_LENGTH = 5


@dataclass
class GameState:
    board: np.ndarray  # 0=empty, 1=black, 2=white
    current_player: int  # 1 or 2
    winner: Optional[int] = None  # None=ongoing, 0=draw, 1=black, 2=white
    last_move: Optional[Tuple[int, int]] = None

    @classmethod
    def new(cls) -> "GameState":
        return cls(board=np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8), current_player=1)

    def valid_moves(self) -> list[Tuple[int, int]]:
        """Return list of (row, col) for empty cells."""
        rows, cols = np.where(self.board == 0)
        return list(zip(rows.tolist(), cols.tolist()))

    def make_move(self, row: int, col: int) -> bool:
        """Place stone. Returns True if valid, False otherwise."""
        if self.winner is not None:
            return False
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False
        if self.board[row, col] != 0:
            return False
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        if self._check_win(row, col):
            self.winner = self.current_player
        elif len(self.valid_moves()) == 0:
            self.winner = 0  # draw
        else:
            self.current_player = 3 - self.current_player
        return True

    def _check_win(self, row: int, col: int) -> bool:
        """Check if the last move at (row, col) wins."""
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r, c] == player:
                    count += 1
                    r += dr * sign
                    c += dc * sign
            if count >= WIN_LENGTH:
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "board": self.board.tolist(),
            "current_player": self.current_player,
            "winner": self.winner,
            "last_move": self.last_move,
        }
