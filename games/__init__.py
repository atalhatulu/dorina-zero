"""games paketi."""
from games.base_game import BaseGame
from games.connect4 import Connect4Game
from games.othello import OthelloGame

__all__ = ["BaseGame", "Connect4Game", "OthelloGame"]
