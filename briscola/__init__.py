"""Two-player Briscola simulator for reinforcement learning and equilibrium computation."""

from .agents import GreedyAgent, LowestCardAgent, RandomAgent, make_agent
from .arena import MatchStats, evaluate, play_game
from .cards import Card
from .game import GameState, IllegalMoveError
from .observation import Observation, Trick
from .rules import STANDARD, GameConfig, trick_winner

__all__ = [
    "Card", "GameConfig", "STANDARD", "trick_winner",
    "GameState", "IllegalMoveError", "Observation", "Trick",
    "RandomAgent", "LowestCardAgent", "GreedyAgent", "make_agent",
    "play_game", "evaluate", "MatchStats",
]
