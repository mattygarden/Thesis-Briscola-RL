import itertools
import random
from collections import Counter

import pytest

from briscola import STANDARD, GameConfig, GameState, IllegalMoveError


def play_random(state: GameState, rng: random.Random) -> GameState:
    while not state.is_terminal:
        state.play(rng.choice(state.legal_actions()))
    return state


def check_finished_game(state: GameState) -> None:
    cfg = state.config
    assert state.is_terminal
    assert len(state.tricks) == cfg.num_tricks
    played = [c for _, c in state.plays]
    assert Counter(played) == Counter(cfg.deck)                   # every card exactly once
    assert sum(state.scores) == cfg.total_points
    for p in (0, 1):
        assert state.scores[p] == sum(c.points for c in state.captured[p])
    assert state.hands == [[], []] and state.stock == [] and state.table == []
    assert len(state.draws) == cfg.initial_stock_size
    for prev, trick in zip(state.tricks, state.tricks[1:]):
        assert trick.leader == prev.winner                        # winner leads next


def test_deal():
    deck = list(STANDARD.deck)
    state = GameState(STANDARD, deck, first_player=1)
    assert state.hands[1] == [deck[0], deck[2], deck[4]]
    assert state.hands[0] == [deck[1], deck[3], deck[5]]
    assert state.trump_card == deck[6]
    assert state.stock[-1] == deck[6] and len(state.stock) == 34
    assert state.current_player == 1


def test_random_games_invariants():
    for seed in range(300):
        rng = random.Random(seed)
        state = GameState.new_game(STANDARD, rng=rng, first_player=seed % 2)
        hand_sizes = []
        while not state.is_terminal:
            if not state.table:
                hand_sizes.append((len(state.hands[0]), len(state.hands[1])))
            state.play(rng.choice(state.legal_actions()))
        assert hand_sizes == [(3, 3)] * 18 + [(2, 2), (1, 1)]
        check_finished_game(state)


def test_winner_draws_first_and_loser_gets_trump_card():
    rng = random.Random(1)
    state = GameState.new_game(STANDARD, rng=rng)
    trump = state.trump_card
    last_draw_trick = STANDARD.initial_stock_size // 2
    while not state.is_terminal:
        n_draws = len(state.draws)
        state.play(rng.choice(state.legal_actions()))
        if len(state.draws) > n_draws:
            winner = state.tricks[-1].winner
            assert [q for q, _ in state.draws[-2:]] == [winner, 1 - winner]
            if len(state.tricks) == last_draw_trick:
                assert state.draws[-1] == (1 - winner, trump)
    check_finished_game(state)


def test_reduced_game_every_deal():
    cfg = GameConfig.reduced()
    rng = random.Random(0)
    for deck in itertools.permutations(cfg.deck):
        state = play_random(GameState(cfg, deck), rng)
        check_finished_game(state)


def test_same_seed_same_game():
    a = play_random(GameState.new_game(seed=42), random.Random(7))
    b = play_random(GameState.new_game(seed=42), random.Random(7))
    assert a.plays == b.plays and a.scores == b.scores


def test_illegal_moves():
    state = GameState.new_game(seed=3)
    not_in_hand = next(c for c in STANDARD.deck if c not in state.hands[0])
    with pytest.raises(IllegalMoveError):
        state.play(not_in_hand)
    play_random(state, random.Random(0))
    with pytest.raises(IllegalMoveError):
        state.play(STANDARD.deck[0])
    assert state.legal_actions() == []


def test_bad_deck_rejected():
    with pytest.raises(ValueError):
        GameState(STANDARD, list(STANDARD.deck)[:-1])
    with pytest.raises(ValueError):
        GameState(STANDARD, list(STANDARD.deck[:-1]) + [STANDARD.deck[0]])


def test_clone_is_independent():
    state = GameState.new_game(seed=5)
    copy = state.clone()
    play_random(copy, random.Random(0))
    assert len(state.plays) == 0 and len(state.hands[0]) == 3
    assert copy.is_terminal


def test_returns():
    state = play_random(GameState.new_game(seed=11), random.Random(0))
    w = state.winner()
    expected = {None: (0, 0), 0: (1, -1), 1: (-1, 1)}[w]
    assert state.returns() == expected
    state.scores = [60, 60]
    assert state.winner() is None and state.returns() == (0, 0)
    with pytest.raises(ValueError):
        GameState.new_game(seed=0).winner()
