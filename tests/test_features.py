import random

import pytest

from briscola import STANDARD, Card, GameState, RandomAgent, evaluate, make_agent
from briscola.env import BriscolaEnv
from briscola.features import FEATURE_NAMES, NUM_FEATURES, action_features
from briscola.learners import FeatureQLearner
from briscola.train import train

F = {name: i for i, name in enumerate(FEATURE_NAMES)}


def deal(p0_hand, trump_card, seed=0):
    """A game where player 0 holds ``p0_hand`` and ``trump_card`` is face up."""
    rest = [c for c in STANDARD.deck if c not in p0_hand and c != trump_card]
    random.Random(seed).shuffle(rest)
    deck = [p0_hand[0], rest[0], p0_hand[1], rest[1], p0_hand[2], rest[2], trump_card] + rest[3:]
    return GameState(STANDARD, deck, first_player=0)


def test_hypergeometric_safety_on_first_lead():
    ace, king, four = Card("Coppe", 1), Card("Spade", 10), Card("Bastoni", 4)
    state = deal([ace, king, four], Card("Spade", 7))
    phi = action_features(state.observation(0))
    idx = STANDARD.card_index
    assert len(phi) == 3 and all(len(v) == NUM_FEATURES for v in phi.values())
    # 36 unseen cards, opponent holds 3 of them.
    assert phi[idx[ace]][F["p_not_beaten"]] == pytest.approx(3276 / 7140)   # 8 unseen trumps
    assert phi[idx[king]][F["p_not_beaten"]] == pytest.approx(5984 / 7140)  # Asso, Tre di Spade
    assert phi[idx[four]][F["p_not_beaten"]] == pytest.approx(1140 / 7140)  # 8 Bastoni + 8 trumps
    assert phi[idx[ace]][F["top_of_suit"]] == 1.0
    assert phi[idx[king]][F["top_of_suit"]] == 0.0
    assert phi[idx[king]][F["trump_x_stock"]] == 1.0
    assert phi[idx[ace]][F["points_at_risk"]] == pytest.approx(1 - 3276 / 7140)
    assert all(v[F["bias"]] == 1.0 and v[F["wins_trick"]] == 0.0 for v in phi.values())


def test_following_features_are_certain():
    rng = random.Random(3)
    for _ in range(30):
        state = GameState.new_game(STANDARD, rng=rng)
        state.play(state.legal_actions()[0])            # player 0 leads
        obs = state.observation(1)
        lead = obs.table[0]
        for a, phi in action_features(obs).items():
            card = STANDARD.deck[a]
            wins = phi[F["wins_trick"]]
            assert phi[F["p_not_beaten"]] == wins
            assert phi[F["points_at_risk"]] == pytest.approx(card.points / 11 * (1 - wins))
            assert phi[F["trick_points_if_win"]] == pytest.approx(wins * (lead.points + card.points) / 22)
            assert phi[F["points_x_leading"]] == 0.0


def test_endgame_safety_is_zero_or_one():
    rng = random.Random(8)
    state = GameState.new_game(STANDARD, rng=rng)
    checked = 0
    while not state.is_terminal:
        p = state.current_player
        obs = state.observation(p)
        if obs.stock_size == 0 and not obs.table:
            for phi in action_features(obs).values():
                assert phi[F["p_not_beaten"]] in (0.0, 1.0)
            checked += 1
        state.play(rng.choice(state.legal_actions()))
    assert checked > 0


def test_no_features_when_not_my_turn():
    state = GameState.new_game(seed=1)
    assert action_features(state.observation(1)) == {}


def test_feature_learner_update_and_io(tmp_path):
    learner = FeatureQLearner(lr=0.5)
    state = GameState.new_game(seed=2)
    x = learner.featurize(state.observation(0))
    a = next(iter(x))
    learner.update(x, a, 1.0, {}, [], done=True)
    assert 0.0 < learner.q_value(x[a]) < 1.0
    assert set(learner.weights()) == set(FEATURE_NAMES)
    assert learner.act(state.observation(0), random.Random(0)) in state.hands[0]
    learner.save(tmp_path / "f.json")
    assert FeatureQLearner.load(tmp_path / "f.json").w == learner.w


def test_short_feature_training_beats_random():
    learner = FeatureQLearner()
    env = BriscolaEnv([make_agent("random"), make_agent("greedy")], reward="trick_points")
    train(learner, env, 2000, seed=0)
    stats = evaluate(learner, RandomAgent(), 300, seed=123)
    assert stats.mean_reward - stats.reward_ci95 > 0.3
