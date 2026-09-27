import random

from briscola import STANDARD, GameState, RandomAgent, evaluate, make_agent
from briscola.encoding import action_mask, encode
from briscola.env import BriscolaEnv
from briscola.learners import LinearQLearner, _sparse
from briscola.train import epsilon_at, train


def test_update_moves_q_towards_target():
    learner = LinearQLearner(memory=True, lr=0.5)
    state = GameState.new_game(seed=0)
    obs = state.observation(0)
    x, mask = encode(obs, True), action_mask(obs)
    a = mask.index(True)
    before = learner.q_value(_sparse(x), a)
    learner.update(x, a, 1.0, x, mask, done=True)
    after = learner.q_value(_sparse(x), a)
    assert before == 0.0 and 0.0 < after < 1.0
    others = [b for b, ok in enumerate(mask) if ok and b != a]
    assert all(learner.q_value(_sparse(x), b) == 0.0 for b in others)


def test_greedy_actions_are_legal_and_follow_q():
    learner = LinearQLearner(memory=False)
    obs = GameState.new_game(seed=1).observation(0)
    x, mask = encode(obs, False), action_mask(obs)
    rng = random.Random(0)
    legal = [a for a, ok in enumerate(mask) if ok]
    learner.b[legal[1]] = 1.0
    assert learner.select_action(x, mask, rng) == legal[1]
    assert learner.act(obs, rng) == STANDARD.deck[legal[1]]
    for _ in range(50):
        assert mask[learner.select_action(x, mask, rng, epsilon=1.0)]


def test_save_and_load(tmp_path):
    learner = LinearQLearner(memory=True)
    learner.w[3][7] = 0.25
    learner.b[5] = -1.0
    learner.save(tmp_path / "w.json")
    loaded = LinearQLearner.load(tmp_path / "w.json")
    assert loaded.w == learner.w and loaded.b == learner.b
    assert loaded.memory and loaded.config == STANDARD


def test_epsilon_schedule():
    assert epsilon_at(0, 100, 0.3, 0.02, 0.8) == 0.3
    assert abs(epsilon_at(80, 100, 0.3, 0.02, 0.8) - 0.02) < 1e-12
    assert abs(epsilon_at(99, 100, 0.3, 0.02, 0.8) - 0.02) < 1e-12


def test_short_training_learns_to_beat_random():
    learner = LinearQLearner(memory=True)
    env = BriscolaEnv([make_agent("random"), make_agent("greedy")], reward="trick_points")
    history = train(learner, env, 3000, eval_every=3000,
                    eval_opponents=[RandomAgent()], eval_deals=100, seed=0)
    assert [row["episode"] for row in history] == [0, 3000]
    assert history[0]["mean_reward"] < history[-1]["mean_reward"]
    stats = evaluate(learner, RandomAgent(), 300, seed=123)
    assert stats.mean_reward - stats.reward_ci95 > 0
