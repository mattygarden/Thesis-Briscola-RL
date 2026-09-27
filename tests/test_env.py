import random

import pytest

from briscola import STANDARD, GameConfig, GreedyAgent, IllegalMoveError, RandomAgent
from briscola.encoding import feature_size
from briscola.env import BriscolaEnv


def run_episode(env: BriscolaEnv, rng: random.Random, seed=None):
    x, info = env.reset(seed=seed)
    rewards, steps, done = [], 0, False
    while not done:
        assert len(x) == env.observation_size
        mask = info["action_mask"]
        legal = [a for a, ok in enumerate(mask) if ok]
        assert {STANDARD.deck[a] for a in legal} == set(env.observation().hand)
        x, r, done, truncated, info = env.step(rng.choice(legal))
        assert truncated is False
        rewards.append(r)
        steps += 1
    return rewards, steps, info


@pytest.mark.parametrize("memory", [False, True])
def test_episode_runs_to_the_end(memory):
    env = BriscolaEnv(RandomAgent(), memory=memory, seed=0)
    assert env.observation_size == feature_size(STANDARD, memory)
    assert env.action_size == 40
    rewards, steps, info = run_episode(env, random.Random(0))
    assert steps == 20                     # the agent plays one card per trick
    assert env.state.is_terminal
    assert sum(info["scores"]) == 120
    assert rewards[:-1] == [0.0] * 19 and rewards[-1] == info["outcome"]
    assert not any(info["action_mask"])


def test_reward_modes_sum_to_final_values():
    for seed in range(20):
        env = BriscolaEnv(GreedyAgent(), reward="trick_points", seed=seed)
        rewards, _, info = run_episode(env, random.Random(seed))
        mine, theirs = info["scores"]
        assert sum(rewards) == pytest.approx((mine - theirs) / 120)
        env = BriscolaEnv(GreedyAgent(), reward="points", seed=seed)
        rewards, _, info = run_episode(env, random.Random(seed))
        mine, theirs = info["scores"]
        assert rewards[-1] == pytest.approx((mine - theirs) / 120)
        assert all(r == 0 for r in rewards[:-1])


def test_seats_and_opponent_mixture():
    env = BriscolaEnv([RandomAgent(), GreedyAgent()], seed=1)
    seats, opponents = set(), set()
    for _ in range(50):
        _, info = env.reset()
        seats.add(info["seat"])
        opponents.add(info["opponent"])
        if info["seat"] == 1:
            # The opponent leads the first trick, so one card is already on the table.
            assert len(env.observation().table) == 1
        else:
            assert env.observation().is_leading
    assert seats == {0, 1} and opponents == {"random", "greedy"}
    fixed = BriscolaEnv(RandomAgent(), agent_seat=1, seed=1)
    assert all(fixed.reset()[1]["seat"] == 1 for _ in range(10))


def test_reset_with_seed_is_reproducible():
    env = BriscolaEnv([RandomAgent(), GreedyAgent()])
    a = run_episode(env, random.Random(5), seed=42)
    b = run_episode(env, random.Random(5), seed=42)
    assert a == b


def test_illegal_actions_and_misuse():
    env = BriscolaEnv(RandomAgent(), seed=0)
    with pytest.raises(RuntimeError):
        env.step(0)
    _, info = env.reset()
    illegal = info["action_mask"].index(False)
    with pytest.raises(IllegalMoveError):
        env.step(illegal)
    run_episode(env, random.Random(0))
    with pytest.raises(RuntimeError):
        env.step(0)
    with pytest.raises(ValueError):
        BriscolaEnv([], seed=0)
    with pytest.raises(ValueError):
        BriscolaEnv(RandomAgent(), reward="money")
    with pytest.raises(ValueError):
        BriscolaEnv(RandomAgent(), agent_seat=2)


def test_reduced_game_env():
    env = BriscolaEnv(RandomAgent(), GameConfig.reduced(), seed=0)
    x, info = env.reset()
    assert env.action_size == 6 and len(x) == env.observation_size
    done, steps = False, 0
    while not done:
        a = info["action_mask"].index(True)
        x, r, done, _, info = env.step(a)
        steps += 1
    assert steps == 3
