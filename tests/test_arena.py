from briscola import GameConfig, GreedyAgent, LowestCardAgent, RandomAgent, evaluate, play_game


def test_play_game_is_reproducible():
    agents = (RandomAgent(), RandomAgent())
    assert play_game(agents, seed=9) == play_game(agents, seed=9)
    assert sum(play_game(agents, seed=9).scores) == 120


def test_duplicate_mirror_match_is_exactly_even():
    # Deterministic agents with identical policies: each deal is played twice
    # with seats swapped, so the two games mirror each other exactly.
    stats = evaluate(GreedyAgent(), GreedyAgent(), 200, seed=1)
    assert stats.games == 400
    assert stats.wins == stats.losses
    assert stats.mean_reward == 0 and stats.mean_point_diff == 0


def test_greedy_beats_random():
    stats = evaluate(GreedyAgent(), RandomAgent(), 500, seed=0)
    assert stats.mean_reward - stats.reward_ci95 > 0
    assert stats.wins + stats.draws + stats.losses == 1000


def test_evaluate_on_reduced_game_without_duplicate():
    stats = evaluate(LowestCardAgent(), RandomAgent(), 100, GameConfig.reduced(), duplicate=False)
    assert stats.games == 100
