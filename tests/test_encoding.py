import random

from briscola import STANDARD, GameState
from briscola.encoding import (
    action_mask, action_to_card, card_to_action, encode, feature_size,
)


def test_feature_sizes_and_memory_block():
    rng = random.Random(0)
    state = GameState.new_game(STANDARD, rng=rng)
    for _ in range(10):
        state.play(rng.choice(state.legal_actions()))
    obs = state.observation(state.current_player)
    basic, memory = encode(obs, memory=False), encode(obs, memory=True)
    assert len(basic) == feature_size(STANDARD, memory=False) == 4 * 40 + 4 + 4
    assert len(memory) == feature_size(STANDARD, memory=True) == len(basic) + 40
    assert memory[: len(basic)] == basic
    completed = {c for t in obs.tricks for c in (t.lead, t.follow)}
    assert sum(memory[len(basic):]) == len(completed) == 2 * len(state.tricks)
    assert all(0.0 <= x <= 1.0 for x in memory)


def test_action_mask_and_ids():
    state = GameState.new_game(seed=3)
    obs = state.observation(0)
    mask = action_mask(obs)
    legal = {action_to_card(i, STANDARD) for i, ok in enumerate(mask) if ok}
    assert legal == set(obs.hand)
    assert all(action_to_card(card_to_action(c, STANDARD), STANDARD) == c for c in STANDARD.deck)
    assert not any(action_mask(state.observation(1)))
