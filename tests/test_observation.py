import random

from briscola import STANDARD, GameConfig, GameState


def test_observation_hides_opponent_hand_and_stock():
    # Swap one of player 1's cards with a card deep in the stock: player 0
    # cannot tell the two deals apart, so its observation must be identical.
    deck = list(STANDARD.deck)
    random.Random(0).shuffle(deck)
    swapped = list(deck)
    swapped[1], swapped[20] = swapped[20], swapped[1]
    a, b = GameState(STANDARD, deck), GameState(STANDARD, swapped)
    assert a.hands[1] != b.hands[1]
    assert a.observation(0) == b.observation(0)
    assert a.observation(0).info_state_key() == b.observation(0).info_state_key()
    assert a.observation(1).info_state_key() != b.observation(1).info_state_key()


def test_observation_contents():
    state = GameState.new_game(seed=1)
    obs = state.observation(0)
    assert obs.hand == tuple(state.hands[0])
    assert obs.is_leading and obs.legal_actions == obs.hand
    assert obs.stock_size == 34 and obs.known_opponent_cards == ()
    state.play(obs.hand[0])
    obs0, obs1 = state.observation(0), state.observation(1)
    assert obs0.legal_actions == () and not obs0.is_my_turn
    assert obs1.table == (state.plays[0][1],) and not obs1.is_leading
    # 40 minus 2 cards in hand, the card on the table and the trump card.
    assert len(obs0.unseen_cards()) == 40 - 2 - 1 - 1


def test_trump_card_holder_is_public_and_endgame_is_inferable():
    rng = random.Random(4)
    state = GameState.new_game(STANDARD, rng=rng)
    trump = state.trump_card
    while not state.is_terminal:
        for p in (0, 1):
            obs = state.observation(p)
            opp_hand = set(state.hands[1 - p])
            assert set(obs.known_opponent_cards) == ({trump} & opp_hand)
            if obs.stock_size == 0:
                # Card counting reveals the opponent's whole hand.
                assert obs.unseen_cards() | set(obs.known_opponent_cards) == opp_hand
        state.play(rng.choice(state.legal_actions()))


def test_my_draws_are_private():
    rng = random.Random(2)
    state = GameState.new_game(GameConfig.reduced(suits=("Coppe", "Denari"), ranks=(1, 3, 10, 2)), rng=rng)
    while not state.is_terminal:
        state.play(rng.choice(state.legal_actions()))
    for p in (0, 1):
        assert state.observation(p).my_draws == tuple(c for q, c in state.draws if q == p)
