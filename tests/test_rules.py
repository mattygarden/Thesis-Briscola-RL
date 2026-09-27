import pytest

from briscola import STANDARD, Card, GameConfig, trick_winner


def test_standard_deck():
    assert STANDARD.deck_size == 40
    assert len(set(STANDARD.deck)) == 40
    assert STANDARD.total_points == 120
    assert STANDARD.num_tricks == 20
    assert STANDARD.initial_stock_size == 34


def test_card_points_and_strength_order():
    points = {r: Card("Coppe", r).points for r in range(1, 11)}
    assert points == {1: 11, 2: 0, 3: 10, 4: 0, 5: 0, 6: 0, 7: 0, 8: 2, 9: 3, 10: 4}
    weakest_to_strongest = [2, 4, 5, 6, 7, 8, 9, 10, 3, 1]
    strengths = [Card("Coppe", r).strength for r in weakest_to_strongest]
    assert strengths == sorted(strengths)
    assert len(set(strengths)) == 10


def test_card_labels():
    assert str(Card("Denari", 1)) == "Asso di Denari"
    assert Card("Bastoni", 9).short == "Cb"


@pytest.mark.parametrize(
    "lead, follow, trump, expected",
    [
        (Card("Coppe", 7), Card("Coppe", 1), "Spade", 1),     # higher card of lead suit
        (Card("Coppe", 3), Card("Coppe", 10), "Spade", 0),    # Tre beats Re
        (Card("Coppe", 2), Card("Denari", 1), "Spade", 0),    # off-suit, not trump
        (Card("Coppe", 1), Card("Spade", 2), "Spade", 1),     # any trump beats a non-trump
        (Card("Spade", 2), Card("Coppe", 1), "Spade", 0),     # trump lead, off-suit reply
        (Card("Spade", 4), Card("Spade", 2), "Spade", 0),     # lower trump
        (Card("Spade", 10), Card("Spade", 3), "Spade", 1),    # higher trump
    ],
)
def test_trick_winner(lead, follow, trump, expected):
    assert trick_winner(lead, follow, trump) == expected


def test_reduced_config():
    cfg = GameConfig.reduced()
    assert cfg.deck_size == 6
    assert cfg.num_tricks == 3
    assert cfg.initial_stock_size == 2
    assert cfg.total_points == 50


@pytest.mark.parametrize(
    "kwargs",
    [
        {"suits": ()},
        {"suits": ("Coppe", "Coppe")},
        {"suits": ("Cuori",)},
        {"ranks": (0, 1)},
        {"hand_size": 0},
        {"suits": ("Coppe",), "ranks": (1, 3, 10)},                  # odd deck size
        {"suits": ("Coppe", "Denari"), "ranks": (1, 3), "hand_size": 2},  # no card left for trump
    ],
)
def test_invalid_configs(kwargs):
    with pytest.raises(ValueError):
        GameConfig(**kwargs)
