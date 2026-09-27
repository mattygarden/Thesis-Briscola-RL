import pytest

from briscola.stats import holm_adjust, t_two_sided_p, welch_t_test


@pytest.mark.parametrize(
    "t, df, p",
    [   # critical values from standard t tables
        (2.776, 4, 0.05),
        (2.228, 10, 0.05),
        (1.960, 1e6, 0.05),
        (4.604, 4, 0.01),
        (0.0, 7, 1.0),
    ],
)
def test_t_distribution_p_values(t, df, p):
    assert t_two_sided_p(t, df) == pytest.approx(p, abs=5e-4)
    assert t_two_sided_p(-t, df) == pytest.approx(p, abs=5e-4)


def test_welch_t_test():
    a, b = [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0, 10.0]
    r = welch_t_test(a, b)
    assert r.diff == pytest.approx(2.5 - 6.0)
    # Hand computation: var a = 5/3, var b = 10; se^2 = 5/12 + 2.
    assert r.t == pytest.approx(-3.5 / (5 / 12 + 2) ** 0.5)
    assert r.df == pytest.approx((5 / 12 + 2) ** 2 / ((5 / 12) ** 2 / 3 + 2 ** 2 / 4))
    assert 0.06 < r.p_value < 0.08            # t = -2.25 with about 5.5 df
    assert welch_t_test([1, 2, 3], [1, 2, 3]).p_value == pytest.approx(1.0)
    with pytest.raises(ValueError):
        welch_t_test([1.0], [1.0, 2.0])


def test_holm_adjust():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5]) == [0.5]


def test_compare_conditions():
    from briscola.experiment import compare_conditions

    final = []
    for seed, (b, m) in enumerate([(0.10, 0.20), (0.12, 0.21), (0.11, 0.19)]):
        final += [
            {"condition": "basic", "seed": seed, "opponent": "greedy", "mean_reward": b},
            {"condition": "memory", "seed": seed, "opponent": "greedy", "mean_reward": m},
            {"condition": "basic", "seed": seed, "opponent": "random", "mean_reward": 0.5},
            {"condition": "memory", "seed": seed, "opponent": "random", "mean_reward": 0.5 + seed / 100},
        ]
    rows = {r["opponent"]: r for r in compare_conditions(final)}
    assert set(rows) == {"greedy", "random", "media"}
    assert rows["greedy"]["diff"] == pytest.approx(0.09)
    assert rows["greedy"]["p_value"] < 0.01
    assert rows["greedy"]["p_holm"] == pytest.approx(2 * rows["greedy"]["p_value"])
    assert rows["media"]["memory"] == pytest.approx((0.20 + 0.21 + 0.19 + 1.5 + 0.03) / 6)
