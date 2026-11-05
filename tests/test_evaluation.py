import numpy as np
import pytest

from latedeliveries.evaluation import (
    choose_threshold,
    cost_by_threshold,
    order_cost,
    policy_costs,
)


def test_cost_of_a_small_example():
    y_true = [1, 0, 0, 1, 0]
    flagged = [True, True, False, False, False]
    # two vouchers (2 x 4) and one missed late order (1 x 15)
    assert order_cost(y_true, flagged) == 2 * 4 + 1 * 15


@pytest.mark.parametrize(
    ("y_true", "flagged", "expected"),
    [
        ([1, 1, 0], [False, False, False], 30),  # two missed late orders
        ([1, 1, 0], [True, True, True], 12),  # three vouchers
        ([0, 0], [True, False], 4),  # one wasted voucher
        ([1, 0], [True, False], 4),  # late order caught, nothing else
    ],
)
def test_order_cost_cases(y_true, flagged, expected):
    assert order_cost(y_true, flagged) == expected


def test_threshold_on_a_hand_worked_example():
    y_true = [1, 1, 0, 0]
    proba = [0.8, 0.3, 0.2, 0.1]
    # Flag nobody: 2 x 15 = 30. Flag 0.8 only: 4 + 15 = 19.
    # Flag 0.8 and 0.3: 2 x 4 = 8. Flag three: 12. Flag four: 16.
    # The cheapest is 8, first reached at threshold 0.25.
    assert choose_threshold(y_true, proba) == (0.25, 8)


def test_flagging_nobody_can_be_cheapest():
    y_true = [0, 0, 0]
    proba = [0.1, 0.2, 0.3]
    threshold, cost = choose_threshold(y_true, proba)
    assert cost == 0
    assert threshold == 0.35  # first threshold above every probability


def test_flagging_everybody_can_be_cheapest():
    y_true = [1, 1, 1]
    proba = [0.5, 0.6, 0.7]
    threshold, cost = choose_threshold(y_true, proba)
    assert cost == 3 * 4
    assert threshold == 0.05  # lowest threshold wins the tie


def test_cost_table_covers_all_thresholds():
    table = cost_by_threshold([1, 0], [0.9, 0.1])
    assert len(table) == 19
    assert table["threshold"].iloc[0] == 0.05
    assert table["threshold"].iloc[-1] == 0.95


def test_policy_costs_on_a_small_example():
    y_true = np.array([1, 0, 0, 1])
    proba = np.array([0.9, 0.5, 0.1, 0.2])
    costs = policy_costs(y_true, proba, 0.4)
    assert costs == {"flag_nobody": 30, "flag_everybody": 16, "model": 8 + 15}
