import math

import pytest

from request_vqa.uncertainty import (
    normalized_entropy,
    routing_threshold,
    spacing_scale,
)

PAPER = {"s_min": 0.2, "rho": 0.07, "u_min": 0.45, "u_max": 1.0}


def test_normalized_entropy_bounds() -> None:
    assert normalized_entropy([0.25] * 4) == pytest.approx(1.0)
    assert normalized_entropy([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


def test_normalized_entropy_divides_by_log_num_options() -> None:
    probs = [0.8, 0.1, 0.05, 0.05]
    raw = -sum(p * math.log(p) for p in probs)
    assert normalized_entropy(probs) == pytest.approx(raw / math.log(4))
    assert normalized_entropy(probs) == pytest.approx(0.511, abs=1e-3)


def test_normalized_entropy_renormalizes_over_options() -> None:
    probs = [0.8, 0.1, 0.05, 0.05]
    leaked = [p * 0.6 for p in probs]  # mass on non-option tokens
    assert normalized_entropy(leaked) == pytest.approx(normalized_entropy(probs))


def test_normalized_entropy_without_mass_is_maximal() -> None:
    assert normalized_entropy([0.0, 0.0, 0.0]) == 1.0


def test_normalized_entropy_rejects_single_option() -> None:
    with pytest.raises(ValueError):
        normalized_entropy([1.0])


def test_routing_threshold_shrinks_with_length_and_saturates() -> None:
    assert routing_threshold(0, 0.55, 0.3, 3600) == pytest.approx(0.55)
    assert routing_threshold(1800, 0.55, 0.3, 3600) == pytest.approx(0.40)
    assert routing_threshold(3600, 0.55, 0.3, 3600) == pytest.approx(0.25)
    assert routing_threshold(7200, 0.55, 0.3, 3600) == pytest.approx(0.25)


def test_spacing_scale_is_uniform_below_u_min() -> None:
    assert spacing_scale(0.44, **PAPER) == 1.0


def test_spacing_scale_reaches_s_min_at_u_max() -> None:
    assert spacing_scale(1.0, **PAPER) == pytest.approx(0.2)
    assert spacing_scale(1.5, **PAPER) == pytest.approx(0.2)


def test_spacing_scale_decreases_with_entropy() -> None:
    values = [spacing_scale(u, **PAPER) for u in (0.45, 0.5, 0.7, 0.9, 1.0)]
    assert values == sorted(values, reverse=True)
