import pytest
from inspect_ai.scorer import Score

from openbench.metrics.pass_hat import pass_hat


def _score_series(values, **kwargs):
    return [Score(value=v, **kwargs) for v in values]


def test_pass_hat_combination_per_sample():
    run_a = _score_series([0.0, 1.0, 1.0], metadata={"id": "a"})
    run_b = _score_series([0.0, 0.0, 1.0], metadata={"id": "b"})

    pass1 = pass_hat(k=1)
    pass2 = pass_hat(k=2)

    reduced_a_k1 = pass1(run_a)
    reduced_a_k2 = pass2(run_a)
    reduced_b_k1 = pass1(run_b)
    reduced_b_k2 = pass2(run_b)

    assert reduced_a_k1.value == pytest.approx(2 / 3)
    assert reduced_a_k2.value == pytest.approx(1 / 3)
    assert reduced_b_k1.value == pytest.approx(1 / 3)
    assert reduced_b_k2.value == pytest.approx(0.0)

    dataset_pass1 = (reduced_a_k1.value + reduced_b_k1.value) / 2
    dataset_pass2 = (reduced_a_k2.value + reduced_b_k2.value) / 2

    assert dataset_pass1 == pytest.approx(0.5)
    assert dataset_pass2 == pytest.approx(1 / 6)


def test_pass_hat_missing_epochs_returns_zero():
    reducer = pass_hat(k=3)
    run = _score_series([1.0, 0.0], metadata={})
    reduced = reducer(run)
    assert reduced.value == pytest.approx(0.0)


def test_pass_hat_counts_only_exact_success():
    reducer = pass_hat(k=1)
    run = _score_series([1.0, 0.9], metadata={})
    reduced = reducer(run)
    assert reduced.value == pytest.approx(0.5)


def test_pass_hat_supports_dict_scores():
    reducer = pass_hat(k=1)
    scores = [
        Score(value={"reward": 1.0}, metadata={}),
        Score(value={"reward": 0.0}, metadata={}),
    ]
    reduced = reducer(scores)
    assert reduced.value == {"reward": pytest.approx(0.5)}
