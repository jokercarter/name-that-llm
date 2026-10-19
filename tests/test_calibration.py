import numpy as np

from ntl_pact18.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    sinkhorn_balance,
    sld_em,
)
from ntl_pact18.labels import N_CLASSES


def _overconfident_probs(y: np.ndarray, strength: float = 6.0) -> np.ndarray:
    logits = np.full((len(y), N_CLASSES), 0.0)
    logits[np.arange(len(y)), y] = strength
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def test_temperature_returns_finite_and_reduces_nll():
    rng = np.random.default_rng(0)
    y = rng.integers(0, N_CLASSES, size=400)
    # deliberately under-confident: temperature < 1 should sharpen
    probs = 0.6 * _overconfident_probs(y, strength=1.0) + 0.4 / N_CLASSES
    t = fit_temperature(probs, y)
    assert 0.05 <= t <= 20.0
    nll_before = -np.log(probs[np.arange(len(y)), y]).mean()
    nll_after = -np.log(apply_temperature(probs, t)[np.arange(len(y)), y]).mean()
    assert nll_after <= nll_before + 1e-6


def test_sinkhorn_makes_marginals_more_uniform():
    probs = np.full((200, N_CLASSES), 0.01)
    probs[:, 0] = 0.9
    probs /= probs.sum(axis=1, keepdims=True)
    before = probs.sum(axis=0)
    after = sinkhorn_balance(probs, alpha=1.0, iters=100).sum(axis=0)
    assert after.std() < before.std()
    assert after.min() > before.min()


def test_sld_em_recovers_a_shifted_prior():
    rng = np.random.default_rng(1)
    n = 3000
    true_prior = np.full(N_CLASSES, 0.2 / (N_CLASSES - 1))
    true_prior[0] = 0.8
    y = rng.choice(N_CLASSES, size=n, p=true_prior)
    probs = _overconfident_probs(y, strength=2.5)
    probs = 0.7 * probs + 0.3 / N_CLASSES

    pi_train = np.full(N_CLASSES, 1.0 / N_CLASSES)
    adjusted, pi_hat = sld_em(probs, pi_train, iters=200)
    assert pi_hat.shape == (N_CLASSES,)
    assert abs(pi_hat[0] - true_prior[0]) < abs(pi_train[0] - true_prior[0])
    assert np.allclose(adjusted.sum(axis=1), 1.0)


def test_ece_is_bounded_and_zero_when_perfect():
    y = np.arange(10)
    probs = np.zeros((10, N_CLASSES))
    probs[np.arange(10), y] = 1.0
    assert expected_calibration_error(probs, y) < 1e-6
    uniform = np.full((10, N_CLASSES), 1.0 / N_CLASSES)
    ece = expected_calibration_error(uniform, y)
    assert 0.0 <= ece <= 1.0
