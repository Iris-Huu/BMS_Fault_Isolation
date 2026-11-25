"""Independent closed-form and nonlinear checks for the Week 3 calculation."""
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from src.distinguishability import colored_covariance, profile, short_boundary, window_signatures
from src.model import simulate


def check():
    t = np.arange(301.)
    zero = np.zeros_like(t)
    ambient = zero + 25.
    ocv = pd.DataFrame({'soc': [0., 1.], 'ocv_v': [3.6, 3.6]})
    p = dict(Q_Ah=3., R0_ohm=.04, R1_ohm=.01, C1_F=500.,
             R2_ohm=.02, C2_F=1000., Cth_J_K=60., hA_W_K=.12)
    signatures, healthy = window_signatures(t, zero, ambient, p, ocv, [.5, 0., 0., 25.])
    bias, short, chamber = signatures.transpose(2, 0, 1)
    rise = -np.expm1(-t * p['hA_W_K'] / p['Cth_J_K'])
    assert_allclose(short[:, 0], -3.6 * bias[:, 0], atol=2e-8)
    assert_allclose(short[:, 1], 3.6**2 / p['hA_W_K'] * rise, atol=2e-7)
    assert_allclose(chamber[:, 1], rise, atol=2e-8)
    d_electric, *_ = profile(short[:, 0] / 100, bias[:, 0], zero, 0.)
    d_thermal, *_ = profile(short.ravel() / 100, bias.ravel(), chamber.ravel(), 0.)
    d_profiled, *_ = profile(short.ravel() / 100, bias.ravel(), chamber.ravel(), .7)
    # Independent closed-form KL for known ambient and unit output covariance.
    assert_allclose(d_thermal, .5 * np.sum((short[:, 1] / 100) ** 2), rtol=1e-9)
    for allowed_ambient in [0., .7]:
        distances = [profile(short.ravel() * conductance, bias.ravel(), chamber.ravel(),
                             allowed_ambient)[0] for conductance in [0., .001, .003, .01, .03, .1]]
        assert distances[0] == 0.
        assert np.all(np.diff(distances) >= -1e-12)
    assert d_electric < 1e-14 and d_thermal > 1. and d_profiled < d_thermal
    for resistance in [10, 100, 300, 1000]:
        actual = simulate(t, zero, ambient, p, ocv, soc0=.5, faults={'fisc': 1 / resistance})
        delta = actual[['voltage_v', 'temperature_c']].to_numpy() - healthy
        assert np.linalg.norm(delta - short / resistance) / np.linalg.norm(delta) < .02
    d, alpha, offset = profile(np.array([2., 3.]), np.array([1., 0.]), np.array([0., 1.]), 1.)
    assert_allclose([d, alpha, offset], [2., 2., 1.])
    d, alpha, _ = profile(np.array([-2., 3.]), np.array([1., 0.]), np.array([0., 1.]), 1., True)
    assert_allclose([d, alpha], [4., 0.])
    resistance, status = short_boundary(np.array([1., 0.]), np.array([0., 100.]),
                                       np.zeros(2), 0., .5)
    assert status == 'within_grid'
    assert_allclose(resistance, 100.)
    covariance = colored_covariance(np.array([np.eye(2), np.eye(2) * .4]), 4, 1, [0, 1])
    assert_allclose(np.diag(covariance), 1.)
    assert_allclose(covariance[0, 2], .2, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.
    return {'electric_KL': d_electric, 'thermal_KL': d_thermal, 'profiled_KL': d_profiled}


if __name__ == '__main__':
    print(check())
