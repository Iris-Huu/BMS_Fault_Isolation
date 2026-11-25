"""Local Gaussian fault distinguishability conditional on measured current."""
import numpy as np
from scipy.linalg import solve_triangular

from src.model import simulate


def window_signatures(time, current, ambient, parameters, ocv, initial, step=1e-4):
    """Differentiate the frozen trajectory in A, siemens and degrees Celsius."""
    def output(faults=None, offset=0.):
        response = simulate(time, current, ambient + offset, parameters, ocv,
                            soc0=initial[0], polarization0=initial[1:3],
                            temperature0=initial[3], faults=faults)
        return response[['voltage_v', 'temperature_c']].to_numpy()

    healthy = output()
    bias = (output({'fi': step}) - output({'fi': -step})) / (2 * step)
    # One-sided stencil keeps short conductance physical.
    short = (4 * (output({'fisc': step}) - healthy)
             - (output({'fisc': 2 * step}) - healthy)) / (2 * step)
    chamber = (output(offset=step) - output(offset=-step)) / (2 * step)
    return np.stack([bias, short, chamber], axis=-1), healthy


def colored_covariance(gamma, sample_count, stride, channels):
    """Bartlett-tapered block Toeplitz covariance with spectral PSD repair."""
    gamma = gamma[::stride, :, :][:, channels][:, :, channels]
    lags = np.arange(len(gamma))
    taper = 1. - lags / len(gamma)
    scale = np.sqrt(np.diag(gamma[0]))
    scaled = gamma * taper[:, None, None] / np.outer(scale, scale)
    # Positive lags describe E[e(t)e(t+k)^T]. Build a circulant embedding.
    length = 2 * max(len(gamma), sample_count)
    sequence = np.zeros((length, len(channels), len(channels)))
    sequence[:len(gamma)] = scaled
    if len(gamma) > 1:
        sequence[-(len(gamma) - 1):] = scaled[1:][::-1].swapaxes(1, 2)
    spectrum = np.fft.fft(sequence, axis=0)
    values, vectors = np.linalg.eigh(spectrum)
    repaired = (vectors * np.maximum(values, 1e-6)[:, None, :]) @ vectors.conj().swapaxes(1, 2)
    sequence = np.fft.ifft(repaired, axis=0).real * np.outer(scale, scale)
    # Preserve empirical lag-zero marginal variances following PSD repair.
    correction = scale / np.sqrt(np.diag(sequence[0]))
    sequence *= np.outer(correction, correction)
    indices = np.arange(sample_count)
    blocks = sequence[(indices[None, :] - indices[:, None]) % length]
    covariance = blocks.transpose(0, 2, 1, 3).reshape(sample_count * len(channels), -1)
    return (covariance + covariance.T) / 2


def whiten(signatures, factor, channels, independent):
    selected = signatures[:, channels, :]
    if independent:
        return solve_triangular(factor, selected.transpose(1, 0, 2).reshape(len(channels), -1),
                                lower=True).reshape(len(channels), -1, 3).transpose(1, 0, 2).reshape(-1, 3)
    return solve_triangular(factor, selected.reshape(-1, 3), lower=True)


def profile(target, rival, ambient, ambient_limit, rival_nonnegative=False):
    """Return KL distance and closest rival/ambient shift in whitened coordinates."""
    rival_norm = rival @ rival
    perpendicular_target = target.copy()
    perpendicular_ambient = ambient.copy()
    if rival_norm > 0:
        perpendicular_target -= rival * ((rival @ target) / rival_norm)
        perpendicular_ambient -= rival * ((rival @ ambient) / rival_norm)
    ambient_norm = perpendicular_ambient @ perpendicular_ambient
    offset = np.clip((perpendicular_target @ perpendicular_ambient) / ambient_norm,
                     -ambient_limit, ambient_limit) if ambient_norm > 0 else 0.
    amplitude = (rival @ (target - offset * ambient)) / rival_norm if rival_norm > 0 else 0.
    if rival_nonnegative and amplitude < 0:
        ambient_norm = ambient @ ambient
        offset = np.clip((target @ ambient) / ambient_norm,
                         -ambient_limit, ambient_limit) if ambient_norm > 0 else 0.
        amplitude = 0.
    difference = target - amplitude * rival - offset * ambient
    return float(difference @ difference / 2), float(amplitude), float(offset)


def short_boundary(bias, short, ambient, ambient_limit, threshold):
    """Invert monotonic profiled KL for positive conductance."""
    from scipy.optimize import brentq

    def margin(conductance):
        return profile(conductance * short, bias, ambient, ambient_limit)[0] - threshold

    # Restrict the boundary to the modeled 10–1000 ohm injection span.
    if margin(.001) >= 0:
        return 1000., 'at_least_1000'
    if margin(.1) < 0:
        return np.nan, 'below_10'
    return 1. / brentq(margin, .001, .1, xtol=1e-12), 'within_grid'
