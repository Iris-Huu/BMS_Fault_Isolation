"""Evidence and noise preparation for a fixed numerical battery model.

No parameter optimizer is called here. Time stamps are instrument-local times.
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def rest_tail_metrics(meas, target_start, tail_s=600., max_gap_s=300.):
    """Use only a terminal zero-current run strictly before the target test."""
    stamp = pd.to_datetime(np.ravel(meas['TimeStamp']), format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    current = np.ravel(meas['Current']).astype(float)
    temperature = np.ravel(meas['Battery_Temp_degC']).astype(float)
    valid = (~stamp.isna()) & (stamp < target_start) & np.isfinite(current) & np.isfinite(temperature)
    ids = np.flatnonzero(valid)
    # Pause exports repeat the terminal timestamp; keep its first record.
    ids = ids[~stamp[ids].duplicated(keep='first')]
    if len(ids) < 2 or abs(current[ids[-1]]) > .02:
        return None
    # End-of-file rest only: never take a rest followed by charging before the test.
    end = len(ids)-1
    begin = end
    while begin > 0:
        a, b = ids[begin-1], ids[begin]
        dt = (stamp[b]-stamp[a]).total_seconds()
        if abs(current[a]) > .02 or not 0 < dt <= 180:
            break
        begin -= 1
    ids = ids[begin:]
    duration = (stamp[ids[-1]]-stamp[ids[0]]).total_seconds()
    age = (target_start-stamp[ids[-1]]).total_seconds()
    tail = ids[(stamp[ids]-stamp[ids[-1]]).total_seconds() >= -tail_s]
    if len(tail) < 2:
        return None
    x = (stamp[tail]-stamp[tail[0]]).total_seconds().to_numpy()/60
    y = temperature[tail]
    slope = float(np.polyfit(x, y, 1)[0]) if x[-1] > 0 else float('inf')
    spread = float(np.ptp(y))
    flags=[]
    if age > max_gap_s: flags.append('stale_preceding_rest')
    if duration < 1800: flags.append('short_rest')
    if x[-1]*60 < 300 or len(tail) < 6: flags.append('short_tail')
    if abs(slope) > .01: flags.append('tail_drift')
    if spread > .45: flags.append('tail_range')
    return dict(rest_end=str(stamp[ids[-1]]), gap_to_test_s=age,
                rest_duration_s=duration, tail_samples=len(tail),
                tail_median_temperature_c=float(np.median(y)),
                tail_slope_c_min=slope, tail_range_c=spread,
                qualified=not flags, flags=';'.join(flags),
                interpretation='pre-test cell-temperature proxy; equilibrium and common thermometry bias remain assumptions')


def hppc_rising_edges(frame):
    """Immediate/next-sample voltage drop for a transition from rest to discharge."""
    t=frame.Time.to_numpy(); i=frame.Current_discharge_positive.to_numpy(); v=frame.Voltage.to_numpy()
    edges=np.flatnonzero((i[1:]>.2)&(abs(i[:-1])<.05))+1
    rows=[]
    for k in edges:
        if k<2 or k+2>=len(t) or t[k+2]-t[k-2]>2.5:
            continue
        drop_now=float(v[k-1]-v[k]);drop_next=float(v[k-1]-v[k+1])
        if drop_next<=.01 or i[k]-i[k-1]<.5:
            continue
        fraction=drop_now/drop_next
        rows.append(dict(row=int(k),time_s=float(t[k]),current_step_a=float(i[k]-i[k-1]),
                         sample_interval_s=float(t[k]-t[k-1]),voltage_drop_now_v=drop_now,
                         voltage_drop_next_v=drop_next,immediate_fraction=fraction,
                         voltage_not_yet_jumped=bool(fraction<.25)))
    return pd.DataFrame(rows)


def lagged_autocovariance(samples, segments, max_lag):
    """Biased autocovariance, normalized by total N; never bridge time gaps.

    Gamma[k,a,b] = sum_t x[t,a]*x[t+k,b]/N. Common global mean is removed.
    This normalization preserves a positive-semidefinite window construction.
    """
    x=np.asarray(samples,float)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('Finite samples of shape [time,channel] required')
    x=x-x.mean(axis=0)
    channels=x.shape[1]; out=np.zeros((max_lag+1,channels,channels))
    for a,b in zip(segments[:-1],segments[1:]):
        part=x[int(a):int(b)]; n=len(part)
        if n<2:continue
        size=1<<(2*n-1).bit_length(); f=np.fft.rfft(part,n=size,axis=0)
        count=min(max_lag+1,n)
        for j in range(channels):
            for k in range(channels):
                out[:count,j,k]+=np.fft.irfft(np.conj(f[:,j])*f[:,k],n=size)[:count]/len(x)
    return out


def residual_noise_models(residuals, sensor_covariance):
    """Avoid adding sensor noise twice to measured Cycle3 prediction residuals.

    Residuals already include measurement noise. The PSD positive part of
    C_residual-C_sensor is called a model-discrepancy surrogate, not a pure
    independently observed process noise. Bias is kept separately.
    """
    sensor=np.asarray(sensor_covariance,float)
    raw=(residuals+residuals.T)/2
    delta=raw-sensor
    values,vectors=np.linalg.eigh(delta)
    model=(vectors*np.maximum(values,0))@vectors.T
    return model,sensor+model


def effective_sample_counts(gamma, windows_s):
    """Conservative scalar ESS at a declared 1-second cadence.

    Var(mean) uses Bartlett weights. Negative-correlation gains are capped at N.
    These summaries do not replace the full multichannel covariance.
    """
    rows=[]
    for seconds in windows_s:
        n=int(seconds)
        for channel,label in enumerate(['voltage_V','temperature_C']):
            variance=float(gamma[0,channel,channel])
            lag=min(n-1,len(gamma)-1)
            rho=gamma[1:lag+1,channel,channel]/max(variance,1e-30)
            inflation=float(1+2*np.sum((1-np.arange(1,lag+1)/n)*rho))
            inflation=max(1.,inflation)
            rows.append(dict(window_s=n,channel=label,nominal_samples=n,
                             variance_inflation=inflation,effective_samples=n/inflation))
    return pd.DataFrame(rows)


def file_digest(path):
    """Stable text hashes across Git LF/CRLF checkouts; binary files stay exact."""
    path=Path(path);content=path.read_bytes()
    if path.suffix.lower() in ['.py','.json','.csv','.txt','.md']:
        content=content.decode('utf-8-sig').replace('\r\n','\n').encode('utf-8')
    return hashlib.sha256(content).hexdigest()
