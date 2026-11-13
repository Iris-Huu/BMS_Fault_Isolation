"""Empirical colored-noise checks and explicit quantization sensitivity models.

Empirical raw-rest blocks are primary. Gaussian/quantized likelihoods are named
surrogates, not asserted instrument laws. Current noise cannot be separated
from controller ripple using these files alone.
"""
import numpy as np
from scipy.special import log_ndtr


def quantized_normal_log_probability(observed,mean,sigma,step):
    """Log probability of a bin [observed-step/2, observed+step/2]."""
    if sigma<=0 or step<=0:raise ValueError("Positive sigma and bin width required")
    lower=(np.asarray(observed)-np.asarray(mean)-step/2)/sigma
    upper=lower+step/sigma
    # Use complementary tails on the positive side to avoid subtracting ones.
    a=np.where(lower>=0,log_ndtr(-lower),log_ndtr(upper))
    b=np.where(lower>=0,log_ndtr(-upper),log_ndtr(lower))
    return a+np.log(-np.expm1(np.minimum(b-a,-np.finfo(float).eps)))


def separable_window_covariance(sigma,length,step_s,native_step_s,lag1):
    """PSD Gaussian comparison covariance with a declared common AR(1) kernel.

Order: [I0,V0,T0,I1,V1,T1,...]. This is not a white-noise or exact
non-Gaussian likelihood. Correlation below the measured native cadence is a
model extrapolation and must be included in sensitivity analysis.
"""
    sigma=np.asarray(sigma,dtype=float)
    if length<1 or length>1000 or step_s<=0 or native_step_s<=0:
        raise ValueError("Invalid window size or cadence")
    if sigma.shape!=(3,3) or np.linalg.eigvalsh(sigma).min() < -1e-12:
        raise ValueError("Expected positive-semidefinite 3x3 covariance")
    rho=float(np.clip(lag1,0,.999))
    distance=np.abs(np.arange(length)[:,None]-np.arange(length)[None,:])*step_s/native_step_s
    temporal=np.where(distance==0,1.,rho**distance)
    return np.kron(temporal,sigma)


def fit_empirical_rest_noise(frame,windows):
    blocks=[];cadences=[];large_steps=[];minimum_steps=[];pulse_current=[]
    for w in windows:
        if w["rest_s"]>=600:
            part=frame.iloc[w["stop"]:w["end"]]
            part=part[part.Time>=part.Time.iloc[-1]-120]
            if len(part)>=30:
                t=part.Time.to_numpy()
                data=part[["Current_discharge_positive","Voltage","Battery_Temp_degC"]].to_numpy()
                design=np.c_[np.ones(len(t)),t-t.mean()]
                residual=data-design@np.linalg.lstsq(design,data,rcond=None)[0]
                blocks.append(residual);cadences.append(float(np.median(np.diff(t))))
                jump=np.abs(np.diff(data[:,2]));large_steps.extend(jump[(jump>.05)&(jump<.5)])
                spacing=np.diff(np.unique(data[:,2]));minimum_steps.extend(spacing[spacing>1e-7])
        if w["duration_s"]>=9:
            part=frame.iloc[w["start"]:w["stop"]]
            part=part[(part.Time>part.Time.iloc[0]+1)&(part.Time<part.Time.iloc[-1]-1)]
            if len(part)>=20:
                t=part.Time.to_numpy();current=part.Current_discharge_positive.to_numpy()
                design=np.c_[np.ones(len(t)),t-t.mean()]
                pulse_current.extend(current-design@np.linalg.lstsq(design,current,rcond=None)[0])
    if len(blocks)<12:raise ValueError("Too few independent rest windows")
    training=[block for k,block in enumerate(blocks) if k%4!=3]
    test=[block for k,block in enumerate(blocks) if k%4==3]
    pooled=np.vstack(training);held=np.vstack(test)
    empirical=np.cov(pooled,rowvar=False)
    surrogate=empirical.copy()
    current_std=float(np.std(pulse_current,ddof=1))
    surrogate[0,0]=max(surrogate[0,0],current_std**2)
    surrogate+=np.eye(3)*1e-12
    correlations=[]
    for channel in range(3):
        x=np.concatenate([b[:-1,channel] for b in training]);y=np.concatenate([b[1:,channel] for b in training])
        den=np.sqrt(np.sum(x*x)*np.sum(y*y))
        correlations.append(float(np.sum(x*y)/den) if den>1e-15 else None)
    evaluations=[]
    for channel,label in [(1,"voltage_V"),(2,"temperature_C")]:
        for level in [.9,.95]:
            threshold=float(np.quantile(np.abs(pooled[:,channel]),level))
            # Simultaneous within-block interval, reported separately from point intervals.
            block_threshold=float(np.quantile([abs(b[:,channel]).max() for b in training],level))
            evaluations.append(dict(channel=label,nominal_fraction=level,point_halfwidth=threshold,
                                    held_out_point_coverage=float(np.mean(abs(held[:,channel])<=threshold)),
                                    block_halfwidth=block_threshold,
                                    held_out_block_coverage=float(np.mean([abs(b[:,channel]).max()<=block_threshold for b in test])),
                                    held_out_blocks=len(test)))
    q=float(np.median(large_steps)) if large_steps else None
    model=dict(channel_order=["current_A","voltage_V","temperature_C"],
               native_cadence_s=float(np.median(cadences)),training_blocks=len(training),test_blocks=len(test),
               lag1=correlations,common_lag1_surrogate=max([r for r in correlations if r is not None]+[0.]),
               empirical_covariance=empirical.tolist(),gaussian_surrogate_covariance=surrogate.tolist(),
               temperature_large_jump_bin_sensitivity_c=q,
               temperature_min_observed_spacing_c=float(min(minimum_steps)),
               current_plateau_fluctuation_std_proxy_a=current_std,
               current_sensor_noise_identified=False,
               point_and_block_coverage=evaluations,
               limitations=["Per-window linear detrending conditions on local drift.",
                            "Windows are not proven exchangeable; coverage is empirical, not guaranteed.",
                            "Pulse-current fluctuations include controller ripple; current sensor noise is not separately identified.",
                            "Large temperature jump is a sensitivity bin width, not a proven ADC quantizer.",
                            "Sub-native-cadence correlation is extrapolated, not directly measured."])
    # Store training blocks with their boundaries so resampling preserves dependence.
    boundaries=np.r_[0,np.cumsum([len(b) for b in training])]
    return model,pooled,boundaries


def sample_empirical_blocks(samples,boundaries,length,rng):
    """Draw complete contiguous blocks without interpolation or Gaussianization."""
    choices=np.flatnonzero(np.diff(boundaries)>=length)
    if not len(choices):raise ValueError("Requested window exceeds available native noise blocks")
    block=int(rng.choice(choices));a,b=int(boundaries[block]),int(boundaries[block+1])
    start=int(rng.integers(a,b-length+1))
    return np.asarray(samples[start:start+length]).copy()
