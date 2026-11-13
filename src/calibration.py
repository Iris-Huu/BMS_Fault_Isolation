"""Calibration helpers. Every assumption is exported with the run results."""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.isotonic import IsotonicRegression
from scipy.signal import lfilter
from scipy.optimize import lsq_linear
from scipy.integrate import cumulative_trapezoid


def prepare_dynamic_calibration(frame,ocv,tau1,tau2,reference_temp,activations):
    """Training design uses measured calibration temperature, never validation outputs."""
    t=frame.Time.to_numpy();i=frame.Current_discharge_positive.to_numpy()
    grid=np.arange(t[0],t[-1]+.05,.1)
    gi=np.interp(grid,t,i)
    filters=[]
    for tau in [tau1,tau2]:
        a=np.exp(-.1/tau)
        filters.append(np.interp(t,grid,lfilter([1-a],[1,-a],np.r_[0.,gi[:-1]])))
    # A consistent charge coordinate, not a claim of equilibrium capacity.
    soc=1-cumulative_trapezoid(i,t,initial=0)/(3600*2.9)
    knots=np.linspace(0,1,6)
    clipped=np.clip(soc,0,1)
    hats=np.maximum(0,1-np.abs(clipped[:,None]-knots)/.2)
    thermal=[np.exp(b*(1/(frame.Battery_Temp_degC.to_numpy()+273.15)-1/(reference_temp+273.15))) for b in activations]
    baseline=np.interp(soc,ocv.soc,ocv.ocv_v)
    return dict(time=t,current=i,previous_current=np.r_[i[0],i[:-1]],soc=soc,hats=hats,
                thermal=thermal,filters=filters,ocv=baseline,voltage=frame.Voltage.to_numpy(),
                temperature=frame.Battery_Temp_degC.to_numpy(),ambient=frame.Chamber_Temp_degC.to_numpy())


def dynamic_design(record,blend):
    hats=record["hats"]
    current=(1-blend)*record["current"]+blend*record["previous_current"]
    return np.column_stack([hats*(current*record["thermal"][0])[:,None],
                            hats*(record["filters"][0]*record["thermal"][1])[:,None],
                            hats*(record["filters"][1]*record["thermal"][2])[:,None],-hats])


def fit_dynamic_electrical(records,prior,reference_temp,activations,tau1,tau2):
    """Cycle1 fit / Cycle2 readout-blend selection, then refit both calibration cycles.

Coefficients are bounded positive resistances plus bounded OCV corrections.
Hyperparameters and priors are never selected using US06/UDDS errors.
"""
    scale=np.r_[prior,np.full(6,.01)]
    center=np.r_[np.ones(18),np.zeros(6)]
    lower=np.r_[np.full(18,.25),np.full(6,-3.)]
    upper=np.r_[np.full(18,4.),np.full(6,3.)]
    def fit(record_list,blend):
        aa=[];yy=[]
        for record in record_list:
            # About two samples/second; keep original sampled values, not smoothed outputs.
            keep=np.r_[True,np.floor(record["time"][1:]*2)!=np.floor(record["time"][:-1]*2)]
            aa.append(dynamic_design(record,blend)[keep]*scale)
            yy.append((record["ocv"]-record["voltage"])[keep])
        a=np.vstack(aa);y=np.concatenate(yy)
        regularization=np.diag(np.r_[np.full(18,.0015),np.full(6,.003)])*np.sqrt(len(y))
        weights=np.ones(len(y));solution=center.copy()
        for iteration in range(4):
            gram=a.T@(weights[:,None]*a)+regularization.T@regularization
            rhs=a.T@(weights*y)+regularization.T@regularization@center
            chol=np.linalg.cholesky(gram+1e-12*np.eye(24))
            transformed_target=np.linalg.solve(chol,rhs)
            fitted=lsq_linear(chol.T,transformed_target,bounds=(lower,upper),tol=1e-10,max_iter=200)
            solution=fitted.x
            residual=a@solution-y
            weights=np.minimum(1,.025/np.maximum(np.abs(residual),1e-12))
        return solution*scale
    scores=[]
    second=records[1]
    for blend in [0.,.25,.5,.75,1.]:
        coefficients=fit(records[:1],blend)
        residual=second["ocv"]-dynamic_design(second,blend)@coefficients-second["voltage"]
        scores.append(dict(blend=blend,calibration_cycle2_rmse_mv=float(np.sqrt(np.mean(residual**2))*1000)))
    chosen=min(scores,key=lambda r:r["calibration_cycle2_rmse_mv"])["blend"]
    coefficients=fit(records,chosen)
    result=dict(reference_temperature_c=float(reference_temp),soc_knots=np.linspace(0,1,6).tolist(),
                R0_ohm=coefficients[:6].tolist(),R1_ohm=coefficients[6:12].tolist(),
                R2_ohm=coefficients[12:18].tolist(),ocv_correction_v=coefficients[18:].tolist(),
                R0_activation_k=float(activations[0]),R1_activation_k=float(activations[1]),R2_activation_k=float(activations[2]),
                tau1_s=float(tau1),tau2_s=float(tau2))
    return result,chosen,scores


def calibrated_heat(record,resistance_model):
    coeff=np.r_[resistance_model["R0_ohm"],resistance_model["R1_ohm"],resistance_model["R2_ohm"],resistance_model["ocv_correction_v"]]
    # Heat uses physical instantaneous voltage; the readout blend never creates heat.
    physical_voltage=record["ocv"]-dynamic_design(record,0)@coeff
    equilibrium=record["ocv"]+record["hats"]@np.asarray(resistance_model["ocv_correction_v"])
    return record["current"]*(equilibrium-physical_voltage)


def fit_thermal_offset(records,offset_prior):
    """Fit forward trajectories with an explicit effective ambient/thermometry offset.

Offset is a healthy calibration nuisance, not an identified chamber temperature
or independently measured sensor bias. It remains uncertain and is profiled.
"""
    prepared=[]
    for record,heat in records:
        t=record["time"];grid=np.arange(t[0],t[-1],1.)
        heat_integral=cumulative_trapezoid(heat,t,initial=0)
        energy=np.interp(np.r_[grid,grid[-1]+1],t,heat_integral)
        averaged_heat=np.diff(energy)
        prepared.append((np.interp(grid,t,record["temperature"]),np.interp(grid,t,record["ambient"]),averaged_heat))
    def residual(parameters,fixed_offset=None):
        capacity,transfer=np.exp(parameters[:2])
        offset=parameters[2] if fixed_offset is None else fixed_offset
        a=np.exp(-transfer/capacity)
        errors=[]
        for measured,ambient,heat in prepared:
            forcing=ambient+offset+heat/transfer
            predicted=np.r_[measured[0],lfilter([1-a],[1,-a],forcing[:-1],zi=[a*measured[0]])[0]]
            errors.append(predicted-measured)
        combined=np.concatenate(errors)
        return np.r_[combined,.2*np.sqrt(len(combined))*(offset-offset_prior)]
    fitted=least_squares(residual,[np.log(60),np.log(.12),offset_prior],
                         bounds=([np.log(10),np.log(.005),-2],[np.log(200),np.log(.6),2]),
                         loss="soft_l1",f_scale=.25,max_nfev=150)
    c,h=np.exp(fitted.x[:2]);offset=float(fitted.x[2])
    profiles=[]
    for delta in [-.5,0.,.5]:
        fixed=offset+delta
        trial=least_squares(lambda z:residual(z,fixed),fitted.x[:2],
                            bounds=([np.log(10),np.log(.005)],[np.log(200),np.log(.6)]),
                            loss="soft_l1",f_scale=.25,max_nfev=100)
        cc,hh=np.exp(trial.x)
        profiles.append(dict(effective_ambient_offset_c=fixed,Cth_J_K=float(cc),hA_W_K=float(hh)))
    return dict(Cth_J_K=float(c),hA_W_K=float(h),effective_ambient_offset_c=offset,
                offset_prior_c=float(offset_prior),calibration_rmse_c=float(np.sqrt(np.mean(residual(fitted.x)[:-1]**2))),
                sensitivity_profiles=profiles,
                interpretation="joint effective ambient/thermometry offset, not an independent sensor-bias measurement")


def pulse_windows(df, current_threshold=.2, max_gap_s=2.):
    t=df.Time.to_numpy(); i=df.Current_discharge_positive.to_numpy(); ah=df.Ah.to_numpy()
    on=i>current_threshold
    b=np.r_[0,np.flatnonzero(np.diff(on))+1,len(df)]
    pulses=[(a,z) for a,z in zip(b[:-1],b[1:]) if on[a]]
    windows=[]
    for k,(a,z) in enumerate(pulses):
        end=pulses[k+1][0] if k+1<len(pulses) else len(df)
        if end>z:
            cut=np.flatnonzero((np.diff(t[z:end])>max_gap_s)|(np.abs(np.diff(ah[z:end]))>1e-4))
            if len(cut): end=z+int(cut[0])+1
        windows.append(dict(pulse=k+1,start=a,stop=z,end=end,
                            duration_s=float(t[z-1]-t[a]),current_a=float(np.median(i[a:z])),
                            rest_s=float(t[end-1]-t[z]) if end>z else 0.))
    return windows


def rested_ocv(df, windows, capacity_ah=2.9):
    points=[]
    for w in windows:
        a,z,end=w["start"],w["stop"],w["end"]
        if a>0 and (a==windows[0]["start"] or df.Time.iloc[a]-df.Time.iloc[a-1]<2):
            points.append((1+float(df.Ah.iloc[a-1]-df.Ah.iloc[0])/capacity_ah,
                           float(df.Voltage.iloc[a-1]),"pre_pulse",w["pulse"]))
        if w["rest_s"]>=600:
            tail=df.iloc[z:end]
            tail=tail[tail.Time>=tail.Time.iloc[-1]-60]
            points.append((1+float(tail.Ah.median()-df.Ah.iloc[0])/capacity_ah,
                           float(tail.Voltage.median()),"rest_tail",w["pulse"]))
    raw=pd.DataFrame(points,columns=["soc","ocv_v","source_kind","pulse"])
    grouped=raw.groupby("soc",as_index=False).ocv_v.median().sort_values("soc")
    grouped["ocv_v"]=IsotonicRegression(increasing=True).fit_transform(grouped.soc,grouped.ocv_v)
    return grouped,raw


def fit_joint_pulse(df,w):
    """Fit pulse loading and recovery together with an Ah-dependent baseline.

Bounds separate two RC time scales. Switching times are midpoint estimates;
samples within 0.25s of switches are excluded to limit logging skew effects.
"""
    a,z,end=w["start"],w["stop"],w["end"]
    if a==0 or end<=z: raise ValueError("Missing pulse context")
    t=df.Time.to_numpy(); v=df.Voltage.to_numpy(); ah=df.Ah.to_numpy()
    current=df.Current_discharge_positive.to_numpy()
    start=(t[a-1]+t[a])/2
    stop=(t[z-1]+t[z])/2
    length=stop-start
    loaded=np.arange(a,z)
    # Use a time-balanced recovery sample, never bridge omitted intervals.
    rest=np.arange(z,end)
    targets=np.geomspace(max(t[z]-stop,.05),t[end-1]-stop,100)
    picked=rest[np.clip(np.searchsorted(t[rest]-stop,targets),0,len(rest)-1)]
    idx=np.unique(np.r_[loaded,picked])
    idx=idx[(np.abs(t[idx]-start)>.25)&(np.abs(t[idx]-stop)>.25)]
    x=t[idx]-start
    delta_ah=ah[idx]-ah[a-1]
    im=current[idx]
    y=v[idx]
    baseline=float(v[a-1])
    nominal=w["current_a"]
    def predict(q):
        r0,r1,r2,tau1,tau2=np.exp(q[:5])
        u=np.minimum(np.maximum(x,0),length)
        after=np.maximum(x-length,0)
        pol=nominal*(r1*(-np.expm1(-u/tau1))*np.exp(-after/tau1)
                     +r2*(-np.expm1(-u/tau2))*np.exp(-after/tau2))
        return baseline+q[5]+q[6]*delta_ah-r0*im-pol
    lo=np.r_[np.log([.001,.00005,.00005,.5,15]),-.02,0.]
    hi=np.r_[np.log([.3,.3,.5,15,600]),.02,8.]
    fits=[]
    for fast,slow in [(1,25),(5,80),(10,250)]:
        q=np.r_[np.log([.025,.008,.02,fast,slow]),0.,.5]
        result=least_squares(lambda p:predict(p)-y,q,bounds=(lo,hi),loss="soft_l1",
                             f_scale=.002,max_nfev=350)
        fits.append(result)
    best=min(fits,key=lambda r:np.sum((predict(r.x)-y)**2))
    r0,r1,r2,tau1,tau2=np.exp(best.x[:5])
    error=predict(best.x)-y
    flags=[]
    if np.any(np.minimum(best.x[:5]-lo[:5],hi[:5]-best.x[:5])<.02): flags.append("parameter_near_bound")
    if not best.success: flags.append("optimizer_limit")
    if tau2/tau1<2: flags.append("weak_time_scale_separation")
    if abs(best.x[5])>.01: flags.append("baseline_shift_over_10mV")
    return dict(pulse=w["pulse"],current_a=nominal,
                soc_reference=1+float(ah[a]-ah[0])/2.9,
                R0_ohm=r0,R1_ohm=r1,C1_F=tau1/r1,R2_ohm=r2,C2_F=tau2/r2,
                tau1_s=tau1,tau2_s=tau2,baseline_shift_v=best.x[5],ocv_slope_v_ah=best.x[6],
                rmse_mv=float(np.sqrt(np.mean(error**2))*1000),
                max_error_mv=float(np.max(np.abs(error))*1000),
                jacobian_condition=float(np.linalg.cond(best.jac)),flags=";".join(flags)),(idx,predict(best.x))


def covariance_from_rests(df,windows):
    """Detrended raw-sample covariance, plus separate effective quantization floor.

Channel order is [current A, voltage V, temperature C]. No interpolation.
The large temperature jump scale is empirical, not a proven ADC LSB.
"""
    pieces=[]; lag_pairs=[]; steps=[]; small_steps=[]
    for w in windows:
        if w["rest_s"]<600: continue
        part=df.iloc[w["stop"]:w["end"]]
        part=part[part.Time>=part.Time.iloc[-1]-120]
        if len(part)<30: continue
        t=part.Time.to_numpy(); x=part[["Current_discharge_positive","Voltage","Battery_Temp_degC"]].to_numpy()
        design=np.c_[np.ones(len(t)),t-t.mean()]
        residual=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
        pieces.append(residual)
        lag_pairs.append((residual[:-1],residual[1:]))
        delta=np.abs(np.diff(x[:,2])); steps.extend(delta[(delta>.05)&(delta<.5)])
        unique=np.diff(np.unique(x[:,2])); small_steps.extend(unique[unique>1e-7])
    if len(pieces)<3: raise ValueError("Insufficient independent rest windows")
    residual=np.vstack(pieces)
    sigma=np.cov(residual,rowvar=False)
    # Some resting-current channels log exactly zero. This is a censoring/data
    # resolution issue, NOT evidence of zero current-sensor noise.
    q_effective=float(np.median(steps)) if len(steps) else None
    active_variances=[]
    for w in windows:
        part=df.iloc[w["start"]:w["stop"]]
        if len(part)<30 or w["duration_s"]<9: continue
        part=part[(part.Time>part.Time.iloc[0]+1)&(part.Time<part.Time.iloc[-1]-1)]
        t=part.Time.to_numpy()
        if len(t)<20: continue
        signal=part.Current_discharge_positive.to_numpy()
        design=np.c_[np.ones(len(t)),t-t.mean()]
        noise_proxy=signal-design@np.linalg.lstsq(design,signal,rcond=None)[0]
        active_variances.append(float(np.var(noise_proxy,ddof=2)))
    active_current_variance=float(np.median(active_variances)) if active_variances else 0.
    floor=np.diag([max(0.,active_current_variance-sigma[0,0]),0.,q_effective**2/12 if q_effective else 0.])
    # Tiny documented numerical regularization; not a current noise estimate.
    regularization=np.diag([1e-12,1e-12,1e-12])
    effective=sigma+floor+regularization
    prev=np.vstack([p[0] for p in lag_pairs]); nxt=np.vstack([p[1] for p in lag_pairs])
    lag=[]
    for k in range(3):
        den=np.sqrt(np.sum(prev[:,k]**2)*np.sum(nxt[:,k]**2))
        lag.append(float(np.sum(prev[:,k]*nxt[:,k])/den) if den>1e-15 else None)
    info=dict(channels=["current_A","voltage_V","temperature_C"],rest_windows=len(pieces),samples=len(residual),
              lag1_correlation=lag,temperature_large_jump_scale_c=q_effective,
              temperature_min_observed_spacing_c=float(min(small_steps)) if small_steps else None,
              current_noise_unidentified=bool(sigma[0,0]<1e-12),
              active_plateau_current_std_proxy_a=float(np.sqrt(active_current_variance)),
              current_proxy_definition="median detrended current variance on constant-current pulse interiors; includes controller ripple, not a pure sensor-noise estimate",
              temperature_floor_definition="empirical large-jump scale squared / 12; sensitivity assumption, not proven uniform ADC noise",
              limitations=["Demeaning/detrending removes local drift, not all process noise.",
                           "Quantization and serial correlation violate the untested iid-Gaussian assumption.",
                           "Effective covariance adds a conservative temperature sensitivity floor and numerical regularization.",
                           "If resting current is identically zero, current-sensor noise remains unidentified."])
    return sigma,effective,info,residual


def fit_thermal_integrals(records):
    """Identify alpha=1/Cth and beta=hA/Cth from 60s integral balances.

Each record is (time, measured temperature, ambient, heat). Both excited and
resting windows are included. Actual irregular sampling intervals are retained.
"""
    rows=[]; targets=[]
    for time,temp,ambient,heat in records:
        t=np.asarray(time); temp=np.asarray(temp); ambient=np.asarray(ambient); heat=np.asarray(heat)
        boundaries=np.unique(np.r_[0,np.searchsorted(t,np.arange(t[0]+60,t[-1],60)),len(t)-1])
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            if b-a<2 or t[b]-t[a]<30 or np.max(np.diff(t[a:b+1]))>15: continue
            rows.append([np.trapezoid(heat[a:b+1],t[a:b+1]),
                         -np.trapezoid(temp[a:b+1]-ambient[a:b+1],t[a:b+1])])
            targets.append(temp[b]-temp[a])
    design=np.asarray(rows); target=np.asarray(targets)
    if len(target)<10: raise ValueError("Insufficient thermal calibration windows")
    fitted=least_squares(lambda p:design@p-target,[1/50,.001],
                         bounds=([1/300,1/20000],[1/5,1/20]),loss="soft_l1",f_scale=.1)
    alpha,beta=fitted.x
    corr=float(np.corrcoef(design.T)[0,1])
    return dict(Cth_J_K=float(1/alpha),hA_W_K=float(beta/alpha),
                thermal_tau_s=float(1/beta),integral_window_count=len(target),
                integral_rmse_c=float(np.sqrt(np.mean((design@fitted.x-target)**2))),
                design_column_correlation=corr,
                at_bound=bool(np.any(fitted.active_mask)),
                design_condition=float(np.linalg.cond(design)))


def fit_thermal_entropy(records):
    """Sensitivity model with effective k(SOC)=k0+k1*(SOC-.5), not measured entropy.

Fit on calibration cycles only; weak regularization keeps correction small.
Each record is (time, Tcell, Tamb, irreversible_heat, SOC, current).
"""
    rows=[];targets=[]
    for time,temp,ambient,heat,soc,current in records:
        b=np.unique(np.r_[0,np.searchsorted(time,np.arange(time[0]+60,time[-1],60)),len(time)-1])
        for a,z in zip(b[:-1],b[1:]):
            if z-a<2 or time[z]-time[a]<30 or np.max(np.diff(time[a:z+1]))>15:continue
            sl=slice(a,z+1);t=time[sl]
            rev=current[sl]*(temp[sl]+273.15)
            rows.append([np.trapezoid(heat[sl],t),-np.trapezoid(temp[sl]-ambient[sl],t),
                         np.trapezoid(rev,t),np.trapezoid(rev*(soc[sl]-.5),t)])
            targets.append(temp[z]-temp[a])
    x=np.asarray(rows);y=np.asarray(targets)
    def residual(p):
        c,h=np.exp(p[:2]);k0,k1=p[2:]
        prediction=x@np.array([1/c,h/c,k0/c,k1/c])
        return np.r_[prediction-y,.1*k0/.0005,.1*k1/.001]
    result=least_squares(residual,[np.log(60),np.log(.1),0.,0.],
                         bounds=([np.log(5),np.log(.002),-.001,-.002],
                                 [np.log(300),np.log(1),.001,.002]),
                         loss="soft_l1",f_scale=.1,max_nfev=500)
    c,h=np.exp(result.x[:2]);k0,k1=result.x[2:]
    return dict(Cth_J_K=float(c),hA_W_K=float(h),entropy_k0_v_k=float(k0),entropy_k1_v_k=float(k1),
                integral_rmse_c=float(np.sqrt(np.mean(residual(result.x)[:-2]**2))),
                at_bound=bool(np.any(result.active_mask)),
                basis="effective heat correction fitted on Cycle1/2; not independent thermodynamic entropy measurement")
