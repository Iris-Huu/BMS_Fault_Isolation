"""Numerical 2RC + lumped-thermal cell model (separate from symbolic analysis).

Units: seconds, A (discharge positive), V, Ah, ohm, farad, Celsius, J/K, W/K.
Fault fi is added to the terminal-current measurement. fisc is conductance S.
The leakage variable is kept explicitly: i_s=fisc*v; Icell=i+i_s.
No measured output is fed back after initialization during validation.
"""
import numpy as np
import pandas as pd


def simulate(time, current_measured, ambient, parameters, ocv_table,
             soc0=1., temperature0=25., polarization0=(0., 0.), faults=None):
    time = np.asarray(time, dtype=float)
    current = np.asarray(current_measured, dtype=float)
    ambient = np.asarray(ambient, dtype=float)
    if not (len(time) == len(current) == len(ambient)) or len(time) < 2:
        raise ValueError("Input arrays must have matching lengths >=2")
    if not all(np.all(np.isfinite(a)) for a in [time,current,ambient]):
        raise ValueError("Inputs must be finite")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Time must strictly increase")
    p = parameters
    for key in ["Q_Ah","R0_ohm","R1_ohm","C1_F","R2_ohm","C2_F","Cth_J_K","hA_W_K"]:
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"Invalid positive parameter: {key}")
    f = dict(fi=0.,fv=0.,fT=0.,fisc=0.,fQ=0.,fR=0.,onset_s=-np.inf)
    if faults:
        f.update(faults)
    if f["fisc"]<0 or not 0<=f["fQ"]<1 or f["fR"]<=-1:
        raise ValueError("Nonphysical fault magnitude")
    soc_grid=np.asarray(ocv_table["soc"],dtype=float)
    ocv_grid=np.asarray(ocv_table["ocv_v"],dtype=float)
    if len(soc_grid)<2 or np.any(np.diff(soc_grid)<=0):
        raise ValueError("OCV SOC grid must strictly increase")
    state=np.array([soc0,*polarization0,temperature0],dtype=float)
    out=np.empty((len(time),9))
    electrical_map=p.get("electrical_map")
    resistance_model=p.get("resistance_soc_model")
    voltage_blend=float(p.get("voltage_readout_blend",0.))
    if not 0<=voltage_blend<=1: raise ValueError("Readout blend must be in [0,1]")
    previous_physical_voltage=None
    cached_key=None
    r0_base,r1,r2=p["R0_ohm"],p["R1_ohm"],p["R2_ohm"]
    tau1=r1*p["C1_F"];tau2=r2*p["C2_F"]
    for n in range(len(time)):
        active=time[n]>=f["onset_s"]
        if resistance_model:
            key=(float(state[0]),float(state[3])) if p.get("exact_parameter_lookup",False) else (round(float(state[0]),3),round(float(state[3]),1))
            if key!=cached_key:
                temperature_kelvin=max(key[1]+273.15,200.)
                reference_kelvin=resistance_model["reference_temperature_c"]+273.15
                rr=[]
                for branch in ["R0","R1","R2"]:
                    base=np.interp(key[0],resistance_model["soc_knots"],resistance_model[branch+"_ohm"])
                    rr.append(float(base*np.exp(resistance_model[branch+"_activation_k"]*(1/temperature_kelvin-1/reference_kelvin))))
                r0_base,r1,r2=rr
                tau1=resistance_model["tau1_s"];tau2=resistance_model["tau2_s"]
                cached_key=key
        if electrical_map:
            # Parameter lookup resolution, NOT sensor quantization. Caching at
            # 0.001 SOC / 0.1 C avoids millions of repeated interpolations.
            key=(round(float(state[0]),3),round(float(state[3]),1))
            if key!=cached_key:
                values=[]
                for curve in electrical_map:
                    values.append([np.interp(key[0],curve["soc"],curve[k])
                                   for k in ["R0_ohm","R1_ohm","R2_ohm","tau1_s","tau2_s"]])
                temperatures=[curve["cell_temperature_c"] for curve in electrical_map]
                values=np.asarray(values)
                r0_base,r1,r2,tau1,tau2=[float(np.interp(key[1],temperatures,values[:,k])) for k in range(5)]
                cached_key=key
        fi=f["fi"] if active else 0.
        conductance=f["fisc"] if active else 0.
        r0=r0_base*(1+(f["fR"] if active else 0.))
        q=p["Q_Ah"]*(1-(f["fQ"] if active else 0.))
        ocv=float(np.interp(state[0],soc_grid,ocv_grid))
        if resistance_model:
            ocv+=float(np.interp(state[0],resistance_model["soc_knots"],resistance_model["ocv_correction_v"]))
        terminal_i=current[n]-fi
        # Algebraic solution of v=E-R0*(i+fisc*v)-v1-v2.
        voltage=(ocv-state[1]-state[2]-r0*terminal_i)/(1+r0*conductance)
        leakage=conductance*voltage
        cell_i=terminal_i+leakage
        heat=cell_i*(ocv-voltage)+leakage*voltage
        # Optional effective reversible-heat sensitivity. Coefficients are V/K;
        # they are not a direct thermodynamic dOCV/dT measurement.
        entropy=p.get("entropy_k0_v_k",0.)+p.get("entropy_k1_v_k",0.)*(state[0]-.5)
        heat+=cell_i*(state[3]+273.15)*entropy
        observed_voltage=voltage if previous_physical_voltage is None else (1-voltage_blend)*voltage+voltage_blend*previous_physical_voltage
        previous_physical_voltage=voltage
        out[n]=[state[0],state[1],state[2],state[3],observed_voltage+(f["fv"] if active else 0.),
                state[3]+(f["fT"] if active else 0.),cell_i,leakage,heat]
        if n+1<len(time):
            dt=time[n+1]-time[n]
            # Exact RC update for a zero-order-held current. Heat uses the same
            # causal beginning-of-step approximation, not measured voltage.
            state[0]-=cell_i*dt/(3600*q)
            a1=np.exp(-dt/tau1); a2=np.exp(-dt/tau2)
            state[1]=a1*state[1]+r1*(1-a1)*cell_i
            state[2]=a2*state[2]+r2*(1-a2)*cell_i
            at=np.exp(-p["hA_W_K"]*dt/p["Cth_J_K"])
            effective_ambient=ambient[n]+p.get("effective_ambient_offset_c",0.)
            state[3]=effective_ambient+heat/p["hA_W_K"]+(state[3]-effective_ambient-heat/p["hA_W_K"])*at
    result=pd.DataFrame(out,columns=["soc","v1_v","v2_v","temperature_c","voltage_v",
                                    "temperature_measurement_c","cell_current_a","leakage_current_a","heat_w"])
    result["time_s"]=time
    result["ocv_outside_supported_soc"]=(out[:,0]<soc_grid[0])|(out[:,0]>soc_grid[-1])
    return result
