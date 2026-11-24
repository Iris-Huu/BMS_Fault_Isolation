"""Synthetic and interface checks for Week 3 data preparation."""
from pathlib import Path
import sys
import unittest
import hashlib
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.preparation import (file_digest,rest_tail_metrics,hppc_rising_edges,lagged_autocovariance,
                             residual_noise_models,effective_sample_counts)
from src.calibration import dynamic_design,voltage_readout
from src.model import simulate


class PreparationChecks(unittest.TestCase):
    def test_readout_design_matches_full_voltage_operator(self):
        rng=np.random.default_rng(42);n=50
        record=dict(hats=rng.random((n,6)),current=rng.normal(size=n),
                    thermal=[rng.random(n)+.5 for _ in range(3)],
                    filters=[rng.normal(size=n),rng.normal(size=n)],ocv=rng.random(n)+3.5)
        coefficients=rng.normal(size=24)*.01
        physical=record['ocv']-dynamic_design(record,0)@coefficients
        for blend in [0,.25,.75,1]:
            fitted=voltage_readout(record['ocv'],blend)-dynamic_design(record,blend)@coefficients
            np.testing.assert_allclose(fitted,voltage_readout(physical,blend),atol=1e-14)
        p=dict(Q_Ah=2.9,R0_ohm=.03,R1_ohm=.01,C1_F=200,R2_ohm=.02,C2_F=1500,Cth_J_K=60,hA_W_K=.1)
        t=np.arange(0,20,.1);i=np.sin(t)+1;ocv=pd.DataFrame(dict(soc=[0,1],ocv_v=[3.3,4.1]))
        a=simulate(t,i,np.full(len(t),25.),p,ocv)
        b=simulate(t,i,np.full(len(t),25.),dict(p,voltage_readout_blend=.75),ocv)
        np.testing.assert_allclose(b.voltage_v,voltage_readout(a.voltage_v,.75))
        np.testing.assert_allclose(b.heat_w,a.heat_w)

    def test_rest_is_causal_and_quality_gated(self):
        stamp=pd.date_range('2017-01-01',periods=61,freq='60s')
        m=dict(TimeStamp=stamp.strftime('%m/%d/%Y %I:%M:%S %p').to_numpy(),Current=np.zeros(61),
               Battery_Temp_degC=np.full(61,25.6))
        target=stamp[-1]+pd.Timedelta(seconds=2)
        r=rest_tail_metrics(m,target);self.assertTrue(r['qualified'])
        self.assertAlmostEqual(r['tail_median_temperature_c'],25.6)
        repeated=dict(m,TimeStamp=np.concatenate([m['TimeStamp'],m['TimeStamp'][-1:]]),
                      Current=np.zeros(62),Battery_Temp_degC=np.full(62,25.6))
        self.assertTrue(rest_tail_metrics(repeated,target)['qualified'])
        self.assertFalse(rest_tail_metrics(m,target+pd.Timedelta(hours=1))['qualified'])
        future=dict(m,TimeStamp=np.concatenate([m['TimeStamp'],[(target+pd.Timedelta(seconds=60)).strftime('%m/%d/%Y %I:%M:%S %p')]]),
                    Current=np.zeros(62),Battery_Temp_degC=np.r_[m['Battery_Temp_degC'],99.])
        self.assertAlmostEqual(rest_tail_metrics(future,target)['tail_median_temperature_c'],25.6)
        active=dict(m,Current=np.r_[np.zeros(60),1.]);self.assertIsNone(rest_tail_metrics(active,target))

    def test_immediate_and_delayed_edges_are_distinguished(self):
        t=np.arange(20)*.1;i=np.r_[np.zeros(5),np.full(15,3.)]
        now=np.r_[np.full(5,4.),3.91,np.full(14,3.90)]
        delayed=np.r_[np.full(6,4.),np.full(14,3.90)]
        def frame(v):return pd.DataFrame(dict(Time=t,Current_discharge_positive=i,Voltage=v))
        self.assertFalse(hppc_rising_edges(frame(now)).voltage_not_yet_jumped.iloc[0])
        self.assertTrue(hppc_rising_edges(frame(delayed)).voltage_not_yet_jumped.iloc[0])

    def test_colored_covariance_is_psd_and_never_bridges_gaps(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(100,2));cuts=[0,40,100]
        gamma=lagged_autocovariance(x,cuts,20)
        centered=x-x.mean(axis=0)
        manual=(centered[:39].T@centered[1:40]+centered[40:99].T@centered[41:])/100
        np.testing.assert_allclose(gamma[1],manual,atol=1e-14)
        k=np.empty((40,40))
        for a in range(20):
            for b in range(20):k[2*a:2*a+2,2*b:2*b+2]=gamma[b-a] if b>=a else gamma[a-b].T
        self.assertGreaterEqual(np.linalg.eigvalsh(k).min(),-1e-12)
        sensor=np.diag([1e-8,.0036]);model,total=residual_noise_models(gamma[0],sensor)
        self.assertGreaterEqual(np.linalg.eigvalsh(model).min(),-1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(total-gamma[0]).min(),-1e-12)
        np.testing.assert_allclose(total,sensor+model)

    def test_colored_samples_have_lower_ess(self):
        white=np.zeros((61,2,2));white[0]=np.eye(2)
        a=effective_sample_counts(white,[10,60]);np.testing.assert_allclose(a.effective_samples,a.nominal_samples)
        colored=white.copy()
        for k in range(61):colored[k]=np.eye(2)*.95**k
        b=effective_sample_counts(colored,[10,60]);self.assertTrue((b.effective_samples<b.nominal_samples).all())

    def test_frozen_model_and_noise_interfaces_match(self):
        folder=ROOT/'results/week3/preparation'
        manifest_path=folder/'freeze_manifest.json'
        if not manifest_path.exists():
            self.skipTest('No fixed-model interface generated yet')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        for filename,digest in manifest['files'].items():
            self.assertEqual(file_digest(ROOT/filename),digest,filename)
        for temp in [25,10,0]:
            path=folder/f'model_v1/{temp}C.json'
            data=json.loads(path.read_text(encoding='utf-8'))
            reference=json.loads((ROOT/data['source_parameters']).read_text(encoding='utf-8'))['parameters']
            expected=dict(reference,voltage_readout_blend=0.)
            self.assertEqual(data['parameters'],expected)
            interface=json.loads((folder/f'noise/{temp}C/noise_interface.json').read_text(encoding='utf-8'))
            self.assertEqual(interface['frozen_parameter_sha256'],file_digest(path))
            g=np.load(folder/f'noise/{temp}C/cycle3_colored_autocovariance.npz')['gamma']
            covariance=np.block([[g[b-a] if b>=a else g[a-b].T for b in range(20)] for a in range(20)])
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(),-1e-10)

    def test_healthy_thermal_scale_is_not_identifiable_from_temperature_alone(self):
        t=np.arange(600);heat=1+np.sin(t/20);ambient=25.
        def prediction(c,h,power):
            values=[25.]
            for q in power:
                a=np.exp(-h/c);values.append(ambient+q/h+(values[-1]-ambient-q/h)*a)
            return values
        np.testing.assert_allclose(prediction(60,.12,heat),prediction(45,.09,.75*heat),atol=1e-12)


if __name__=='__main__':unittest.main(verbosity=2)
