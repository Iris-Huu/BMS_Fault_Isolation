"""Regression checks for integrated symbolic structural analysis."""
from pathlib import Path
import sys
import io
import unittest
import numpy as np
import sympy as sym
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.structural_model import build_battery_model
from src.structural import select_tests_with_detection,save_dm_plot
from src.preparation import file_digest
from src.model import simulate
import json


class StructuralChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models={thermal:build_battery_model(thermal) for thermal in [False,True]}

    def test_symbolic_and_numerical_models_are_separate(self):
        self.assertTrue(callable(simulate))
        manifest=json.loads((ROOT/'results/week3/preparation/freeze_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(file_digest(ROOT/'src/model.py'),manifest['files']['src/model.py'])

    def test_faults_appear_once_and_short_heat_uses_actual_current(self):
        for thermal,model in self.models.items():
            self.assertTrue(np.all(np.sum(model.F!=0,axis=0)==1))
            self.assertEqual(model.X.shape,(14,12) if thermal else (11,10))
            self.assertEqual(np.flatnonzero(model.F[:,list(model.f).index('f_isc')]).tolist(),[1])
        model=self.models[True];relation=model.syme[11]
        heat=relation.lhs-relation.rhs if isinstance(relation,sym.Equality) else relation
        symbols={str(x):x for x in heat.free_symbols}
        self.assertIn('I_cell',symbols);self.assertIn('i_s',symbols)
        self.assertNotIn('y_i',symbols);self.assertNotIn('f_isc',symbols)
        self.assertEqual(sym.diff(heat,symbols['i_s']),-symbols['v'])
        charge=model.syme[2];self.assertIn('I_cell',{str(x) for x in charge.free_symbols})

    def test_reference_mso_counts_signatures_and_causality(self):
        for thermal,model in self.models.items():
            msos=model.MSO();self.assertEqual(len(msos),6 if thermal else 1)
            self.assertTrue(all(model.IsLowIndex(s) for s in msos))
            fsm=np.asarray(model.FSM(msos),int)
            if thermal:self.assertTrue(np.all(np.sum(fsm==0,axis=0)==1))
            else:self.assertTrue(np.all(fsm==1))
            expected=np.eye(6,dtype=int) if thermal else np.ones((5,5),dtype=int)
            for causality in ['int','mixed']:
                np.testing.assert_array_equal(model.IsolabilityAnalysis(causality=causality,permute=False),expected)
            for s in msos:
                unknowns=np.sum(np.any(model.X[s,:]!=0,axis=0))
                self.assertEqual(len(s)-unknowns,1)

    def test_selection_retains_detection_without_isolable_pairs(self):
        for model in self.models.values():
            low=[s for s in model.MSO() if model.IsLowIndex(s)]
            chosen=select_tests_with_detection(model,low)
            self.assertEqual(len(chosen),6 if len(model.f)==6 else 1)
            fsm=np.asarray(model.FSM([low[k] for k in chosen]))
            self.assertTrue(np.all(np.any(fsm!=0,axis=0)))
            np.testing.assert_array_equal(model.IsolabilityAnalysisFSM(fsm,permute=False),
                                          model.IsolabilityAnalysis(causality='mixed',permute=False))

    def test_nonthermal_dm_plot_handles_empty_square_block(self):
        buffer=io.BytesIO()
        result=save_dm_plot(self.models[False],buffer,'Nonthermal reference')
        self.assertEqual(result['just_determined_equations'],0)
        self.assertEqual(result['overdetermined_equations'],11)
        buffer.seek(0)
        with Image.open(buffer) as image:image.verify()



if __name__=='__main__':unittest.main(verbosity=2)
