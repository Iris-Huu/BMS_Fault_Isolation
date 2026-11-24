"""Run thermal/nonthermal structural analysis without battery calibration."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.structural_model import build_battery_model
from src.structural import run_structural_analysis,draw_isolability


def run(root,out):
    if version('faultdiagnosistoolbox')!='0.12.5':
        raise RuntimeError('This structural reference is validated with faultdiagnosistoolbox==0.12.5')
    out.mkdir(parents=True,exist_ok=True)
    results={}
    for thermal,label in [(False,'nonthermal_model'),(True,'thermal_model')]:
        results[label]=run_structural_analysis(build_battery_model(thermal),label,out/label)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,label,title in zip(axes,['nonthermal_model','thermal_model'],['Without temperature','With temperature']):
        result=results[label];draw_isolability(result['mixed_matrix'],result['fault_names'],title,ax)
    fig.tight_layout();fig.savefig(out/'figure2_structural_isolability.png',dpi=200,bbox_inches='tight');plt.close(fig)
    summary=dict(toolbox_version=version('faultdiagnosistoolbox'),upstream_structural_commit='fc42cff4a1066331227ea5f6cbcc0c0246fb307e',
        symbolic_model='src/structural_model.py',numerical_model='src/model.py (unchanged)',
        result='Without the thermal channel fi/fisc are structurally inseparable; adding thermal balance and temperature measurement separates all six reference faults.',
        scope='Constant-parameter symbolic reference; quantitative robustness under model-v1 parameter maps and ambient nuisance is a separate calculation.',models=results)
    (out/'run_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();root=args.project_root.resolve()
    run(root,args.output_dir or root/'results/week1/structural')
