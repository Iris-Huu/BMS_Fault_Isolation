"""Structural detectability, MSOs and fault signatures.

Isolability matrices use 1=NOT isolable and 0=isolable (diagonal is 1).
Fault signature matrices instead use 1=sensitive, 0=insensitive.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from faultdiagnosistoolbox.dmperm import GetDMParts

MATRIX_CONVENTION='rows/columns follow model.f; 1=not isolable, 0=isolable; diagonal=1'


def save_matrix(matrix,row_names,col_names,filepath):
    with Path(filepath).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream,lineterminator="\n");writer.writerow(['']+list(col_names))
        for name,row in zip(row_names,np.asarray(matrix)):writer.writerow([name]+list(row))


def draw_isolability(matrix,fault_names,title,ax):
    matrix=np.asarray(matrix,dtype=int)
    ax.imshow(matrix,cmap='Greys',vmin=0,vmax=1,interpolation='nearest')
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j] else 'black')
    ax.set_xticks(range(len(fault_names)),fault_names,rotation=45)
    ax.set_yticks(range(len(fault_names)),fault_names)
    ax.set_title(title);ax.set_xlabel('Competing fault');ax.set_ylabel('True fault')
    ax.text(.5,-.24,'1 = not isolable; 0 = isolable',transform=ax.transAxes,ha='center',fontsize=9)


def save_isolability_plot(model,causality,fault_names,title,filepath):
    # Do not let plotting permute the matrix relative to saved fault labels.
    matrix=np.asarray(model.IsolabilityAnalysis(causality=causality,permute=False),dtype=int)
    fig,ax=plt.subplots(figsize=(6,5));draw_isolability(matrix,fault_names,title,ax)
    fig.tight_layout();fig.savefig(filepath,dpi=180,bbox_inches='tight');plt.close(fig)
    return matrix


def save_dm_plot(model,filepath,title):
    """Use toolbox DM partition, with explicit integer indices for empty blocks.

    Avoid the toolbox 0.12.5 fault annotation path that fails when M0 is empty.
    This is the same DM decomposition, not a replacement graph algorithm.
    """
    dm=GetDMParts(model.X)
    rows=np.asarray(dm.rowp,dtype=int);cols=np.asarray(dm.colp,dtype=int)
    reordered=model.X[np.ix_(rows,cols)]
    fig,ax=plt.subplots(figsize=(8,6));ax.spy(reordered!=0,markersize=6,color='tab:blue')
    ax.set_xticks(range(len(cols)),[model.x[k] for k in cols],rotation=90)
    fault_equations=np.any(model.F!=0,axis=1)
    ax.set_yticks(range(len(rows)),[f'e{k}'+(' *' if fault_equations[k] else '') for k in rows])
    components=[('M-',dm.Mm)]+[('M0',part) for part in dm.M0]+[('M+',dm.Mp)]
    r0=c0=0
    for label,part in components:
        nr,nc=len(part.row),len(part.col)
        if nr and nc:
            ax.add_patch(Rectangle((c0-.5,r0-.5),nc,nr,fill=False,edgecolor='tab:red',linewidth=1.2))
            ax.text(c0+nc-.6,r0+.1,label,ha='right',color='tab:red',fontsize=9)
        r0+=nr;c0+=nc
    ax.set_title(title+'; * denotes a fault equation')
    fig.tight_layout();fig.savefig(filepath,dpi=180,bbox_inches='tight');plt.close(fig)
    return dict(row_permutation=rows.tolist(),column_permutation=cols.tolist(),
                underdetermined_equations=len(dm.Mm.row),just_determined_equations=len(dm.M0eqs),
                overdetermined_equations=len(dm.Mp.row))


def select_tests_with_detection(model,low_index_msos):
    """Retain toolbox isolability selection and add missing detection coverage.

    With one nonthermal MSO and no isolable fault pairs, TestSelection returns
    an empty set. That set is sufficient for isolation equivalence but loses
    fault detection, so it cannot be used as the diagnostic test set.
    """
    full=np.asarray(model.FSM(low_index_msos),dtype=int)
    chosen=[int(k) for k in np.asarray(model.TestSelection(low_index_msos)).ravel()]
    requested=np.any(full!=0,axis=0)
    covered=np.any(full[chosen]!=0,axis=0) if chosen else np.zeros(len(model.f),dtype=bool)
    while np.any(requested&~covered):
        candidates=[k for k in range(len(full)) if k not in chosen]
        k=max(candidates,key=lambda k:int(np.sum((full[k]!=0)&requested&~covered)))
        chosen.append(k);covered|=full[k]!=0
    return chosen


def run_structural_analysis(model,model_label,output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    fault_names=list(model.f)
    model.Lint()
    occurrence=np.sum(model.F!=0,axis=0)
    if not np.all(occurrence==1):raise AssertionError('Each fault must occur in exactly one equation')
    detectable,not_detectable=model.DetectabilityAnalysis(causality='mixed')
    (output_dir/'detectability.txt').write_text('Detectable faults:\n'+str(detectable)+'\n\nNon-detectable faults:\n'+str(not_detectable),encoding='utf-8')
    matrices={}
    for causality,label in [('int','integral'),('mixed','mixed')]:
        matrix=save_isolability_plot(model,causality,fault_names,f'{model_label}: {label}',output_dir/f'isolability_{label}.png')
        save_matrix(matrix,fault_names,fault_names,output_dir/f'isolability_{label}.csv');matrices[label]=matrix
    dm=save_dm_plot(model,output_dir/'DM_decomposition.png',model_label+' DM decomposition')
    msos=[np.asarray(s,dtype=int).tolist() for s in model.MSO()]
    low=[s for s in msos if model.IsLowIndex(s)]
    (output_dir/'MSO_list.txt').write_text(f'Total MSO sets: {len(msos)}\n\n'+'\n'.join(f'MSO {k+1}: {s}' for k,s in enumerate(msos))+'\n',encoding='utf-8')
    (output_dir/'low_index_MSO.txt').write_text(f'Total MSO sets: {len(msos)}\nLow-index MSO sets: {len(low)}\n\n'+'\n'.join(f'Low-index MSO {k+1}: {s}' for k,s in enumerate(low))+'\n',encoding='utf-8')
    details=[]
    for k,s in enumerate(msos):
        unknowns=np.flatnonzero(np.any(model.X[s,:]!=0,axis=0))
        details.append(dict(mso=k+1,equation_indices_zero_based=s,equation_count=len(s),
            unknowns=[model.x[j] for j in unknowns],unknown_count=len(unknowns),redundancy=len(s)-len(unknowns),
            equations=[str(model.syme[j]) for j in s],fault_signature=np.asarray(model.FSM([s]),dtype=int)[0].tolist(),
            low_index=bool(model.IsLowIndex(s))))
    (output_dir/'MSO_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    all_fsm=np.asarray(model.FSM(msos),dtype=int)
    save_matrix(all_fsm,[f'MSO_{k+1}' for k in range(len(msos))],fault_names,output_dir/'FSM_all_labeled.csv')
    selected_indices=select_tests_with_detection(model,low) if low else []
    selected=[low[k] for k in selected_indices]
    fsm=np.asarray(model.FSM(selected),dtype=int) if selected else np.empty((0,len(fault_names)),dtype=int)
    np.savetxt(output_dir/'FSM.csv',fsm,delimiter=',',fmt='%d')
    save_matrix(fsm,[f'Test_{k+1}' for k in range(len(selected))],fault_names,output_dir/'FSM_labeled.csv')
    selected_isolation=np.asarray(model.IsolabilityAnalysisFSM(fsm,permute=False),dtype=int)
    save_matrix(selected_isolation,fault_names,fault_names,output_dir/'selected_test_isolability.csv')
    (output_dir/'selected_tests.txt').write_text(f'Number of selected tests: {len(selected)}\n\n'+'\n'.join(f'Test {k+1}: low-index MSO {j+1}, equations {low[j]}' for k,j in enumerate(selected_indices))+'\n',encoding='utf-8')
    if not np.array_equal(selected_isolation,matrices['mixed']):raise AssertionError('Selected tests do not preserve mixed-causality isolability')
    if len(fsm) and not np.all(np.any(fsm!=0,axis=0)):
        raise AssertionError('Selected tests lose detectable faults')
    with (output_dir/'equation_inventory.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream,lineterminator="\n");writer.writerow(['index_zero_based','equation','unknowns','faults'])
        for k in range(len(model.X)):
            writer.writerow([k,str(model.syme[k]),';'.join(model.x[j] for j in np.flatnonzero(model.X[k])),
                             ';'.join(model.f[j] for j in np.flatnonzero(model.F[k]))])
    summary=dict(model=model_label,fault_names=fault_names,equation_count=len(model.X),unknown_count=len(model.x),
        detectable=list(detectable),not_detectable=list(not_detectable),mso_count=len(msos),low_index_mso_count=len(low),
        selected_test_count=len(selected),selected_low_index_indices_zero_based=selected_indices,
        isolability_convention=MATRIX_CONVENTION,fsm_convention='1=sensitive, 0=insensitive',
        integral_matrix=matrices['integral'].tolist(),mixed_matrix=matrices['mixed'].tolist(),dm=dm)
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'{model_label}: {len(msos)} MSOs, {len(low)} low-index MSOs, {len(selected)} selected tests')
    return summary
