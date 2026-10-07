"""Scientific before/after figures from saved measurements, no model inference."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
ranking=json.loads((ROOT/'ranking/results.json').read_text())
qwen=json.loads((ROOT/'results/qwen/results.json').read_text())
clef=json.loads((ROOT/'clef_results/results.json').read_text())
lora=json.loads((ROOT/'results/qwen_qlora/results.json').read_text())
centered=json.loads((ROOT/'results/qwen/centered-results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
colors=['#c75a48','#888a8d','#269989','#4563ad']
fig,axs=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained')
methods=['original','uncentered_projection','activation_orbit_mean','score_orbit_mean']
labels=['Original','Projection','Activation mean','Score mean']
for ax,model in zip(axs,['Qwen1.5B','Clef9B']):
    values=[ranking['models'][model][m]['shortlist']['totals']['all_finite_family_assignments']['pools_with_any_selected_set_change'] for m in methods]
    bars=ax.bar(labels,values,color=colors,width=.65)
    ax.bar_label(bars,labels=[f'{v}/6' for v in values],padding=4)
    ax.set_ylim(0,6.7);ax.set_yticks(range(7));ax.set_ylabel('Pools with any top-1 candidate-ID change');ax.set_title(model)
    ax.tick_params(axis='x',labelrotation=20)
fig.suptitle('Fixed-pool shortlist sensitivity to names',fontweight='bold')
fig.supxlabel('Six convenience pools, 16 held-out CVs; complete 288-assignment census, not 288 independent samples',fontsize=10)
for ext in ['png','pdf','svg']: fig.savefig(OUT/f'fixed-pool-before-after.{ext}')
plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
qmethods=['raw','learned_full_difference_projection','fixed_orbit_activation_average']
qvals=[qwen['methods'][m]['shortlist']['thesis_normalized_presence'] for m in qmethods]
cvals=[clef['activation_metrics'][m]['last']['test']['normalized_mean_direction_presence'] for m in ['raw','projected','orbit_mean']]
for ax,vals,title in zip(axs[:2],[qvals,cvals],['Qwen: normalized final state','Clef: final-token summary']):
    bars=ax.bar(['Original','Projection','Activation mean'],vals,color=colors[:3])
    ax.bar_label(bars,labels=[f'{v:.4f}' for v in vals],padding=4,fontsize=10)
    ax.set_ylim(0,max(vals)*1.3);ax.set_ylabel('Presence = ||group mean difference|| / mean(||h||)');ax.set_title(title)
    ax.tick_params(axis='x',labelrotation=20)
vals=[lora['methods'][m]['shortlist']['thesis_normalized_presence'] for m in ['quantized_base','quantized_base_plus_lora']]
bars=axs[2].bar(['4-bit base','+ LoRA'],vals,color=['#c75a48','#888a8d']);axs[2].bar_label(bars,labels=[f'{v:.4f}' for v in vals],padding=4)
axs[2].set_ylim(0,max(vals)*1.3);axs[2].set_title('Lower Presence did not repair rankings');axs[2].set_ylabel('Presence')
fig.supxlabel('Averaged interface is invariant; original branch states persist. Clef summary does not describe its whole tensor.',fontsize=10)
for ext in ['png','pdf','svg']: fig.savefig(OUT/f'activation-before-after.{ext}')
plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained')
layer=qwen['layerwise_raw_shortlist_presence'];xs=list(map(int,layer.keys()));ys=list(layer.values())
axs[0].plot(xs,ys,marker='o',color='#4563ad',linewidth=2);axs[0].set_xticks(xs)
axs[0].set_title('Qwen original selected residual layers');axs[0].set_xlabel('Transformer block (1-indexed)');axs[0].set_ylabel('Normalized mean-direction Presence')
axs[0].text(.03,.94,'Last token, before final normalization\nConcentration does not prove causal localization',transform=axs[0].transAxes,va='top',fontsize=10)
ps=json.loads((ROOT/'results/qwen/scores.json').read_text());cases=json.loads((ROOT/'cases.json').read_text())['cases'];tests={c['case_id'] for c in cases if c['split']=='test'}
variants=[[r['raw_p_A'] for r in ps if r['task']=='shortlist' and r['case_id']==c['case_id']] for c in cases if c['case_id'] in tests]
for i,v in enumerate(variants): axs[1].plot([min(v),max(v)],[i,i],color='#c75a48',linewidth=3);axs[1].scatter(v,[i]*4,s=12,color='#c75a48')
axs[1].axvline(.5,color='#555555',linestyle='--',label='0.5 threshold');axs[1].set_yticks(range(16),[str(i+1) for i in range(16)])
axs[1].set_xlabel('Constrained A/B shortlist probability');axs[1].set_ylabel('Held-out applicant (opaque index)');axs[1].set_title('Same qualifications, four naming cues');axs[1].legend(loc='lower right')
for ext in ['png','pdf','svg']: fig.savefig(OUT/f'layers-and-matched-scores.{ext}')
plt.close(fig)
print(OUT)
