#!/usr/bin/env python3
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--results',type=Path,default=HERE.parent/'results/component_stages.json')
ap.add_argument('--output-dir',type=Path,default=HERE)
args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
data=json.loads(args.results.read_text())
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
    'legend.frameon':False,'savefig.dpi':300})
fig,(ax,rec)=plt.subplots(1,2,figsize=(7.0,2.5),gridspec_kw={'width_ratios':[1.5,1]},layout='constrained')
y=np.array([2,1,0]); labels=['Relation representation','Cost normalization','Novelty']
for metric,color,mark,offset,label in [('reader_F1','#bc5b30','o',.13,'Reader'),('endpoint_F1','#167a89','D',-.13,'Endpoints')]:
    for i,r in enumerate(data):
        v=r[metric]; m=v['full_minus_variant']*100;lo,hi=np.array(v['ci95'])*100
        ax.errorbar(m,y[i]+offset,xerr=[[m-lo],[hi-m]],fmt=mark,color=color,
                    markersize=4.6,capsize=2,elinewidth=1.1,label=label if i==0 else None)
for i,r in enumerate(data):
    v=r['endpoint_recall'];m=v['full_minus_variant']*100;lo,hi=np.array(v['ci95'])*100
    rec.errorbar(m,y[i],xerr=[[m-lo],[hi-m]],fmt='D',color='#167a89',markersize=4.6,capsize=2,elinewidth=1.1)
for a in [ax,rec]:
    a.axvline(0,color='#727b80',lw=.8,ls='--',zorder=0)
    a.set_ylim(-.5,2.65);a.tick_params(axis='y',length=0)
    a.grid(axis='x',color='#e4e7e8',lw=.5);a.set_axisbelow(True)
ax.set_yticks(y,labels);rec.set_yticks(y,[])
ax.set_xlim(-1.5,7.5);rec.set_xlim(-1.5,10.5)
ax.set_xlabel('Full RACPS − variant (F1 percentage points)')
rec.set_xlabel('Endpoint-recall difference (points)')
ax.set_title('a  Gains measured at two answer stages',loc='left',fontweight='bold',pad=9)
rec.set_title('b  Gold-answer coverage',loc='left',fontweight='bold',pad=9)
ax.legend(loc='upper left',ncol=2,fontsize=8,handletextpad=.5,columnspacing=1)
fig.savefig(args.output_dir/'component_stages.pdf')
fig.savefig(args.output_dir/'component_stages.png')
plt.close(fig)
