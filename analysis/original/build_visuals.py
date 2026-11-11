#!/usr/bin/env python3
"""Seven bilingual vector figures, generated only from verified frozen results.
Run after analyze_revision.py. JSON figure data are exported for audit.
"""
from pathlib import Path
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parent; M=R/'manuscript'; F=M/'figures'
A=json.loads((R/'derived/revision_analysis.json').read_text())
C=json.loads((R/'results/stats_clean_component_ablation_qwen_B256.json').read_text())
T=json.loads((R/'results/template_cluster_stats_qwen_main.json').read_text())
RD=json.loads((R/'derived/reader_diagnostics.json').read_text())
components=[r for r in C['comparisons'] if r['scope']=='all-hops' and r['metric']=='raw_f1']
COL={'bfs':'#9B9B9B','bm25':'#424B54','ba_mmr':'#3F8A78','ra_mmr':'#C34F27','racps':'#C34F27','relation_density':'#8D7350','e5_rank':'#2878A5','e5_density':'#7E92AD'}
MARK={'bfs':'v','bm25':'s','ba_mmr':'^','ra_mmr':'o'}
NAME={'bfs':'BFS-prefix','bm25':'BM25','ba_mmr':'BA-MMR','ra_mmr':'RACPS','racps':'RACPS','relation_density':'Relation-density','e5_rank':'E5-rank','e5_density':'E5-density'}
ZH={'BFS-prefix':'BFS顺序','Relation-density':'关系密度排序','E5-rank':'E5排序','E5-density':'E5密度排序'}
D={'source':'derived/revision_analysis.json; derived/reader_diagnostics.json; results/stats_clean_component_ablation_qwen_B256.json; results/template_cluster_stats_qwen_main.json','primary':A['primary'],'primary_pairs':A['primary_pairs'],'endpoint_reference':A['endpoint_reference'],'supplement':A['supplement'],'supplement_primary_pairs':A['supplement_primary_pairs'],'components':components,'heterogeneity':[r for r in A['heterogeneity'] if r['hop']==2 and r['feature']=='candidate_paths'],'depth':[]}
for b in [128,256,512]:
 records=json.loads((R/f'results/qwen3_awq_test300_B{b}.json').read_text())['results']
 for hop in [1,2,3]:
  for c in MARK:D['depth'].append({'budget':b,'hop':hop,'condition':c,'f1':float(np.mean([r['f1'] for r in records if r['condition']==c and r['hop']==hop]))})
D['reader_diagnostics']=RD
D['template_clusters']=[r for r in T['comparisons'] if r['scope']=='all-hops' and r['baseline']=='bm25']
D['supplement_strata']=[r for r in A['supplement_strata'] if r['group']=='extension1077']
(R/'derived/figure_data_20260910.json').write_text(json.dumps(D,indent=2,ensure_ascii=False)+'\n')

from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
QA=R/'figure_qa';QA.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'axes.edgecolor':'#A3ADB4','legend.frameon':False,'grid.color':'#E4E8EA','grid.linewidth':.5,'axes.axisbelow':True,'lines.linewidth':1.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
INK='#233744';MUTED='#667985';ORANGE=COL['racps'];BLUE=COL['e5_rank'];TEAL=COL['ba_mmr']
def save(fig,name,lang):
 fig.canvas.draw()
 fig.savefig(F/f'{name}_{lang}.pdf')
 fig.savefig(F/f'{name}_{lang}.svg')
 fig.savefig(F/f'{name}_{lang}.png',dpi=320)
 plt.close(fig)
def clean(ax,axis='y'):
 ax.grid(axis=axis) if axis=='y' else ax.grid(False);ax.tick_params(length=2.5,width=.6)
def rowaxis(ax,labels,ys=None):
 ys=np.arange(len(labels)) if ys is None else ys
 ax.set_yticks(ys,labels);ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
def point_ci(ax,x,lo,hi,y,color,marker='o',open_marker=False):
 assert lo<=x<=hi
 ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=marker,color=color,mfc='white' if open_marker else color,ms=4.7,capsize=2.4,lw=1.25,zorder=3)
def title(ax,letter,en,zh):
 ax.set_title(letter+'  '+h(en,zh),loc='left',pad=11,fontweight='bold',color=INK)
def legend_methods(fig,cs,y=.04):
 fig.legend(handles=[Line2D([],[],color=COL[c],marker=MARK.get(c,'o'),ms=4,lw=1.4,label=name(c)) for c in cs],loc='lower center',bbox_to_anchor=(.5,y),ncol=len(cs),fontsize=7.5,columnspacing=1.5,handlelength=1.5)
for lang in os.environ.get('KG_FIGURE_LANGUAGES', 'en').split(','):
 if lang=='zh':
  p=os.environ.get('KG_CJK_FONT','/System/Library/Fonts/Supplemental/Songti.ttc')
  font_manager.fontManager.addfont(p);plt.rcParams['font.family']=[font_manager.FontProperties(fname=p).get_name(),'DejaVu Sans']
 else:plt.rcParams['font.family']='DejaVu Sans'
 def h(en,zh):return zh if lang=='zh' else en
 def name(c):return ZH.get(NAME[c],NAME[c]) if lang=='zh' else NAME[c]
 # Original vector architecture: graph -> path packing -> two answer stages.
 fig,ax=plt.subplots(figsize=(7.2,4.15));fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.975)
 ax.set_xlim(0,104);ax.set_ylim(0,100);ax.axis('off')
 def text(x,y,en,zh=None,**kw):
  return ax.text(x,y,h(en,zh or en),color=INK,fontsize=7.8,va='center',**kw)
 def box(x,y,w,ht,fc='#F2F5F6',ec='none',radius=1.4):
  ax.add_patch(FancyBboxPatch((x,y),w,ht,boxstyle=f'round,pad=0,rounding_size={radius}',facecolor=fc,edgecolor=ec,lw=.8))
 def arrow(a,b,color=MUTED,**kw):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.9,color=color,**kw))
 text(1,97,'a  Evidence selection','a  证据选择',fontweight='bold')
 text(68,97,'b  Answer production','b  答案生成',fontweight='bold')
 box(0,78,24,12,'#EDF3F6');text(12,85,'Question + topic entity','问题 + 主题实体',ha='center',fontweight='bold');text(12,80.7,'Protocol-specific candidate pool','各协议内固定候选池',ha='center',fontsize=7.2) if False else None
 # graph geometry is illustrative, not experimental measurements
 nodes={0:(6,62),1:(16,70),2:(17,58),3:(9,46),4:(26,70),5:(27,57),6:(20,43)}
 for i,j,c in [(0,1,BLUE),(0,2,ORANGE),(0,3,'#BFC9CF'),(1,4,BLUE),(2,5,ORANGE),(2,6,TEAL),(3,6,'#BFC9CF')]:arrow(nodes[i],nodes[j],c,shrinkA=6,shrinkB=6)
 for i,(x,y) in nodes.items():
  ax.add_patch(Circle((x,y),1.75,facecolor=INK if i==0 else 'white',edgecolor=INK if i==0 else MUTED,lw=.9))
  ax.text(x,y,str(i),ha='center',va='center',fontsize=7.2,color='white' if i==0 else INK)
 text(1,36,'Complete directed paths','完整有向路径',fontweight='bold')
 text(1,31,'No partial-path truncation','不截断路径内部结构')
 arrow((26,65),(32,65))
 box(33,39,31,49,'#F7F3EF')
 text(35,83,'RACPS: rank, then pack','RACPS：排序与装填',fontweight='bold')
 for y,c,seq in [(73,BLUE,'0  →  1  →  4'),(65,ORANGE,'0  →  2  →  5'),(57,TEAL,'0  →  2  →  6')]:
  ax.plot([36,39],[y,y],color=c,lw=3,solid_capstyle='round');text(41,y,seq)
 text(35,47,'Relevance · novelty · cost','相关性 · 新颖性 · 成本')
 text(35,42,'Component + label interventions','组件干预 + 标签干预')
 box(33,20,31,14,'#EAF2EF',ec=TEAL);text(48.5,29,'Exact evidence-token gate','精确证据 token 检查',ha='center',fontweight='bold');text(48.5,23.5,'Tokens(concatenated paths) ≤ B','拼接后路径的 token 数 ≤ B',ha='center')
 arrow((48.5,39),(48.5,34),TEAL)
 arrow((64,27),(68,27),TEAL);ax.plot([68,68],[27,78],color=TEAL,lw=.9)
 text(68,88,'Same selected evidence','同一份所选证据',fontweight='bold')
 box(72,66,28,15,'#EDF3F6');arrow((68,74),(72,74),TEAL)
 text(86,76,'Distinct endpoints','去重的终点实体',ha='center',fontweight='bold');text(86,69.5,'0 reader calls','0 次阅读器调用',ha='center')
 box(72,43,28,15,'#F7EEE8');arrow((68,51),(72,51),TEAL)
 text(86,53,'Language-model reader','语言模型阅读器',ha='center',fontweight='bold');text(86,46.5,'1 reader call','1 次阅读器调用',ha='center')
 ax.plot([100,102,102],[73.5,73.5,33],color=MUTED,lw=.9);arrow((102,33),(99,33));arrow((86,43),(86,37))
 text(86,33,'Paired answer-set F1','配对答案集合 F1',ha='center',fontweight='bold');text(86,26,'Reader − endpoints','阅读器 − 端点',ha='center')
 ax.plot([0,100],[15,15],color='#C8D1D6',lw=.7)
 labels=[('Budget response','预算响应','MetaQA · 128 / 256 / 512'),('Component effects','组件效应','MetaQA · B = 256'),('Shared proposals','共享候选池','WebQSP · n = 1,077'),('Answer-stage utility','答案阶段效用','Same-evidence comparison')]
 for i,(a,b,c) in enumerate(labels):
  x=i*25+1;text(x,10,a,b,fontweight='bold');text(x,4,c,('同证据比较' if i==3 else c))
 save(fig,'study_framework',lang)
 # Quality and measured computation are joint coordinates, with paired inference alongside.
 fig,axs=plt.subplots(1,2,figsize=(7.2,3.05),gridspec_kw={'width_ratios':[1.25,1]})
 for c in MARK:
  rs=[next(r for r in A['primary'] if r['condition']==c and r['budget']==b) for b in [128,256,512]]
  axs[0].plot([r['retrieval_seconds'] for r in rs],[r['f1'] for r in rs],color=COL[c],lw=1.5)
  for j,r in enumerate(rs):axs[0].scatter(r['retrieval_seconds'],r['f1'],s=[18,32,50][j],marker=MARK[c],color=COL[c],edgecolor='white',linewidth=.4,zorder=3)
  r=rs[-1];off={'bfs':(0,12),'bm25':(8,-16),'ba_mmr':(-4,-13),'ra_mmr':(-4,6)}[c]
  axs[0].annotate(name(c),(r['retrieval_seconds'],r['f1']),xytext=off,textcoords='offset points',ha='right' if c in ['ra_mmr','ba_mmr','bfs'] else 'left',fontsize=7.5,color=COL[c])
 axs[0].set_xlim(.15,2.42);axs[0].set_ylim(.24,.389);axs[0].set_ylabel(h('Macro answer-set F1','宏平均答案集合 F1'));axs[0].set_xlabel(h('Measured retrieval time (s / question)','实测检索时间（秒／问题）'));clean(axs[0],'x');title(axs[0],'a','Quality–computation trajectory','质量与计算成本轨迹')
 axs[0].legend(handles=[Line2D([],[],marker='o',ls='',color=MUTED,ms=s,label=str(b)) for s,b in [(3.5,128),(4.8,256),(6.2,512)]],title=h('Token cap','Token 上限'),loc='lower right',ncol=3,fontsize=7, title_fontsize=7,handletextpad=.1,columnspacing=.6)
 for i,b in enumerate([128,256,512]):
  r=next(r for r in A['primary_pairs'] if r['budget']==b and r['baseline']=='bm25');t=next(r for r in D['template_clusters'] if r['budget']==b)
  point_ci(axs[1],100*r['mean_difference'],100*r['bootstrap_95_ci_low'],100*r['bootstrap_95_ci_high'],i-.12,ORANGE)
  point_ci(axs[1],100*t['question_weighted_difference'],100*t['cluster_bootstrap_95_ci_low'],100*t['cluster_bootstrap_95_ci_high'],i+.12,MUTED,'D',True)
  axs[1].text(9.5,i,f"{100*r['mean_difference']:+.2f}",ha='right',va='center',fontsize=8,color=ORANGE,fontweight='bold')
 axs[1].axvline(0,color=MUTED,lw=.8);axs[1].set_xlim(-.5,9.9);axs[1].set_xticks([0,2,4,6]);rowaxis(axs[1],['128','256','512']);axs[1].set_ylim(2.5,-.65);axs[1].set_xlabel(h('RACPS − BM25 (F1 points)','RACPS − BM25（F1百分点）'));clean(axs[1],'x');title(axs[1],'b','Paired quality gain','配对质量增益')
 fig.legend(handles=[Line2D([],[],marker='o',color=ORANGE,ls='',ms=4,label=h('Question-paired 95% CI','逐题配对95%区间')),Line2D([],[],marker='D',mfc='white',color=MUTED,ls='',ms=4,label=h('Template-cluster 95% CI','模板簇95%区间'))],loc='lower right',bbox_to_anchor=(.98,.00),ncol=2,fontsize=7.2,handletextpad=.3,columnspacing=.8)
 fig.subplots_adjust(left=.085,right=.98,bottom=.24,top=.84,wspace=.38);save(fig,'budget_tradeoffs',lang)
 # Intervention matrix supplies experimental semantics alongside measured effects.
 fig,axs=plt.subplots(1,2,figsize=(7.2,3.55),gridspec_kw={'width_ratios':[1.25,1.05]});ys=np.array([0,1,2,3.6,4.6])
 cells=[['Full','On','On','Expanded'],['Relation','Off','On','Expanded'],['Relation','On','Off','Expanded'],['Relation','On','On','Raw-S'],['Relation','On','On','Raw-E2E']]
 zhcell={'Full':'完整路径','Relation':'关系','On':'开','Off':'关','Expanded':'扩展','Raw-S':'选择器原始','Raw-E2E':'全流程原始'}
 for i,y in enumerate(ys):
  for j,v in enumerate(cells[i]):
   changed=(i==0 and j==0) or (i==1 and j==1) or (i==2 and j==2) or (i>2 and j==3)
   axs[0].add_patch(Rectangle((j-.48,y-.35),.96,.70,facecolor='#F4DED0' if changed else '#F0F3F4',edgecolor='white',lw=1))
   axs[0].text(j,y,zhcell[v] if lang=='zh' else v,ha='center',va='center',fontsize=7.2,color=INK,fontweight='bold' if changed else 'normal')
 for ax in axs:ax.set_ylim(5.35,-1.15)
 axs[0].set_xlim(-.5,3.5);rowaxis(axs[0],['1','2','3','4','5'],ys);axs[0].set_xticks(range(4),[h('Relevance','相关性'),h('Cost','成本'),h('Novelty','新颖性'),h('Labels','标签')]);axs[0].xaxis.tick_top();axs[0].tick_params(axis='x',length=0,pad=5);axs[0].spines['bottom'].set_visible(False);title(axs[0],'a','Changed configuration','变更的实验配置')
 for i,(r,y) in enumerate(zip(components,ys)):
  x,lo,hi=[r[k]*100 for k in ['mean_difference','bootstrap_95_ci_low','bootstrap_95_ci_high']];col=ORANGE if i<3 else MUTED
  point_ci(axs[1],x,lo,hi,y,col);axs[1].text(7.2,y,f'{x:+.2f}',ha='right',va='center',color=col,fontsize=8,fontweight='bold')
 axs[1].axvline(0,color=MUTED,lw=.8);axs[1].set_xlim(-1.6,7.5);axs[1].set_xticks([0,2,4,6]);rowaxis(axs[1],['1','2','3','4','5'],ys);clean(axs[1],'x');axs[1].set_xlabel(h('Full RACPS − variant (F1 points)','完整RACPS − 变体（F1百分点）'));title(axs[1],'b','Paired component effect','配对组件效应')
 for ax in axs:
  ax.title.set_position((0,1.20))
  ax.axhline(2.8,color='#BFCAD0',lw=.7,ls=(0,(3,3)))
 fig.text(.07,.04,h('Highlighted cells identify the intervention; dashed line separates the two exploratory test families.','高亮单元格表示干预；虚线区分两组探索性检验族。'),fontsize=7.4,color=MUTED)
 fig.subplots_adjust(left=.06,right=.98,bottom=.22,top=.77,wspace=.24);save(fig,'components',lang)
 # Depth profiles use a common zero-based scale; filled difference is descriptive.
 fig,axs=plt.subplots(1,3,figsize=(7.2,2.9),sharey=True)
 for hop,ax in zip([1,2,3],axs):
  vv={c:[next(r['f1'] for r in D['depth'] if r['budget']==b and r['hop']==hop and r['condition']==c) for b in [128,256,512]] for c in MARK}
  ax.fill_between(range(3),vv['bm25'],vv['ra_mmr'],color=ORANGE,alpha=.10,zorder=0)
  for c,vals in vv.items():ax.plot(range(3),vals,marker=MARK[c],ms=4,color=COL[c],label=name(c))
  title(ax,chr(96+hop),f'{hop} hop'+('s' if hop>1 else '')+' · n = 300',f'{hop}跳 · n = 300');ax.set_xticks(range(3),['128','256','512']);ax.set_xlabel(h('Evidence-token cap','证据 token 上限'));ax.set_ylim(0,.6);ax.set_xlim(-.2,2.2);clean(ax)
 axs[0].set_ylabel('F1');legend_methods(fig,list(MARK),.005);fig.subplots_adjust(left=.08,right=.985,bottom=.29,top=.83,wspace=.17);save(fig,'budget_depth',lang)
 # Shared pools: comparable absolute scores and a separate inferential panel.
 fig=plt.figure(figsize=(7.2,4.2));gs=fig.add_gridspec(2,3,height_ratios=[1,1.0]);axs=[fig.add_subplot(gs[0,i]) for i in range(3)];axc=fig.add_subplot(gs[1,:])
 cs=['e5_rank','relation_density','racps','e5_density','bm25']
 for j,ax in enumerate(axs):
  vals=[next(r['f1'] for r in A['supplement'] if r['group']=='extension1077' and r['condition']==c) if j==0 else next(r['f1'] for r in D['supplement_strata'] if r['chain']==['1','2+'][j-1] and r['condition']==c) for c in cs]
  for i,(c,v) in enumerate(zip(cs,vals)):
   ax.plot([0,v],[i,i],color=COL[c],alpha=.28,lw=4,solid_capstyle='round');ax.plot(v,i,'o',color=COL[c],ms=4.5);ax.text(v+.015,i,f'{v:.4f}',fontsize=7.2,va='center',color=COL[c])
  rowaxis(ax,[name(c) for c in cs] if j==0 else ['']*5);ax.set_xlim(0,.49);ax.set_xticks([0,.2,.4]);ax.set_ylim(4.55,-.65);ax.set_xlabel('F1');clean(ax,'x')
  title(ax,chr(97+j),['All · n = 1,077','Max. chain = 1 · n = 671','Max. chain ≥ 2 · n = 406'][j],['总体 · n = 1,077','最大链长 1 · n = 671','最大链长 ≥2 · n = 406'][j])
 for y,c in enumerate(['bm25','relation_density','e5_rank']):
  r=next(r for r in A['supplement_primary_pairs'] if r['baseline']==c);x,lo,hi=[100*r[k] for k in ['difference','ci_low','ci_high']];point_ci(axc,x,lo,hi,y,COL[c]);axc.text(19.7,y,f'{x:+.2f}  [{lo:+.2f}, {hi:+.2f}]',ha='right',va='center',fontsize=7.7,color=COL[c])
 rowaxis(axc,[h('vs ','对')+name(c) for c in ['bm25','relation_density','e5_rank']]);axc.axvline(0,color=MUTED,lw=.8);axc.set_xlim(-14,20.4);axc.set_xticks([-10,-5,0,5,10]);axc.set_ylim(2.55,-.65);clean(axc,'x');axc.set_xlabel(h('RACPS difference (F1 points; 95% paired CI)','RACPS差值（F1百分点；95%配对区间）'));title(axc,'d','Overall paired contrasts · n = 1,077','总体配对对比 · n = 1,077')
 fig.subplots_adjust(left=.16,right=.98,bottom=.13,top=.88,hspace=.95,wspace=.16);save(fig,'shared_pool',lang)
 # Seed panels retain potentially overlapping samples; counts are printed beside each interval.
 fig,axs=plt.subplots(1,3,figsize=(7.2,2.9),sharey=True)
 for k,(seed,ax,col) in enumerate(zip([20260818,20260819,20260820],axs,[BLUE,TEAL,ORANGE])):
  for j,b in enumerate(['1-100','101-1000','>1000']):
   r=next(x for x in D['heterogeneity'] if x['seed']==seed and x['bin']==b)
   point_ci(ax,100*r['difference'],100*r['ci_low'],100*r['ci_high'],j,col);ax.text(31,j+.29,f"n={r['n']}",ha='right',va='center',fontsize=7.2,color=col)
  ax.axvline(0,color=MUTED,lw=.8);rowaxis(ax,['1–100','101–1,000','>1,000']);ax.set_ylim(2.55,-.65);ax.set_xlim(-9,33);ax.set_xticks([-5,0,10,20,30]);clean(ax,'x');title(ax,chr(97+k),f'Seed {seed}',f'种子 {seed}')
 axs[0].set_ylabel(h('Candidate paths · 2 hops','候选路径数 · 2跳'));fig.text(.54,.14,h('RACPS − BM25 (F1 points; 95% paired CI)','RACPS − BM25（F1百分点；95%配对区间）'),ha='center',fontsize=8)
 fig.text(.54,.025,h('Each seed is a separate sample; intervals are not pooled across seeds.','每个种子对应一份样本；不跨种子合并区间。'),ha='center',fontsize=7.4,color=MUTED)
 fig.subplots_adjust(left=.14,right=.985,bottom=.28,top=.82,wspace=.12);save(fig,'candidate_strata',lang)
 # Absolute scores + same-evidence increments form a two-row evidence dashboard.
 fig,axs=plt.subplots(2,2,figsize=(7.2,4.7),gridspec_kw={'height_ratios':[3,5],'width_ratios':[1,1]})
 for k in range(2):
  if k==0:
   rs=[next(r for r in A['endpoint_reference'] if r['condition']=='ra_mmr' and r['hop']=='all' and r['budget']==b) for b in [128,256,512]]
   labels=[f'B = {b}' for b in [128,256,512]];cols=[ORANGE]*3;effects=[(-r['difference'],-r['ci_high'],-r['ci_low']) for r in rs]
  else:
   cs=['e5_rank','relation_density','racps','e5_density','bm25'];rs=[next(r for r in RD['webqsp_reader_increments'] if r['condition']==c) for c in cs];labels=[name(c) for c in cs];cols=[COL[c] for c in cs];effects=[(r['reader_minus_endpoint'],r['ci_low'],r['ci_high']) for r in rs]
  left,right=axs[k]
  for y,(r,col,(v,lo,hi)) in enumerate(zip(rs,cols,effects)):
   left.plot([r['endpoint_f1'],r['reader_f1']],[y,y],color=col,lw=2)
   left.plot(r['endpoint_f1'],y,'o',mfc='white',mec=col,ms=5);left.plot(r['reader_f1'],y,'o',color=col,ms=5)
   point_ci(right,100*v,100*lo,100*hi,y,col);right.text(16.5,y,f'{100*v:+.2f}',ha='right',va='center',fontsize=8,color=col,fontweight='bold')
  for ax in [left,right]:ax.set_ylim(len(rs)-.45,-.65);clean(ax,'x')
  rowaxis(left,labels);rowaxis(right,['']*len(rs));left.set_xlim(0,.42);left.set_xticks([0,.1,.2,.3,.4]);right.set_xlim(-7,17.5);right.set_xticks([-5,0,5,10]);right.axvline(0,color=MUTED,lw=.8)
  title(left,chr(97+2*k),'MetaQA · RACPS · n = 900' if k==0 else 'WebQSP · B = 256 · n = 1,077','MetaQA · RACPS · n = 900' if k==0 else 'WebQSP · B = 256 · n = 1,077')
  title(right,chr(98+2*k),'Same-evidence increment','同证据下的阅读增益')
  left.set_xlabel(h('Macro answer-set F1','宏平均答案集合 F1'));right.set_xlabel(h('Reader − endpoints (F1 points)','阅读器 − 端点（F1百分点）'))
 fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=INK,color=INK,ls='',label=h('Endpoints · 0 calls','端点 · 0次调用')),Line2D([],[],marker='o',color=INK,ls='',label=h('Reader · 1 call','阅读器 · 1次调用')),Line2D([],[],color=MUTED,marker='|',label=h('Paired 95% CI','配对95%区间'))],loc='lower center',bbox_to_anchor=(.55,.005),ncol=3,fontsize=7.5)
 fig.subplots_adjust(left=.17,right=.985,bottom=.15,top=.91,wspace=.30,hspace=.85);save(fig,'reader_profile',lang)
print('Wrote scientific figures from frozen sources; languages: ' + os.environ.get('KG_FIGURE_LANGUAGES', 'en'))
