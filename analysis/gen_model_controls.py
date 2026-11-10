#!/usr/bin/env python3
"""Plot completed model-control analyses without embedding experimental values."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';GRAY='#80919B';INK='#27333B'


def load(path):return json.loads(path.read_text())
def readcsv(path):return list(csv.DictReader(path.open()))
def style(zh):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial Unicode MS','Heiti SC','DejaVu Sans'] if zh else ['DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#9AA5AB','axes.linewidth':.6,'xtick.color':INK,'ytick.color':INK,'text.color':INK,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False,'legend.frameon':False})
def clean(ax,axis='x'):
    ax.set_axisbelow(True);ax.grid(axis=axis,color='#E7EBEE',lw=.55);ax.tick_params(length=2.5,width=.6)
def point(ax,row,y,color=BLUE,marker='o'):
    x=row['difference']*100;lo,hi=np.asarray(row['ci95'])*100
    ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],marker=marker,color=color,ms=4,capsize=2,lw=1.1)
def title(ax,text):ax.set_title(text,loc='left',weight='bold',y=1.16,pad=10)
def save(fig,out,stem,zh):
    out.mkdir(parents=True,exist_ok=True);suffix='_zh' if zh else '_en'
    fig.savefig(out/(stem+suffix+'.pdf'),bbox_inches='tight');fig.savefig(out/(stem+suffix+'.png'),dpi=300,bbox_inches='tight');plt.close(fig)


def reader_figure(source,out,zh,cost_source=None):
    rows=readcsv(source/'condition_metrics.csv');contrasts=load(source/'contrasts.json');direct=load(source/'direct_interactions.json')
    def metric(condition):return next(r for r in rows if r['dataset']=='MetaQA' and r['budget']=='256' and r['condition']==condition)
    def contrast(name,dataset='MetaQA',budget=256,selector='racps'):
        return next(r for r in contrasts if r['contrast']==name and r['dataset']==dataset and r['budget']==budget and r['selector']==selector)
    fig,axes=plt.subplots(2,2,figsize=(7.5,6.1));fig.subplots_adjust(left=.14,right=.98,bottom=.10,top=.93,hspace=.64,wspace=.72)
    a,b,c,d=axes.flat
    for ax in axes.flat:clean(ax)
    title(a,'a  同一证据：输出接口与约束' if zh else 'a  Interface × decoding constraints')
    a.grid(False);a.grid(axis='y',color='#E7EBEE',lw=.55)
    for mode,color,marker,label in [('names',ORANGE,'o','实体名' if zh else 'Entity names'),('codes',BLUE,'s','候选代码' if zh else 'Answer codes')]:
        vals=[float(metric('labeled_'+mode+suffix)['f1'])*100 for suffix in ('','_no_constraints')]
        a.plot([0,1],vals,color=color,marker=marker,lw=1.5,ms=4,label=label)
        for x,v in enumerate(vals):a.annotate(f'{v:.1f}',(x,v),xytext=(-7 if x==0 else 7,7),textcoords='offset points',ha='right' if x==0 else 'left',fontsize=7,color=color)
    ep=float(metric('labeled_names')['endpoint_f1'])*100
    a.axhline(ep,color=GRAY,ls=':',lw=1.1,label='直接端点' if zh else 'Endpoints')
    a.set_xlim(-.3,1.3);a.set_xticks([0,1],['原约束','移除两项'] if zh else ['Original','Both removed'])
    a.set_ylabel('答案集 F1 / %' if zh else 'Answer-set F1 / %');a.set_xlabel('重复惩罚与三元 n-gram 限制' if zh else 'Repetition penalty and trigram block')
    a.legend(fontsize=7,loc='best')
    title(b,'b  自由输出：分开移除约束' if zh else 'b  Free strings: separate constraints')
    names=['no_rep_penalty','no_ngram_block','no_rep_constraints']
    for i,n in enumerate(names):point(b,contrast(n+'_minus_original'),i,BLUE if i<2 else ORANGE)
    b.set_yticks(range(3),['无重复惩罚','无三元 n-gram 限制','同时移除'] if zh else ['No repeat penalty','No trigram block','Neither constraint']);b.invert_yaxis()
    b.axvline(0,color=GRAY,ls='--',lw=.8);b.set_xlabel('相对原配置的 F1 变化 / 百分点' if zh else 'F1 change from original / points')
    title(c,'c  成本项：顺序与解码' if zh else 'c  Cost effect: order and decoding')
    for i,(n,col) in enumerate([('cost_effect_original',ORANGE),('cost_effect_canonical',BLUE)]):point(c,contrast(n),i,col)
    labels=['原顺序','统一顺序'] if zh else ['Original order','Canonical order']
    if cost_source:
        effects=load(cost_source/'paired_effects.json')
        for i,(name,color,marker) in enumerate([('cost_original',GRAY,'o'),('cost_no_constraints',BLUE,'o'),('decoder_modulation_relaxed_minus_original',GREEN,'D')],2):
            point(c,effects[name],i,color,marker)
        labels+=['原解码（匹配重跑）','移除两项约束','解码交互'] if zh else ['Original decoder','Neither constraint','Decoder interaction']
        c.axhline(1.5,color='#CBD3D8',ls=':',lw=.7)
    else:
        point(c,direct['order_modulation_original_minus_canonical'],2,GREEN,'D')
        labels+=['效应差：原−统一'] if zh else ['Original − canonical']
    c.set_yticks(range(len(labels)),labels);c.set_ylim(len(labels)-.5,-.5)
    c.axvline(0,color=GRAY,ls='--',lw=.8);c.set_xlabel('F1 效应 / 百分点' if zh else 'F1 effect / percentage points')
    title(d,'d  放宽输出上限' if zh else 'd  Larger generation caps')
    specs=[('MetaQA',128,'racps','cap192'),('MetaQA',256,'racps','cap192'),('MetaQA',512,'racps','cap192'),('WebQSP',256,'racps','cap64'),('WebQSP',256,'e5_rank','cap64')]
    for i,(ds,budget,sel,condition) in enumerate(specs):point(d,contrast(condition+'_minus_original',ds,budget,sel),i,ORANGE if ds=='MetaQA' else BLUE)
    d.set_yticks(range(5),['MetaQA B128','MetaQA B256','MetaQA B512','WebQSP RACPS','WebQSP E5-rank']);d.invert_yaxis()
    d.axvline(0,color=GRAY,ls='--',lw=.8);d.set_xlabel('上限翻倍后的 F1 变化 / 百分点' if zh else 'F1 change after doubling cap / points')
    save(fig,out,'reader_interventions',zh)


def semantic_figure(source,out,zh):
    metrics=readcsv(source/'condition_metrics.csv');selection=readcsv(source/'selection_summary.csv');choice=load(source/'development_choice.json')
    fig,axes=plt.subplots(2,2,figsize=(7.5,5.65));fig.subplots_adjust(left=.15,right=.985,bottom=.10,top=.93,hspace=.6,wspace=.75)
    a,b,c,d=axes.flat
    for ax in axes.flat:clean(ax)
    title(a,'a  训练题上的规则选择' if zh else 'a  Rule selection on training questions')
    keys=list(choice['development_endpoint_macro_f1']);names=['Rank','Max / $c^{0.5}$','Max / $c$','Min–max / $c^{0.5}$','Min–max / $c$']
    for i,key in enumerate(keys):
        value=100*choice['development_endpoint_macro_f1'][key];color=GREEN if key==choice['rule'] else GRAY
        a.plot([0,value],[i,i],color=color,alpha=.4,lw=3);a.plot(value,i,'D' if key==choice['rule'] else 'o',color=color,ms=5)
        a.annotate(f'{value:.1f}',(value,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=7,color=color)
    a.set_yticks(range(5),names);a.invert_yaxis();a.set_xlim(0,max(choice['development_endpoint_macro_f1'].values())*100+8)
    a.set_xlabel('开发样本端点 F1 / %' if zh else 'Development endpoint F1 / %')
    for ax,ds,selectors,names,heading in [
        (b,'MetaQA',['bm25','racps','e5_rank'],['BM25','RACPS','E5-rank'],'b  MetaQA：共同完整候选池' if zh else 'b  MetaQA: complete shared pools'),
        (c,'WebQSP',['e5_rank','e5_density','e5_development_choice'],['E5-rank','E5-density','Dev. choice'],'c  WebQSP：冻结后测试' if zh else 'c  WebQSP: frozen-rule test')]:
        title(ax,heading)
        if ds=='WebQSP' and choice['rule'] in ('rank','max_density_g1'):
            selectors=selectors[:2];names=names[:2]
            chosen_index=0 if choice['rule']=='rank' else 1
            names[chosen_index]+='\n（开发集选中）' if zh else '\n(dev. choice)'
        for i,sel in enumerate(selectors):
            r=next(r for r in metrics if r['dataset']==ds and r['selector']==sel and r['stratum']=='all' and r['condition']=='original');v=[float(r[k])*100 for k in ('endpoint_f1','f1')]
            colors=[GRAY,BLUE];markers=['o','s'];labels=['端点' if zh else 'Endpoints','原约束' if zh else 'Original']
            if ds=='MetaQA':
                r2=next(r for r in metrics if r['dataset']==ds and r['selector']==sel and r['stratum']=='all' and r['condition']=='no_rep_constraints')
                v.append(float(r2['f1'])*100);colors.append(GREEN);markers.append('^');labels.append('无两项约束' if zh else 'Repetition rules removed')
            ax.plot([min(v),max(v)],[i,i],color='#C5CDD2',lw=2.4)
            for j,(val,color,marker,label) in enumerate(zip(v,colors,markers,labels)):
                ax.plot(val,i,marker,color=color,ms=5,label=label if i==0 else None)
                ax.annotate(f'{val:.1f}',(val,i),xytext=(0,-12 if j==0 else 8),textcoords='offset points',ha='center',fontsize=7,color=color)
        ax.set_yticks(range(len(selectors)),names);ax.set_ylim(len(selectors)-.35,-.65);ax.set_xlabel('答案集 F1 / %' if zh else 'Answer-set F1 / %');ax.legend(fontsize=6.5,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2)
    title(d,'d  选择器与解码的交互' if zh else 'd  Selector × decoding interaction')
    interactions=load(source/'selector_decoder_interactions.json')
    contrasts=['e5_rank_minus_racps','racps_minus_bm25','e5_rank_minus_bm25']
    for i,name in enumerate(contrasts):
        row=next(r for r in interactions if r['contrast']==name)
        point(d,row,i,GREEN if i==0 else BLUE)
    d.set_yticks(range(3),['E5-rank − RACPS','RACPS − BM25','E5-rank − BM25']);d.set_ylim(2.5,-.5)
    d.axvline(0,color=GRAY,ls='--',lw=.8)
    d.set_xlabel('移除约束后的对照效应变化 / 百分点' if zh else 'Change in selector contrast / F1 points')
    save(fig,out,'semantic_controls',zh)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--reader-analysis',type=Path);ap.add_argument('--semantic-analysis',type=Path)
    ap.add_argument('--cost-analysis',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--chinese',action='store_true');a=ap.parse_args();style(a.chinese)
    if not a.reader_analysis and not a.semantic_analysis:ap.error('Provide at least one completed analysis.')
    if a.reader_analysis:reader_figure(a.reader_analysis,a.output,a.chinese,a.cost_analysis)
    if a.semantic_analysis:semantic_figure(a.semantic_analysis,a.output,a.chinese)


if __name__=='__main__':main()
