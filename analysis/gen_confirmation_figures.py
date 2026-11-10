#!/usr/bin/env python3
"""Render answer composition and confirmation diagnostics from audited summaries."""
import argparse
from pathlib import Path
import numpy as np
from gen_model_controls import load,readcsv,style,clean,point,save,BLUE,ORANGE,GREEN,GRAY
import matplotlib.pyplot as plt

def provenance(root,out,zh):
    rows=readcsv(root/'答案来源分析/answer_provenance_summary.csv')
    rows=[r for r in rows if r['matrix']=='semantic']
    rows=sorted(rows,key=lambda r:(['bm25','racps','e5_rank'].index(r['selector']),r['condition']!='original'))
    fig,axs=plt.subplots(2,2,figsize=(8.0,6.0));fig.subplots_adjust(left=.14,right=.98,bottom=.11,top=.94,wspace=.55,hspace=.65)
    a,b,c,d=axs.flat
    names=[s+' / '+t for s in ['BM25','RACPS','E5-rank'] for t in (['原','移除'] if zh else ['orig.','removed'])]
    for ax,keys,title in [(a,['correct_in','correct_out'],'a  正确预测的端点归属' if zh else 'a  Correct predictions'),(b,['wrong_in','wrong_out'],'b  错误预测的端点归属' if zh else 'b  Incorrect predictions')]:
        left=np.zeros(6)
        for key,col,lab in zip(keys,[BLUE,ORANGE],['端点内','端点外'] if zh else ['Within endpoints','Outside endpoints']):
            vals=np.array([int(r[key]) for r in rows]);ax.barh(range(6),vals,left=left,color=col,height=.65,label=lab)
            for i,(v,l) in enumerate(zip(vals,left)):
                if v>300 or key.endswith('in'):ax.text(l+v/2,i,str(v),color='white',ha='center',va='center',fontsize=6)
                elif key.endswith('out'):ax.text(l+v+15,i,str(v),ha='left',va='center',color=col,fontsize=7)
            left+=vals
        ax.set_yticks(range(6),names);ax.invert_yaxis();ax.set_xlim(0,max(left)*1.13);clean(ax)
        ax.set_title(title,loc='left',weight='bold',pad=28)
        ax.legend(fontsize=7,ncol=2,loc='lower left',bbox_to_anchor=(0,1.0));ax.set_xlabel('题目—答案对数' if zh else 'Question–answer pairs')
        for y in [1.5,3.5]:ax.axhline(y,color='#CBD3D8',lw=.5)
    vals=np.array([[float(r[k])*100 for k in ['precision','recall','f1','filtered_f1']] for r in rows])
    c.imshow(vals,aspect='auto',cmap='Blues',vmin=25,vmax=75)
    c.set_xticks(range(4),['P','R','F1','过滤F1' if zh else 'Filtered F1']);c.set_yticks(range(6),names)
    c.tick_params(length=0);c.set_title('c  问题宏平均 / %' if zh else 'c  Question-macro scores / %',loc='left',weight='bold',pad=10)
    for i in range(6):
        for j in range(4):c.text(j,i,f'{vals[i,j]:.1f}',ha='center',va='center',color='white' if vals[i,j]>55 else '#25313A',fontsize=8)
    effects=load(root/'答案来源分析/answer_filter_effects.json')
    for i,r in enumerate(rows):
        e=next(e for e in effects if all(e[k]==r[k] for k in ['matrix','selector','condition']))
        point(d,e,i,GRAY if r['condition']=='original' else GREEN)
    d.set_yticks(range(6),names);d.set_ylim(5.5,-.5);d.axvline(0,color=GRAY,ls=':',lw=1);clean(d)
    d.set_title('d  端点过滤的收益' if zh else 'd  Endpoint-filtering effect',loc='left',weight='bold',pad=10)
    d.set_xlabel('过滤 − 原输出 / F1 百分点' if zh else 'Filtered − raw / F1 points')
    save(fig,out,'answer_provenance',zh)

def confirmation(root,out,zh):
    rows=readcsv(root/'独立确认分析/condition_metrics.csv');effects=load(root/'独立确认分析/paired_effects.json')
    summary=load(root/'机制核验/mechanism_summary.json');case=load(root/'机制核验/mechanism_cases.json')[0];ev=case['event']
    fig=plt.figure(figsize=(8,5.7));gs=fig.add_gridspec(2,2,left=.12,right=.98,bottom=.1,top=.95,wspace=.5,hspace=.6)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
    for sel,col,lab in [('full',BLUE,'RACPS'),('no_cost',ORANGE,'无成本项' if zh else 'No cost')]:
        vals=[float(next(r for r in rows if r['selector']==sel and r['condition']==dec)['f1'])*100 for dec in ['original','no_rep_constraints']]
        a.plot([0,1],vals,'o-',color=col,label=lab,lw=1.5,ms=5)
        for x,y in enumerate(vals):a.annotate(f'{y:.2f}',(x,y),xytext=(-6 if x==0 else 6,(7 if sel=='full' else -12) if x==0 else (-12 if sel=='full' else 7)),textcoords='offset points',ha='right' if x==0 else 'left',color=col,fontsize=8)
    a.set_ylim(30,63);a.set_xlim(-.3,1.3);a.set_xticks([0,1],['原约束','移除两项'] if zh else ['Original','Both removed']);a.set_ylabel('答案集 F1 / %' if zh else 'Answer-set F1 / %');a.legend(loc='upper left',fontsize=7);clean(a,'y')
    a.set_title('a  新增 300 题：四组对照' if zh else 'a  Four cells on 300 new questions',loc='left',weight='bold',pad=12)
    for i,k in enumerate(['cost_original','cost_no_rep_constraints','interaction']):
        point(b,effects[k],i,GREEN if i==2 else BLUE,marker='D' if i==2 else 'o')
    b.set_yticks(range(3),['原约束成本效应','移除后成本效应','直接交互'] if zh else ['Cost, original rules','Cost, rules removed','Interaction']);b.set_ylim(2.5,-.5);b.axvline(0,color=GRAY,ls=':',lw=1);clean(b)
    b.set_xlabel('F1 效应 / 百分点' if zh else 'F1 effect / percentage points');b.set_title('b  成本项 × 解码' if zh else 'b  Cost × decoding',loc='left',weight='bold',pad=12)
    c.axis('off');c.set_title('c  真实 token 轨迹：原输出与解除阻断输出在此分叉' if zh else 'c  A recorded token-level divergence',loc='left',weight='bold',pad=12)
    c.text(0,.91,case['id']+'   |   '+case['question'],fontsize=9,transform=c.transAxes)
    tokens=[repr(t.replace('Ġ',' ')) for t in ev['trigram_tokens']]
    c.text(0,.64,('提示中的三元 n-gram' if zh else 'Trigram already in prompt')+'  →',fontsize=8.5,transform=c.transAxes)
    for i,t in enumerate(tokens):
        c.text(.40+i*.12,.64,t,ha='center',fontsize=11,fontfamily='monospace',color=ORANGE if i==2 else BLUE,bbox=dict(boxstyle='round,pad=.4',fc='#F8E8E0' if i==2 else '#E6F1F7',ec='none'),transform=c.transAxes)
    c.text(.79,.64,'−∞ '+('禁止' if zh else 'blocked'),color=ORANGE,fontsize=10,transform=c.transAxes)
    c.text(0,.34,('共同生成前缀：' if zh else 'Shared generated prefix: ')+ev['decoded_prefix'],fontsize=9,transform=c.transAxes)
    c.text(0,.08,('原约束下一 token：' if zh else 'Original next token: ')+repr(ev['original_next_token']),color=ORANGE,fontsize=9,transform=c.transAxes)
    c.text(.48,.08,('移除阻断下一 token：' if zh else 'Block removed: ')+repr(ev['token_text']),color=BLUE,fontsize=9,transform=c.transAxes)
    save(fig,out,'mechanism_confirmation',zh)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--chinese',action='store_true');a=p.parse_args();style(a.chinese);provenance(a.source,a.output,a.chinese);confirmation(a.source,a.output,a.chinese)
if __name__=='__main__':main()
