#!/usr/bin/env python3
"""Export exact audited round-two summaries and three recorded mechanism cases."""
import argparse,csv,json
from pathlib import Path
from result_paths import result_file
def load(p):return json.loads(p.read_text())
def rows(p):return list(csv.DictReader(p.open()))
def tex(s):
    return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def table(caption,label,columns,header,data):
    return '\n'.join([r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule',*[' & '.join(map(str,r))+r' \\' for r in data],r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    r=rows(result_file(a.source, '答案来源分析', 'answer_provenance_summary.csv'));out=[]
    labels={'original':'Original','no_rep_constraints':'Neither','no_ngram_block':'No trigram','labeled_codes_no_constraints':'Codes, neither','labeled_names_no_constraints':'Names, neither'}
    sels={'racps':'RACPS','bm25':'BM25','e5_rank':'E5-rank'}
    out.append(table('Answer provenance at B256. Counts are question--answer pairs; every row has 900 questions. R and S denote the reader and semantic matrices. In/out refers to selected endpoints, not the causal origin of an answer. Full precision, recall, F1, filter effects, and cap hits accompany the CSV.','tab:provenance_full','llrrrrrr','Matrix / selector & Decoder & TP in & TP out & FP in & FP out & F1 & Filter F1',[[('R' if x['matrix']=='reader' else 'S')+' / '+sels[x['selector']],labels[x['condition']],*[x[k] for k in ['correct_in','correct_out','wrong_in','wrong_out']],f"{float(x['f1']):.4f}",f"{float(x['filtered_f1']):.4f}"] for x in r]))
    s=load(result_file(a.source, '机制核验', 'mechanism_summary.json'))
    out.append(table('Descriptive token-exposure groups on the reader-control sample. Exposure tests internal name trigrams with and without a leading space. Gain is no-trigram-block F1 minus original F1. Recovered counts newly predicted correct selected endpoints absent from the original output, summed as question--answer pairs; the same entity on different questions counts separately. Groups differ in hop composition and are not randomized.','tab:exposure','lrrrrrr','Group & $n$ & Hop 1 & Hop 2 & Hop 3 & F1 gain & Recovered',[[lab,s[k]['n'],*[s[k]['hop_counts'].get(str(h),0) for h in [1,2,3]],f"{s[k]['mean_f1_change']:.4f}",s[k]['recovered_endpoints']] for k,lab in [('exposed','Exposed'),('unexposed_eligible','Unflagged, eligible'),('no_correct_endpoint','No correct endpoint')]]))
    r=rows(result_file(a.source, '独立确认分析', 'condition_metrics.csv'))
    out.append(table('Frozen confirmation on 300 new questions, 100 per hop. All four cells use B256, output cap 96, and the same checkpoint. EP denotes endpoint F1; hits count outputs reaching the cap.','tab:confirmation_cells','llrrrrrrr','Selector & Decoder & P & R & F1 & EP & Filter F1 & Hits & $n$',[[x['selector'].replace('_',' '),labels[x['condition']],*[f"{float(x[k]):.4f}" for k in ['precision','recall','f1','endpoint_f1','filtered_f1']],x['cap_hits'],x['n']] for x in r]))
    e=load(result_file(a.source, '独立确认分析', 'paired_effects.json'))
    def ci(v):return '['+', '.join(f'{z*100:.2f}' for z in v)+']'
    out.append(table('Confirmation effects in F1 percentage points. Interaction is cost effect after removing both constraints minus original cost effect, the single primary estimand. Other effects are descriptive decompositions. Intervals use 20,000 paired or whole-template resamples.','tab:confirmation_effects','lrrr',r'Effect & Estimate & Hop-stratified 95\% CI & Template 95\% CI',[[lab,f"{e[k]['difference']*100:.2f}",ci(e[k]['ci95']),ci(e[k]['template_cluster']['ci95'])] for k,lab in [('cost_original','Original cost effect'),('cost_no_rep_constraints','Removed-constraint cost effect'),('interaction','Interaction'),('full_decoder_gain','RACPS decoder increment')]]))
    (a.output/'generated_confirmation_tables.tex').write_text('\n'.join(out))
    cases=load(result_file(a.source, '机制核验', 'mechanism_cases.json'));case_tex=[]
    for i,c in enumerate(cases,1):
        e=c['event'];name=f'mechanism_case_{i}.txt'
        text='Question ID: '+c['id']+'\nQuestion: '+c['question']+'\n\nSELECTED EVIDENCE\n'+c['evidence']+'\n\nGOLD ANSWERS\n'+' | '.join(c['gold_answers'])+'\n\nORIGINAL OUTPUT\n'+c['original_output']+'\n\nNO-TRIGRAM-BLOCK OUTPUT\n'+c['no_ngram_output']+'\n\nTOKEN EVENT\n'+json.dumps(e,ensure_ascii=True,indent=2)+'\n'
        (a.output/name).write_text(''.join(c if ord(c)<128 else '\\u%04x'%ord(c) for c in text))
        case_tex.append(r'\paragraph{Case '+str(i)+r'. '+tex(c['id'])+r'.} This case was chosen by question ID within hop, not by effect size. The two runs share the prefix before the flagged token. The full machine-readable trace retains later counterfactual events separately.'+'\n'+r'\lstinputlisting{'+name+'}\n')
    (a.output/'recorded_mechanism_cases.tex').write_text('\n'.join(case_tex))
    print('Exported four exact tables and three token-level cases.')
if __name__=='__main__':main()
