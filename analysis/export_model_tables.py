#!/usr/bin/env python3
"""Export exact completed-control means and intervals as supplementary LaTeX."""
import argparse,csv,json
from pathlib import Path


def load(path):return json.loads(path.read_text())
def rows(path):return list(csv.DictReader(path.open()))
def esc(text):return str(text).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
LABELS={'racps':'RACPS','bm25':'BM25','e5_rank':'E5-rank','e5_density':'E5-density','e5_development_choice':'Development choice',
        'rank':'Raw rank','max_density_g05':'Max, exponent 0.5','max_density_g1':'Max, exponent 1',
        'minmax_density_g05':'Min--max, exponent 0.5','minmax_density_g1':'Min--max, exponent 1',
        'web_dev':'WebQSP training','web_test':'WebQSP test','meta_test':'MetaQA test'}
def human(value):
    if '_minus_' in value:return ' minus '.join(human(x) for x in value.split('_minus_'))
    return LABELS.get(value,value)
def number(v):return f'{float(v):.4f}'
def effect(row):
    lo,hi=row['ci95'];return f"{100*row['difference']:.2f} [{100*lo:.2f}, {100*hi:.2f}]"
def table(caption,label,columns,body,layout=None):
    layout=layout or ('l'+'r'*(len(columns)-1))
    reserve = r'\Needspace{'+str(9+2*len(body))+r'\baselineskip}' if len(body)<=10 else ''
    return '\n'.join([reserve,r'\begin{center}\small',r'\begin{longtable}{'+layout+'}',
        r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',
        ' & '.join(columns)+r' \\',r'\midrule\endfirsthead',r'\toprule',
        ' & '.join(columns)+r' \\',r'\midrule\endhead',
        *[' & '.join(map(str,r))+r' \\' for r in body],r'\bottomrule',r'\end{longtable}',r'\end{center}'])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--reader-analysis',type=Path,required=True)
    ap.add_argument('--semantic-analysis',type=Path,required=True);ap.add_argument('--cost-analysis',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    reader=rows(a.reader_analysis/'condition_metrics.csv');semantic=rows(a.semantic_analysis/'condition_metrics.csv')
    contrasts=load(a.reader_analysis/'contrasts.json');direct=load(a.reader_analysis/'direct_interactions.json')
    names={'original':'Original','no_cost_original':'No cost, original order','labeled_names':'Labeled names','labeled_codes':'Labeled codes',
        'canonical':'Canonical order','no_cost_canonical':'No cost, canonical order','cap192':'Cap 192','cap64':'Cap 64',
        'no_rep_penalty':'No repetition penalty','no_ngram_block':'No trigram block','no_rep_constraints':'Neither constraint',
        'labeled_names_no_constraints':'Names, neither constraint','labeled_codes_no_constraints':'Codes, neither constraint'}
    text=[];body=[]
    for r in reader:
        label=f"{r['dataset']} B{r['budget']} {human(r['selector'])} / {names[r['condition']]}"
        body.append([esc(label),r['n'],number(r['f1']),number(r['filtered_f1']),number(r['endpoint_f1']),r['cap_hits'],r['context_over_cap']])
    text.append(table('Matched reader conditions. F1 is a fraction. Filtered output intersects predictions with selected endpoints. Cap hits count generation limits; over-cap counts refer to evidence lengths after annotation or reordering. All questions are retained.','tab:model_conditions',
        ['Condition','$n$','Reader','Filtered','Endpoint','Hits','Over cap'],body,layout='p{.43\\linewidth}rrrrrr'))
    selected_names=[('codes_minus_names_labeled','Codes minus names, original constraints'),
        ('codes_minus_names_no_constraints','Codes minus names, neither constraint'),('no_rep_penalty_minus_original','Remove repetition penalty'),
        ('no_ngram_block_minus_original','Remove trigram block'),('no_rep_constraints_minus_original','Remove both constraints'),
        ('cost_effect_original','Cost effect, original order'),('cost_effect_canonical','Cost effect, canonical order')]
    body=[]
    for key,label in selected_names:
        r=next(r for r in contrasts if r['dataset']=='MetaQA' and r['budget']==256 and r['contrast']==key)
        body.append([label,r['n'],effect(r),effect(r['template_cluster'])])
    for key,label in [('order_modulation_original_minus_canonical','Order modulation: original minus canonical'),('interface_effect_original_constraints_minus_removed','Interface effect: original constraints minus removed')]:
        r=direct[key];body.append([label,r['n'],effect(r),effect(r['template_cluster'])])
    text.append(table('Direct reader contrasts on MetaQA B256. Differences and pointwise 95\\% intervals are in F1 percentage points. Cluster intervals resample question templates; they are exploratory sensitivity checks.','tab:model_contrasts',
        ['Contrast','$n$','Stratified 95\\% CI','Template 95\\% CI'],body,layout='p{.40\\linewidth}rll'))
    body=[]
    for r in contrasts:
        if r['contrast'] not in ('cap192_minus_original','cap64_minus_original'):continue
        body.append([esc(f"{r['dataset']} B{r['budget']} {human(r['selector'])}"),r['n'],effect(r),effect(r['template_cluster']) if 'template_cluster' in r else '--'])
    text.append(table('Generation-cap interventions. MetaQA increases from 96 to 192 tokens; WebQSP from 32 to 64. Other settings and selected paths remain fixed. Estimates and intervals are in F1 percentage points.','tab:cap_interventions',
        ['Setting','$n$','Estimate [stratified CI]','Estimate [template CI]'],body,layout='p{.34\\linewidth}rll'))
    body=[]
    for r in semantic:
        if r['stratum']!='all':continue
        label=f"{r['dataset']} {human(r['selector'])} / {names[r['condition']]}"
        body.append([esc(label),r['n'],number(r['precision']),number(r['recall']),number(r['f1']),number(r['endpoint_f1'])])
    text.append(table('Semantic selection and matched lexical replays at B256. Original and constraint-removal conditions use the same selected paths. All values are macro means.','tab:semantic_controls',
        ['Condition','$n$','Precision','Recall','F1','Endpoint'],body,layout='p{.46\\linewidth}rrrrr'))
    body=[]
    for r in load(a.semantic_analysis/'paired_contrasts.json'):
        if r['stage']!='metrics':continue
        label=f"{r['dataset']} {human(r['contrast'])} / {names[r['condition']]}"
        body.append([esc(label),effect(r),effect(r['template_cluster']) if 'template_cluster' in r else '--'])
    text.append(table('Matched semantic-versus-lexical and development-rule contrasts. All intervals are exploratory and pointwise, in F1 percentage points.','tab:semantic_contrasts',
        ['Contrast','Stratified 95\\% CI','Template 95\\% CI'],body,layout='p{.50\\linewidth}ll'))
    choice=load(a.semantic_analysis/'development_choice.json')
    body=[[esc(human(rule)),number(score),'Selected' if rule==choice['rule'] else ''] for rule,score in choice['development_endpoint_macro_f1'].items()]
    text.append(table('WebQSP training-sample selection. Each rule uses the same 100 official training questions and B256 cap. Selection uses endpoint macro F1, before test scoring.','tab:semantic_development',
        ['Rule','Development F1','Decision'],body,layout='lrl'))
    body=[]
    for r in rows(a.semantic_analysis/'selection_summary.csv'):
        if r['phase']!='web_test':continue
        body.append([esc(human(r['rule'])),f"{float(r['mean_selected_paths']):.2f}",f"{float(r['mean_evidence_tokens']):.2f}",
            f"{float(r['selected_two_hop_share'])*100:.2f}",f"{float(r['two_hop_candidate_retention'])*100:.4f}"])
    text.append(table('WebQSP test selection diagnostics. Two-hop share divides selected two-hop paths by all selected paths; retention divides them by all two-hop proposal paths, pooling questions in both cases.','tab:semantic_retention',
        ['Rule','Paths','Tokens','Two-hop share (\\%)','Retention (\\%)'],body,layout='lrrrr'))
    body=[]
    for r in load(a.semantic_analysis/'selector_decoder_interactions.json'):
        body.append([esc(human(r['contrast'])),effect(r),effect(r['template_cluster'])])
    text.append(table('Direct selector-by-decoding interactions on MetaQA B256: selector contrast without repetition constraints minus the same contrast under original decoding. Intervals are exploratory and pointwise, in F1 percentage points.','tab:selector_decoder',
        ['Selector contrast','Stratified 95\\% CI','Template 95\\% CI'],body,layout='p{.43\\linewidth}ll'))
    d=load(a.semantic_analysis/'candidate_distributions.json');body=[]
    for phase,v in d['phases'].items():
        c=d['pool_counts'][phase]
        body.append([esc(human(phase)),f"{c['nonempty']}/{c['all']}",*[number(v[k]['mean']) for k in ('min_cosine','median_cosine','max_cosine','dynamic_range')],f"{v['median_cost']['mean']:.2f}"])
    text.append(table('Candidate score and cost distributions. Each value is the mean of the indicated within-question statistic, over nonempty pools; empty pools remain in evaluation. Full distributions and score arrays accompany the archive.','tab:candidate_scores',
        ['Pool','Nonempty/all','Min. cosine','Med. cosine','Max. cosine','Range','Med. cost'],body,layout='lrrrrrr'))
    body=[]
    reader_replay=load(a.reader_analysis/'historical_concordance.json')
    for r in reader_replay:
        label='Reader / '+r['dataset']+' B'+str(r['budget'])+' '+human(r['selector'])+' / '+names[r['condition']]
        body.append([esc(label),r['n'],r['raw_match'],r['f1_match'],f"{100*r['f1_difference']:.4f}"])
    for r in load(a.semantic_analysis/'historical_concordance.json')['reader']:
        body.append([esc('Semantic / '+r['dataset']+' '+human(r['selector'])),r['n'],r['raw_match'],r['f1_match'],f"{100*r['f1_delta']:.4f}"])
    if a.cost_analysis:
        for r in load(a.cost_analysis/'historical_concordance.json'):
            body.append([esc('Cost/decoder / '+names[r['condition']]),r['n'],r['raw_match'],r['f1_match'],f"{100*r['f1_delta']:.4f}"])
    text.append(table('Historical replay concordance. Raw and F1 columns count exact matches; the last column gives new minus historical mean F1 in percentage points. New contrasts use their own matched baselines.','tab:replay_concordance',
        ['Replay','$n$','Raw','F1','Mean change'],body,layout='p{.48\\linewidth}rrrr'))
    if a.cost_analysis:
        cost_names={'original':'Full, original decoding','no_cost_original':'No cost, original decoding',
            'no_rep_constraints':'Full, neither constraint','no_cost_no_constraints':'No cost, neither constraint'}
        body=[[cost_names[r['condition']],r['n'],number(r['precision']),number(r['recall']),number(r['f1']),number(r['endpoint_f1'])]
            for r in rows(a.cost_analysis/'condition_metrics.csv')]
        text.append(table('Second-wave cost-by-decoding matrix at MetaQA B256. All four cells are regenerated with matched batch settings and the same selected paths for each selector. This control was specified after the initial trigram-blocking finding, before its own outcomes.','tab:cost_decoder_means',
            ['Condition','$n$','Precision','Recall','F1','Endpoint'],body,layout='p{.43\\linewidth}rrrrr'))
        labels={'cost_original':'Cost effect, original decoding','cost_no_constraints':'Cost effect, neither constraint',
            'decoder_modulation_relaxed_minus_original':'Cost effect: neither minus original','cost_endpoint':'Endpoint cost effect',
            'stage_effect_original':'Reader minus endpoint effect, original','stage_effect_no_constraints':'Reader minus endpoint effect, neither'}
        body=[[labels[key],effect(r),effect(r['template_cluster'])] for key,r in load(a.cost_analysis/'paired_effects.json').items()]
        text.append(table('Direct cost-by-decoding and stage-effect contrasts. Values are F1 percentage points with exploratory pointwise 95\\% intervals.','tab:cost_decoder_effects',
            ['Contrast','Stratified 95\\% CI','Template 95\\% CI'],body,layout='p{.43\\linewidth}ll'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n\n'.join(text)+'\n')


if __name__=='__main__':main()
