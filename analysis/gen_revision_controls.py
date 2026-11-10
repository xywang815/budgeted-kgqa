#!/usr/bin/env python3
"""Plot saved post-review analyses; numerical data are never embedded in the chart."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(source, output, zh=False):
    interface = json.loads((source / 'output_interface.json').read_text())
    comparisons = json.loads((source / 'reader_comparators.json').read_text())
    contexts = json.loads((source / 'context_and_generation.json').read_text())
    coverage = json.loads((source / 'candidate_coverage.json').read_text())
    metrics = list(csv.DictReader((source / 'complete_main_metrics.csv').open()))
    names = ['BM25', 'RACPS', 'Relation-density', 'E5-rank', 'E5-density']
    methods = ['bm25', 'racps', 'relation_density', 'e5_rank', 'e5_density']
    blue, orange, ink = '#0072B2', '#D55E00', '#27333B'
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Arial Unicode MS', 'Heiti SC', 'DejaVu Sans'] if zh else ['DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#9AA5AB', 'axes.linewidth': .6,
        'xtick.color': ink, 'ytick.color': ink, 'text.color': ink,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False})
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 5.45),
                            gridspec_kw={'width_ratios': [1, 1.22]})
    fig.subplots_adjust(left=.12, right=.985, top=.92, bottom=.105, hspace=.66, wspace=.7)
    a, b, c, d = axs.flat
    for ax in axs.flat:
        ax.set_axisbelow(True); ax.grid(axis='x', color='#E7EBEE', lw=.55)
        ax.tick_params(length=2.5, width=.6)
    a.set_title('a  输出过滤改变增益估计' if zh else 'a  Output-interface sensitivity', loc='left', y=1.2, pad=8, weight='bold')
    for i, row in enumerate(interface):
        for off, key, color, marker, label in [(-.13, 'raw_minus_endpoint', orange, 'o', '原始输出' if zh else 'Raw'),
                                              (.13, 'filtered_minus_endpoint', blue, 's', '端点过滤' if zh else 'Filtered')]:
            val = row[key]; x = val['difference'] * 100; lo, hi = np.array(val['ci95']) * 100
            a.errorbar(x, i + off, xerr=[[x-lo], [hi-x]], color=color, marker=marker,
                       markersize=4, capsize=2, lw=1.05, label=label if i == 0 else None)
    a.axvline(0, color='#88959C', lw=.8, ls='--'); a.set_yticks(range(3), ['128', '256', '512']); a.invert_yaxis()
    a.set_xlim(-6.8, 7.8); a.set_ylabel('证据预算 / tokens' if zh else 'Evidence budget / tokens')
    a.set_xlabel('阅读器 − 端点 F1 / 百分点' if zh else 'Reader − endpoints / F1 points')
    a.legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2)
    b.set_title('b  WebQSP：阅读器与投票筛选' if zh else 'b  Reader versus fixed voting', loc='left', y=1.2, pad=8, weight='bold')
    for i, method in enumerate(methods):
        for off, comparator, color, marker, label in [(-.13, 'max_vote', orange, 'o', '最高票数' if zh else 'Max-vote'),
                                                     (.13, 'vote_top5', blue, 's', '票数前五' if zh else 'Vote-top5')]:
            row = next(r for r in comparisons if r['dataset'] == 'WebQSP' and r['condition'] == method and r['contrast'] == 'reader_minus_' + comparator)
            x = row['difference'] * 100; lo, hi = np.array(row['ci95']) * 100
            b.errorbar(x, i + off, xerr=[[x-lo], [hi-x]], color=color, marker=marker,
                       markersize=3.6, capsize=2, lw=1.05, label=label if i == 0 else None)
    b.set_yticks(range(5), names); b.invert_yaxis(); b.set_xlim(0, 12)
    b.set_xlabel('阅读器 − 投票 F1 / 百分点' if zh else 'Reader − vote baseline / F1 points')
    b.legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2)
    c.set_title('c  已选路径的跳数构成' if zh else 'c  Selected path lengths', loc='left', pad=12, weight='bold')
    for i, method in enumerate(methods):
        row = next(r for r in contexts if r['dataset'] == 'WebQSP' and r['condition'] == method)
        x = row['selected_twohop_path_fraction'] * 100
        color = blue if method == 'e5_rank' else orange if method == 'e5_density' else '#8B9DA7'
        c.plot([0, x], [i, i], color=color, lw=2.4, alpha=.45)
        c.plot(x, i, 'o', color=color, markersize=4.8)
        c.text(x + 1.4, i, f'{x:.2f}%', va='center', color=color, fontsize=7)
    c.set_yticks(range(5), names); c.invert_yaxis(); c.set_xlim(0, 59)
    c.set_xlabel('两跳路径占已选路径的比例 / %' if zh else 'Two-hop share of selected paths / %')
    d.grid(False); d.grid(axis='y', color='#E7EBEE', lw=.55)
    d.set_title('d  候选覆盖逐阶段变化' if zh else 'd  Answer coverage', loc='left', pad=12, weight='bold')
    for method, color, marker in [('racps', orange, 'o'), ('e5_rank', blue, 's')]:
        vals = [coverage['enumerated_endpoint_recall'], coverage['proposal_endpoint_recall']]
        for stage in ['endpoint', 'reader']:
            vals.append(float(next(r['recall'] for r in metrics if r['dataset'] == 'WebQSP' and r['condition'] == method and r['stratum'] == 'all' and r['stage'] == stage)))
        d.plot(range(4), np.array(vals)*100, marker=marker, color=color, lw=1.4, markersize=4,
               label='RACPS' if method == 'racps' else 'E5-rank')
    d.set_xticks(range(4), ['枚举', '提案', '选中端点', '阅读器'] if zh else ['Enumerated', 'Proposals', 'Selected', 'Reader'])
    d.tick_params(axis='x', labelsize=7); d.set_ylim(0, 105)
    d.set_ylabel('宏平均答案召回率 / %' if zh else 'Macro answer recall / %')
    d.legend(frameon=False, fontsize=7, loc='lower left')
    output.mkdir(parents=True, exist_ok=True)
    stem = 'revision_controls_' + ('zh' if zh else 'en')
    fig.savefig(output / (stem + '.pdf'), bbox_inches='tight')
    fig.savefig(output / (stem + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--chinese', action='store_true')
    a = p.parse_args(); draw(a.analysis_dir, a.output, a.chinese)
