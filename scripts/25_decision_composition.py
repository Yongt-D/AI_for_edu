"""Decompose saved prediction sets into decisions, with independent metric checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'outputs/revision_v4'
figdir = ROOT / 'figures/revision_v4'
p = pd.read_parquet(ROOT / 'outputs/revision_v3/predictions.parquet')
primary = ((p.model == 'logistic') & (p.class_weight == 'balanced')) | ((p.model == 'catboost') & (p.class_weight == 'unweighted'))
p = p[primary & (p.scope == 'full') & (p.calibration == 'isotonic') & p.method.isin(['MCP', 'EQ', 'EIQ-batch'])]
m = pd.read_csv(ROOT / 'outputs/revision_v3/metrics.csv')
rows = []
for key, g in p.groupby(['setting', 'model', 'method', 'seed']):
    setting, model, method, seed = key
    assert not g.entity_id.duplicated().any()
    sets = [json.loads(s) for s in g.conformal_set]
    y = g.risk.to_numpy(int)
    auto = np.array([len(s) == 1 for s in sets])
    pred = np.array([s[0] if len(s) == 1 else -1 for s in sets])
    row = dict(zip(['setting', 'model', 'method', 'seed'], key))
    row.update(n=len(g), n_fail=int((y == 1).sum()),
               correct=float((auto & (pred == y)).mean()),
               wrong=float((auto & (pred != y)).mean()), deferred=float((~auto).mean()),
               detected=float((auto & (pred == 1))[y == 1].mean()),
               missed=float((auto & (pred == 0))[y == 1].mean()),
               fail_deferred=float((~auto)[y == 1].mean()))
    assert np.isclose(sum(row[k] for k in ['correct', 'wrong', 'deferred']), 1)
    assert np.isclose(sum(row[k] for k in ['detected', 'missed', 'fail_deferred']), 1)
    weight = 'balanced' if model == 'logistic' else 'unweighted'
    ref = m[(m.setting == setting) & (m.model == model) & (m.method == method) &
            (m.seed == seed) & (m.class_weight == weight) & (m.scope == 'full') & (m.calibration == 'isotonic')]
    assert len(ref) == 1
    r = ref.iloc[0]
    assert np.isclose(row['deferred'], 1-r.automatic_coverage)
    assert np.isclose(row['missed'], r.automatic_false_negative_fraction)
    assert np.isclose(row['wrong']/(1-row['deferred']), r.automatic_error_rate)
    rows.append(row)
data = pd.DataFrame(rows)
assert len(data) == 150
data.to_csv(out / 'decision_composition_seeds.csv', index=False)
metrics = ['correct', 'wrong', 'deferred', 'detected', 'missed', 'fail_deferred']
summary = data.groupby(['setting', 'model', 'method'])[metrics].agg(['mean', 'min', 'max'])
summary.columns = ['_'.join(c) for c in summary.columns]
summary.reset_index().to_csv(out / 'decision_composition_summary.csv', index=False)

plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 9,
                     'axes.labelsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(2, 2, figsize=(9.4, 9.8))
fig.subplots_adjust(left=.17, right=.98, top=.89, bottom=.08, wspace=.30, hspace=.48)
settings = ['iid', 'temporal_1', 'temporal_2', 'course_1', 'course_2']
labels = ['IID', 'Temporal-1', 'Temporal-2', 'Course-1', 'Course-2']
methods = ['MCP', 'EQ', 'EIQ-batch']
colors = ['#187F88', '#C65050', '#DCE4EB']
for col, model in enumerate(['logistic', 'catboost']):
    for row, keys in enumerate([['correct', 'wrong', 'deferred'], ['detected', 'missed', 'fail_deferred']]):
        ax = axes[row, col]
        for i, setting in enumerate(settings):
            for j, method in enumerate(methods):
                y = i*4+j
                v = summary.loc[(setting, model, method)]
                left = 0
                for k, color in zip(keys, colors):
                    width = v[k+'_mean']*100
                    ax.barh(y, width, left=left, height=.77, color=color, edgecolor='white', linewidth=.55)
                    if width >= 8:
                        ax.text(left+width/2, y, f'{width:.0f}', ha='center', va='center',
                                fontsize=9, color='#263747' if color == colors[2] else 'white')
                    left += width
        ax.set_yticks([i*4+j for i in range(5) for j in range(3)])
        ax.set_yticklabels([f'{labels[i]}  /  {method}' if j == 0 and col == 0 else method
                            for i in range(5) for j, method in enumerate(methods)], fontsize=9)
        ax.set_ylim(18.8, -.8)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Share of all enrolments (%)' if row == 0 else 'Share of actual failures (%)')
        ax.tick_params(axis='both', length=0, pad=4)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E9EDF1', linewidth=.6)
        for spine in ax.spines.values(): spine.set_visible(False)
        title = 'Logistic regression' if model == 'logistic' else 'CatBoost'
        ax.set_title(f'{"abcd"[row*2+col]}  {title}', loc='left', fontweight='bold', fontsize=11, pad=10)
fig.text(.17, .966, 'ALL ENROLMENTS', color='#384E62', fontweight='bold', fontsize=10)
fig.text(.17, .49, 'ACTUAL FAILURES', color='#384E62', fontweight='bold', fontsize=10)
for y, labels_ in [(.944, ['Correct singleton', 'Wrong singleton', 'Deferred']),
                    (.466, ['Failure singleton', 'Pass singleton (missed)', 'Deferred'])]:
    fig.legend(handles=[Patch(facecolor=c, label=l) for c,l in zip(colors, labels_)],
               loc='center left', bbox_to_anchor=(.16, y), ncol=3, frameon=False, fontsize=9,
               handlelength=1.3, columnspacing=2)
fig.text(.17, .025, 'Isotonic probabilities  |  90% nominal coverage  |  Mean of five fixed fits',
         color='#526575', fontsize=8)
fig.savefig(figdir / 'decision_composition.pdf', bbox_inches='tight')
fig.savefig(figdir / 'decision_composition.png', dpi=240, bbox_inches='tight')
plt.close(fig)
print('150 fitted decision decompositions checked against saved metrics; 30 mean bars per population.')
print(data.groupby(['setting','model','method'])[metrics].mean().loc['temporal_2'].round(4).to_string())
