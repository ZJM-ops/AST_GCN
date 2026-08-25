#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 SSI (Seizure Severity Index) 的药物疗效检验 (训练集 Ictal 版)
================================================================
论文公式 (4): SSI(S) = P_final(S),  ∀ S ∈ S_ictal
  - SSI = 模型输出 P(Ictal) (soft voting 概率, 连续 0~1)
  - 仅对真实 Ictal 段计算
  - SSI 高(~1.0): 严重发作; SSI 低: 药物抑制发作有效

输入:
  drug_efficacy_train_only/per_segment_predictions.csv
  (仅训练集, 24037 段, 已带 prob_Ictal 列)

输出 (drug_efficacy_ssi_train_only/):
  1. ssi_per_segment.csv       每段 true_phase==Ictal: group + SSI + animal_id + seg_id
  2. ssi_group_summary.csv     各组 n / mean / median / std / p5 / p25 / p75 / p95 SSI
  3. ssi_stats_vs_pilo.csv     各药物 vs pilo 的 Mann-Whitney + Cliff's delta (效应量)
  4. ssi_boxplot.png           4 组 SSI 箱线图
  5. ssi_kde.png               4 组 SSI 分布核密度图
  6. ssi_efficacy_bars.png     各组 mean SSI 柱状 + 显著差异标记
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
import re

DRUG_GROUPS = ['pilo组', 'VPA组', 'mir过表达组', 'sponges组']
GROUP_EN = {'pilo组': 'Pilo', 'VPA组': 'VPA', 'mir过表达组': 'miR-OE', 'sponges组': 'Sponges'}

PER_SEG_CSV = '/home/student/s230005071/shu/code2/summary_results/10mineeg/results/20260804_184146_ASTNet_Standard_3Class/AST-Net_GCN/drug_efficacy_train_only/per_segment_predictions.csv'
OUT_DIR = '/home/student/s230005071/shu/code2/summary_results/10mineeg/results/20260804_184146_ASTNet_Standard_3Class/AST-Net_GCN/drug_efficacy_ssi_train_only'

ALPHA = 0.05


def cliffs_delta(x, y):
    """Cliff's delta: 非参数效应量, 范围 [-1, 1].
    >0 => y 更大; <0 => x 更大; |d|<0.147 小, <0.33 中, ≥0.33 大.
    """
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    mat = np.subtract.outer(y, x)          # y_i - x_j
    gt = (mat > 0).sum()
    lt = (mat < 0).sum()
    return (gt - lt) / (nx * ny)


def p_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(PER_SEG_CSV)
    print(f'读取 {PER_SEG_CSV}: 总 {len(df)} 段')

    df = df[df['true_phase'] == 'Ictal'].copy()
    print(f'筛选 true_phase=Ictal: 剩余 {len(df)} 段 (训练集)')
    df['SSI'] = df['prob_Ictal']

    # 从 seg_id 解析动物 id (格式 "Ictal_VPA_12_VPA组_Ictal_seg0000" / "Ictal_mir过表达_24_mir过表达组_Ictal_seg0000")
    # 结构: <true_phase>_<animal_prefix>_<animal_num>_<组名>_<时相>_segNNNN
    pat = r'^(?:Baseline|Ictal|Chronic)(?:-28d)?_' +           r'(?P<animal_prefix>[^_]+(?:_[^_]+)?)_(?P<animal_num>\d+)_' +           r'(?P<grp>[^_]+(?:_[^_]+组)?)_(?P<phase>[^_]+)_seg(?P<seg>\d+)$'
    m = df['seg_id'].astype(str).str.extract(pat)
    df['animal_id'] = m['animal_prefix'] + '_' + m['animal_num']

    # ============ 1. 保存每段 SSI ============
    cols = ['seg_id', 'group', 'animal_id', 'true_phase',
            'true_label', 'pred_label', 'pred_class', 'prob_Baseline', 'prob_Ictal',
            'prob_Chronic', 'SSI']
    out_cols = [c for c in cols if c in df.columns]
    df[out_cols].to_csv(f'{OUT_DIR}/ssi_per_segment.csv', index=False)
    print(f'1. 保存 SSI 逐段: {OUT_DIR}/ssi_per_segment.csv ({len(df)} 行)')

    # ============ 2. 组级 SSI 统计 ============
    stats_rows = []
    for g in DRUG_GROUPS:
        arr = df.loc[df['group'] == g, 'SSI'].values
        if len(arr) == 0: continue
        stats_rows.append({
            'drug_group': g,
            'drug_group_en': GROUP_EN.get(g, g),
            'n_segments': len(arr),
            'n_animals': df.loc[df['group'] == g, 'animal_id'].nunique(),
            'ssi_mean':  float(np.mean(arr)),
            'ssi_median':float(np.median(arr)),
            'ssi_std':   float(np.std(arr, ddof=1)),
            'ssi_p5':    float(np.percentile(arr, 5)),
            'ssi_p25':   float(np.percentile(arr, 25)),
            'ssi_p75':   float(np.percentile(arr, 75)),
            'ssi_p95':   float(np.percentile(arr, 95)),
        })
    ssi_grp = pd.DataFrame(stats_rows).sort_values('ssi_mean', ascending=False).reset_index(drop=True)
    ssi_grp.to_csv(f'{OUT_DIR}/ssi_group_summary.csv', index=False, float_format='%.6f')
    print('\n2. 各组 SSI 汇总:')
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(ssi_grp.to_string(index=False))

    # ============ 3. Kruskal + vs pilo Mann-Whitney + Cliff's delta ============
    arr_by_group = {g: df.loc[df['group'] == g, 'SSI'].values for g in DRUG_GROUPS
                    if (df['group'] == g).any()}
    kw_h, kw_p = kruskal(*arr_by_group.values())
    print(f'\n3a. 四组 Kruskal-Wallis 整体差异: H={kw_h:.2f}, p={kw_p:.2e}')

    ref = 'pilo组'
    mw_rows = []
    ref_arr = arr_by_group[ref]
    for g in DRUG_GROUPS:
        if g == ref: continue
        if g not in arr_by_group: continue
        drg_arr = arr_by_group[g]
        u_stat, p_val = mannwhitneyu(drg_arr, ref_arr, alternative='two-sided')
        d = cliffs_delta(ref_arr, drg_arr)     # d < 0 => drug SSI < pilo => 有效
        delta_mean = float(np.mean(drg_arr) - np.mean(ref_arr))
        direction = 'beneficial (↓SSI)' if d < 0 else 'worse (↑SSI)' if d > 0 else 'equal'
        mw_rows.append({
            'drug_vs_pilo': f'{g} vs {ref}',
            'drug_group': g,
            'drug_n': len(drg_arr),
            'pilo_n': len(ref_arr),
            'drug_ssi_mean': float(np.mean(drg_arr)),
            'pilo_ssi_mean': float(np.mean(ref_arr)),
            'delta_mean_drug_minus_pilo': delta_mean,
            'mann_whitney_U': u_stat,
            'p_value_two_sided': p_val,
            'p_adj_BH': np.nan,
            'cliffs_delta': d,
            'effect_size_magn': 'small' if abs(d)<0.147 else 'medium' if abs(d)<0.33 else 'large',
            'direction': direction,
            'significant_at_0.05': p_val < ALPHA,
            'stars': p_to_stars(p_val),
        })
    mw_df = pd.DataFrame(mw_rows)
    # BH 校正 p 值
    if len(mw_df) > 0:
        idx = mw_df['p_value_two_sided'].argsort().values
        p_sorted = mw_df['p_value_two_sided'].values[idx]
        m = len(p_sorted)
        qadj = np.empty(m, dtype=float)
        for i in range(m-1, -1, -1):
            if i == m-1:
                qadj[i] = p_sorted[i]
            else:
                qadj[i] = min(p_sorted[i] * (m/(i+1)), qadj[i+1])
        padj = np.empty(m)
        padj[idx] = qadj
        mw_df['p_adj_BH'] = padj
        mw_df['significant_after_BH'] = mw_df['p_adj_BH'] < ALPHA

    mw_df.to_csv(f'{OUT_DIR}/ssi_stats_vs_pilo.csv', index=False, float_format='%.6e')
    print('\n3b. 各药物 vs pilo (两独立样本 Mann-Whitney + Cliff\'s delta):')
    cols_show = ['drug_group', 'drug_ssi_mean', 'pilo_ssi_mean',
                 'delta_mean_drug_minus_pilo', 'p_value_two_sided', 'p_adj_BH',
                 'cliffs_delta', 'effect_size_magn', 'direction', 'stars']
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(mw_df[cols_show].to_string(index=False))

    # ============ 4. 画图 ============
    df_plot = df[df['group'].isin(DRUG_GROUPS)].copy()
    df_plot['group_en'] = df_plot['group'].map(GROUP_EN)
    group_order = [GROUP_EN[g] for g in DRUG_GROUPS if (df_plot['group']==g).any()]
    palette = {'Pilo': '#4C78A8', 'VPA': '#F58518', 'miR-OE': '#E45756', 'Sponges': '#72B7B2'}

    # 4a. 箱线图
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df_plot, x='group_en', y='SSI', order=group_order,
                palette=palette, ax=ax, width=0.55, fliersize=2,
                boxprops=dict(alpha=.9), medianprops=dict(color='black', lw=1.5))
    sns.stripplot(data=df_plot, x='group_en', y='SSI', order=group_order,
                  size=1.2, color='black', alpha=0.12, jitter=0.25, ax=ax)
    ax.set_xlabel('Drug group (training set, true Ictal only)')
    ax.set_ylabel('SSI = P(Ictal) (lower = less severe seizure)')
    ax.set_title('Seizure Severity Index (SSI) by drug group (training set Ictal)')
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(ssi_grp.loc[ssi_grp['drug_group']==ref, 'ssi_mean'].values[0],
               color=palette['Pilo'], ls='--', lw=1, label='Pilo mean SSI')
    ax.legend(loc='lower left')
    # 加显著性横线
    ybase = 0.97
    for i, (_, row) in enumerate(mw_df.iterrows()):
        if not row['significant_after_BH']: continue
        g_en = GROUP_EN[row['drug_group']]
        if g_en not in group_order: continue
        j_g = group_order.index(g_en)
        j_ref = group_order.index('Pilo')
        lo, hi = sorted([j_ref, j_g])
        y = ybase - i*0.025
        if y < 0.82: y = 0.82
        ax.plot([lo+0.02, hi-0.02], [y, y], color='black', lw=1)
        ax.text((lo+hi)/2, y+0.005, p_to_stars(row['p_adj_BH']),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/ssi_boxplot.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'\n4a. 已保存箱线图: {OUT_DIR}/ssi_boxplot.png')

    # 4b. KDE 分布图
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for g_en in group_order:
        orig = [k for k, v in GROUP_EN.items() if v == g_en][0]
        arr = arr_by_group.get(orig, [])
        if len(arr) == 0: continue
        sns.kdeplot(arr, label=g_en, fill=True, alpha=0.18, lw=2,
                    color=palette.get(g_en, None), ax=ax)
    ax.set_xlabel('SSI = P(Ictal)')
    ax.set_ylabel('Density')
    ax.set_title('SSI distribution by drug group (training set Ictal)')
    ax.set_xlim(0, 1.02)
    ax.legend(title='Group')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/ssi_kde.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'4b. 已保存 KDE 图: {OUT_DIR}/ssi_kde.png')

    # 4c. 平均 SSI 柱状图 + 显著性标记
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ssi_plot = ssi_grp.copy()
    ssi_plot['group_en'] = ssi_plot['drug_group'].map(GROUP_EN)
    ssi_plot = ssi_plot.set_index('group_en').loc[group_order].reset_index()
    bars = ax.bar(ssi_plot['group_en'], ssi_plot['ssi_mean'],
                  yerr=ssi_plot['ssi_std'] / np.sqrt(ssi_plot['n_segments']),  # SEM
                  color=[palette.get(g, '#888') for g in ssi_plot['group_en']],
                  edgecolor='black', capsize=5, alpha=0.9)
    for bar, n, val in zip(bars, ssi_plot['n_segments'], ssi_plot['ssi_mean']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{val:.2f}\n(n={n})', ha='center', va='bottom', fontsize=9)
    ref_mean = ssi_grp.loc[ssi_grp['drug_group']==ref, 'ssi_mean'].values[0]
    ax.axhline(ref_mean, color=palette['Pilo'], ls='--', lw=1, label='Pilo mean')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Drug group (training set, true Ictal)')
    ax.set_ylabel('Mean SSI ± SEM')
    ax.set_title('Mean SSI per drug group (training set Ictal)')
    ymax = ssi_plot['ssi_mean'].max() + 0.14
    for _, row in mw_df.iterrows():
        g_en = GROUP_EN[row['drug_group']]
        if g_en not in group_order: continue
        if not row['significant_after_BH']:
            star = 'n.s.'
            c = 'gray'
        else:
            star = p_to_stars(row['p_adj_BH'])
            c = 'darkred' if 'worse' in row['direction'] else 'darkgreen'
        j_ref = group_order.index('Pilo')
        j_g = group_order.index(g_en)
        lo, hi = sorted([j_ref, j_g])
        ax.plot([lo+0.05, hi-0.05], [ymax, ymax], color='black', lw=1)
        ax.text((lo+hi)/2, ymax+0.005, f"{star}\nd={row['cliffs_delta']:+.2f}",
                ha='center', va='bottom', fontsize=9, color=c, fontweight='bold')
        ymax += 0.09
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/ssi_efficacy_bars.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'4c. 已保存柱状图: {OUT_DIR}/ssi_efficacy_bars.png')

    print('\n' + '='*70)
    print('【SSI 分析结论 (训练集 Ictal 子集)】')
    print('='*70)
    for _, row in mw_df.iterrows():
        sig = '显著' if row['significant_after_BH'] else '不显著'
        print(f"  {row['drug_group']:>10} vs pilo:  mean SSI {row['drug_ssi_mean']:.3f} "
              f"vs {row['pilo_ssi_mean']:.3f}  Δ={row['delta_mean_drug_minus_pilo']:+.3f}  "
              f"Cliff's d={row['cliffs_delta']:+.3f}({row['effect_size_magn']:<6})  "
              f"p={row['p_value_two_sided']:.2e}(BH后p={row['p_adj_BH']:.2e}, {sig})  "
              f"→ {row['direction']}")
    print(f"\n  Kruskal-Wallis 四组整体 H={kw_h:.2f}, p={kw_p:.2e}")
    print(f"\n  SSI 指标定义: SSI(S) = P_final(Ictal), 仅 S ∈ S_ictal. 越小说明药物抑制发作越有效.")
    print(f"\n输出目录: {OUT_DIR}")


if __name__ == '__main__':
    main()
