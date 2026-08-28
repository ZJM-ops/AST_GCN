#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于三分类 AST-Net_GCN 模型的药物疗效评估 (训练集版)
================================================
原理:
  所有动物均经 pilo 诱导癫痫建模, 仅给药不同.
  - pilo组:       仅建模, 不给药 (癫痫模型对照 / 疗效基准)
  - VPA组:        丙戊酸钠 (经典抗癫痫药 AED)
  - mir过表达组:  miR 过表达干预
  - sponges组:    海绵对照
模型输出三类概率: Baseline(正常) / Ictal(发作) / Chronic(慢性癫痫)

关键设计:
  仅使用 split_data_42_3class/train_images.txt 列出的训练集样本做推理.
  train split 覆盖全部 4 组 × 3 时相 (val/test 不全), 可形成均衡对比表.
  疗效判断 = 同一真实时相下, 药物组 vs pilo组 的概率差异:
    - 真实 Chronic 时相: 药物 P(Baseline)↑ 且 P(Chronic)↓ => 抑制癫痫慢性化有效
    - 真实 Ictal  时相: 药物 P(Baseline)↑ 且 P(Ictal)↓   => 抑制急性发作有效

注意: 训练集是模型见过的数据, 统计显著性受拟合影响, 结果仅作模式观察, 不作泛化证据.

输出 (输出目录 drug_efficacy_train_only):
  1. per_segment_predictions.csv     每段 seg 的三概率 + 组别 + 真实时相 + 预测
  2. group_phase_summary.csv        组别×真实时相 的平均概率 + 预测分布 (核心疗效表)
  3. group_summary.csv              各药物组整体预测分布 + 平均概率
  4. group_confusion_matrix.csv     各药物组 3x3 混淆矩阵 (行归一化%)
  5. group_phase_confusion_matrix.csv 各药物组×真实时相 3x3 混淆矩阵
  6. stats_kruskal.csv              组间 Kruskal-Wallis (按真实时相分层)
  7. stats_vs_pilo_mannwhitney.csv 同一时相下各药物 vs pilo 的 Mann-Whitney + delta
  8. phase_comparison.png           3 时相 × 4 组 的平均概率对比图 (核心疗效图)
  9. efficacy_vs_pilo.png           各药物 vs pilo 的 delta 图 (Chronic/Ictal 时相)
  10. group_pred_distribution.png   各组整体预测类别占比
  11. group_mean_probability.png     各组整体三类平均概率
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from models import ResNeXt_GCN_Bridge
from datasets.dataset import CompleteGraphDataset

# ============================ 配置 ============================
# 仅用训练集 split 里的样本做推理 (split_data_42_3class/train_images.txt)
TRAIN_SPLIT_TXT = '/home/student/s230005071/shu/split_data_42_3class/train_images.txt'
MODEL_PATH = '/home/student/s230005071/shu/code2/summary_results/10mineeg/results/20260804_184146_ASTNet_Standard_3Class/AST-Net_GCN/best_AST-Net_GCN.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
IMAGE_SIZE = 224
NUM_NODES = 8
NUM_CLASSES = 3
CLASS_NAMES = ['Baseline', 'Ictal', 'Chronic']
LABEL_MAPPING = {'Baseline': 0, 'Ictal': 1, 'Chronic': 2, 'Chronic-28d': 2}

OUTPUT_DIR = '/home/student/s230005071/shu/code2/summary_results/10mineeg/drug_efficacy/drug_efficacy_train_only'

# 药物组顺序 (对照 -> 干预). pilo 组作为未给药建模对照(基准)
DRUG_GROUPS = ['pilo组', 'VPA组', 'mir过表达组', 'sponges组']
# 画图英文显示名 (避免中文字体缺失出 tofu)
GROUP_DISPLAY = {
    'pilo组': 'Pilo (no drug)',
    'VPA组': 'VPA (AED)',
    'mir过表达组': 'miR-overexpr',
    'sponges组': 'Sponges (ctrl)',
}
CLASS_COLOR = {'Baseline': '#4caf50', 'Ictal': '#f44336', 'Chronic': '#ff9800'}

# ============================ 工具函数 ============================
def extract_group(path):
    for g in DRUG_GROUPS:
        if g in path:
            return g
    return 'Unknown'


def collect_image_paths():
    """读取 train_images.txt 列出的训练集样本路径"""
    with open(TRAIN_SPLIT_TXT, 'r') as f:
        paths = sorted(line.strip() for line in f if line.strip())
    return paths


def load_model():
    print(f'加载模型: {MODEL_PATH}')
    model = ResNeXt_GCN_Bridge(
        num_classes=NUM_CLASSES,
        num_nodes=NUM_NODES,
        backbone_name='resnext50_32x4d',
        gradient_checkpointing=False,   # 推理时关闭以加速
        use_anatomical_prior=True,      # 必须与训练一致 (注册 anatomical_adj buffer + prior_weight)
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print(f'模型已加载到 {DEVICE}')
    return model


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ============================ 推理 ============================
def run_inference(model):
    paths = collect_image_paths()
    print(f'扫描到 PNG 数: {len(paths)}')
    transform = build_transform()
    dataset = CompleteGraphDataset(paths, transform=transform, label_mapping=LABEL_MAPPING)
    print(f'完整 8 通道 segment 数: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    records = []
    idx = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='推理中'):
            inputs = inputs.to(DEVICE)              # [B, 8, 3, 224, 224]
            outputs = model(inputs)                 # [B, 3]
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.numpy()
            bs = probs.shape[0]
            for i in range(bs):
                seg_id = dataset.seg_ids[idx]
                seg_paths = dataset.segs[seg_id]
                group = extract_group(seg_paths[0])
                true_l = int(labels_np[i])
                pred_l = int(probs[i].argmax())
                records.append({
                    'seg_id': seg_id,
                    'group': group,
                    'true_label': true_l,
                    'true_phase': CLASS_NAMES[true_l],
                    'pred_label': pred_l,
                    'pred_class': CLASS_NAMES[pred_l],
                    'prob_Baseline': float(probs[i, 0]),
                    'prob_Ictal': float(probs[i, 1]),
                    'prob_Chronic': float(probs[i, 2]),
                })
                idx += 1
    return pd.DataFrame(records)


# ============================ 聚合分析 ============================
def group_phase_aggregate(df):
    """核心疗效表: 组别 × 真实时相"""
    rows = []
    for g in DRUG_GROUPS:
        for tc in range(NUM_CLASSES):
            sub = df[(df['group'] == g) & (df['true_label'] == tc)]
            if len(sub) == 0:
                continue
            n = len(sub)
            rows.append({
                'group': g,
                'true_phase': CLASS_NAMES[tc],
                'n_segments': n,
                'mean_prob_Baseline': sub['prob_Baseline'].mean(),
                'mean_prob_Ictal': sub['prob_Ictal'].mean(),
                'mean_prob_Chronic': sub['prob_Chronic'].mean(),
                'pred_Baseline_pct': (sub['pred_label'] == 0).mean() * 100,
                'pred_Ictal_pct': (sub['pred_label'] == 1).mean() * 100,
                'pred_Chronic_pct': (sub['pred_label'] == 2).mean() * 100,
            })
    return pd.DataFrame(rows)


def group_aggregate(df):
    """各药物组整体预测分布 + 平均概率"""
    rows = []
    for g in DRUG_GROUPS:
        sub = df[df['group'] == g]
        if len(sub) == 0:
            continue
        rows.append({
            'group': g,
            'n_segments': len(sub),
            'accuracy': (sub['pred_label'] == sub['true_label']).mean(),
            'pred_Baseline_pct': (sub['pred_label'] == 0).mean() * 100,
            'pred_Ictal_pct': (sub['pred_label'] == 1).mean() * 100,
            'pred_Chronic_pct': (sub['pred_label'] == 2).mean() * 100,
            'mean_prob_Baseline': sub['prob_Baseline'].mean(),
            'mean_prob_Ictal': sub['prob_Ictal'].mean(),
            'mean_prob_Chronic': sub['prob_Chronic'].mean(),
        })
    return pd.DataFrame(rows)


def confusion_matrix_table(df, group_col='group'):
    """按 group_col 分组的 3x3 混淆矩阵 (行=真实, 列=预测), 行归一化%"""
    rows = []
    keys = df[group_col].unique().tolist()
    for g in DRUG_GROUPS:
        if g not in keys:
            continue
        sub = df[df[group_col] == g]
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for t, p in zip(sub['true_label'], sub['pred_label']):
            cm[t, p] += 1
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_pct = cm / row_sum * 100
        for t in range(NUM_CLASSES):
            for p in range(NUM_CLASSES):
                rows.append({
                    group_col: g,
                    'true_class': CLASS_NAMES[t],
                    'pred_class': CLASS_NAMES[p],
                    'count': int(cm[t, p]),
                    'row_pct': float(cm_pct[t, p]),
                })
    return pd.DataFrame(rows)


def kruskal_test(df):
    """按真实时相分层, 各类概率在 4 组间的 Kruskal-Wallis"""
    out = []
    present_groups = [g for g in DRUG_GROUPS if (df['group'] == g).any()]
    for tc in range(NUM_CLASSES):
        phase = CLASS_NAMES[tc]
        sub_phase = df[df['true_label'] == tc]
        if len(sub_phase) == 0:
            continue
        for cls_name, col in [('Baseline', 'prob_Baseline'),
                              ('Ictal', 'prob_Ictal'),
                              ('Chronic', 'prob_Chronic')]:
            samples = [sub_phase.loc[sub_phase['group'] == g, col].values
                       for g in present_groups if (sub_phase['group'] == g).any()]
            samples = [s for s in samples if len(s) > 0]
            if len(samples) >= 2:
                H, p = stats.kruskal(*samples)
                out.append({
                    'test': 'Kruskal-Wallis',
                    'true_phase': phase,
                    'metric': f'prob_{cls_name}',
                    'groups': ','.join(present_groups),
                    'statistic': H,
                    'p_value': p,
                    'significant': p < 0.05,
                })
    return pd.DataFrame(out)


def vs_pilo_mannwhitney(df):
    """同一真实时相下, 各药物组 vs pilo 的 Mann-Whitney U (two-sided) + delta
    重点指标:
      true Chronic: prob_Chronic (越低越好), prob_Baseline (越高越好)
      true Ictal:   prob_Ictal   (越低越好), prob_Baseline (越高越好)
      true Baseline: prob_Baseline (sanity, 越高越好)
    """
    out = []
    pilo = 'pilo组'
    focus = {
        'Baseline': ['prob_Baseline'],
        'Ictal':    ['prob_Ictal', 'prob_Baseline'],
        'Chronic':  ['prob_Chronic', 'prob_Baseline'],
    }
    for phase, cols in focus.items():
        pilo_mask = (df['group'] == pilo) & (df['true_phase'] == phase)
        pilo_data = df.loc[pilo_mask]
        if len(pilo_data) == 0:
            continue
        for drug in DRUG_GROUPS:
            if drug == pilo:
                continue
            drug_data = df.loc[(df['group'] == drug) & (df['true_phase'] == phase)]
            if len(drug_data) == 0:
                continue
            for col in cols:
                x = pilo_data[col].values
                y = drug_data[col].values
                U, p = stats.mannwhitneyu(x, y, alternative='two-sided')
                delta = float(y.mean() - x.mean())   # drug - pilo
                # 疗效方向: Chronic/Ictal 概率越低越好(delta<0 为有效); Baseline 概率越高越好(delta>0 为有效)
                if col in ('prob_Chronic', 'prob_Ictal'):
                    efficacy = 'beneficial' if delta < 0 else 'worse'
                else:  # prob_Baseline
                    efficacy = 'beneficial' if delta > 0 else 'worse'
                out.append({
                    'true_phase': phase,
                    'metric': col,
                    'control': pilo,
                    'drug': drug,
                    'pilo_mean': float(x.mean()),
                    'drug_mean': float(y.mean()),
                    'delta_drug_minus_pilo': delta,
                    'efficacy_direction': efficacy,
                    'mannwhitney_U': U,
                    'p_value': p,
                    'significant': p < 0.05,
                })
    return pd.DataFrame(out)


# ============================ 画图 ============================
def plot_phase_comparison(group_phase, out_path):
    """3 个时相面板, 每面板 4 组的 P(B)/P(I)/P(C) 分组柱状"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax_i, phase in enumerate(['Baseline', 'Ictal', 'Chronic']):
        ax = axes[ax_i]
        sub = group_phase[group_phase['true_phase'] == phase].set_index('group')
        sub = sub.reindex([g for g in DRUG_GROUPS if g in sub.index])
        if len(sub) == 0:
            ax.set_title(f'true={phase}\n(no data)')
            continue
        labels = [GROUP_DISPLAY.get(g, g) for g in sub.index]
        x = np.arange(len(sub))
        w = 0.27
        for j, (cls, col) in enumerate([('Baseline', 'mean_prob_Baseline'),
                                        ('Ictal', 'mean_prob_Ictal'),
                                        ('Chronic', 'mean_prob_Chronic')]):
            vals = sub[col].values
            bars = ax.bar(x + (j - 1) * w, vals, w, label=f'P({cls})', color=CLASS_COLOR[cls])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.2f}',
                        ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=9)
        ax.set_title(f'true phase = {phase}')
        ax.set_ylim(0, 1.05)
        if ax_i == 0:
            ax.set_ylabel('Mean softmax probability')
        ax.legend(fontsize=8)
    fig.suptitle('Mean class probability by drug group (per true phase)', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'已保存: {out_path}')


def plot_efficacy_vs_pilo(mw_df, out_path):
    """各药物 vs pilo 的 delta (drug - pilo), 按时相 + 指标分面板"""
    if len(mw_df) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_i, phase in enumerate(['Baseline', 'Ictal', 'Chronic']):
        ax = axes[ax_i]
        sub = mw_df[mw_df['true_phase'] == phase].copy()
        if len(sub) == 0:
            ax.set_title(f'true={phase}\n(no data)')
            continue
        metrics = sub['metric'].unique().tolist()
        x = np.arange(len(metrics))
        w = 0.2
        drugs = [d for d in DRUG_GROUPS if d != 'pilo组' and d in sub['drug'].unique().tolist()]
        for k, drug in enumerate(drugs):
            vals = []
            for m in metrics:
                row = sub[(sub['drug'] == drug) & (sub['metric'] == m)]
                vals.append(row['delta_drug_minus_pilo'].values[0] if len(row) else 0.0)
            bars = ax.bar(x + (k - 1) * w, vals, w, label=GROUP_DISPLAY.get(drug, drug))
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2,
                        v + (0.005 if v >= 0 else -0.01), f'{v:+.3f}',
                        ha='center', fontsize=8, va='bottom' if v >= 0 else 'top')
        ax.axhline(0, color='k', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('prob_', 'P(') + ')' for m in metrics], fontsize=9)
        ax.set_title(f'true phase = {phase}\nΔ = drug − pilo (pilo=0)')
        ax.set_ylabel('Δ probability (drug − pilo)')
        ax.legend(fontsize=8)
    fig.suptitle('Drug efficacy vs pilo control (Δ probability)', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'已保存: {out_path}')


def plot_group_distribution(group_summary, out_path):
    groups = group_summary['group'].tolist()
    labels = [GROUP_DISPLAY.get(g, g) for g in groups]
    x = np.arange(len(groups))
    w = 0.27
    _, ax = plt.subplots(figsize=(11, 6))
    b0 = group_summary['pred_Baseline_pct'].values
    b1 = group_summary['pred_Ictal_pct'].values
    b2 = group_summary['pred_Chronic_pct'].values
    ax.bar(x - w, b0, w, label='Baseline (normal)', color=CLASS_COLOR['Baseline'])
    ax.bar(x,     b1, w, label='Ictal (seizure)',   color=CLASS_COLOR['Ictal'])
    ax.bar(x + w, b2, w, label='Chronic (epilepsy)', color=CLASS_COLOR['Chronic'])
    ax.set_ylabel('Predicted proportion (%)')
    ax.set_title('Overall predicted class distribution by drug group')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    for i in range(len(groups)):
        ax.text(i - w, b0[i] + 1, f'{b0[i]:.0f}', ha='center', fontsize=8)
        ax.text(i,     b1[i] + 1, f'{b1[i]:.0f}', ha='center', fontsize=8)
        ax.text(i + w, b2[i] + 1, f'{b2[i]:.0f}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'已保存: {out_path}')


def plot_group_probability(group_summary, out_path):
    groups = group_summary['group'].tolist()
    labels = [GROUP_DISPLAY.get(g, g) for g in groups]
    x = np.arange(len(groups))
    w = 0.27
    _, ax = plt.subplots(figsize=(11, 6))
    p0 = group_summary['mean_prob_Baseline'].values
    p1 = group_summary['mean_prob_Ictal'].values
    p2 = group_summary['mean_prob_Chronic'].values
    ax.bar(x - w, p0, w, label='P(Baseline)', color=CLASS_COLOR['Baseline'])
    ax.bar(x,     p1, w, label='P(Ictal)',    color=CLASS_COLOR['Ictal'])
    ax.bar(x + w, p2, w, label='P(Chronic)',  color=CLASS_COLOR['Chronic'])
    ax.set_ylabel('Mean softmax probability')
    ax.set_title('Overall mean class probability by drug group')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.legend()
    for i in range(len(groups)):
        ax.text(i - w, p0[i] + 0.01, f'{p0[i]:.2f}', ha='center', fontsize=8)
        ax.text(i,     p1[i] + 0.01, f'{p1[i]:.2f}', ha='center', fontsize=8)
        ax.text(i + w, p2[i] + 0.01, f'{p2[i]:.2f}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'已保存: {out_path}')


# ============================ 汇总打印 ============================
def print_summary(group_phase, group_summary, mw_df):
    print('\n' + '=' * 80)
    print('【药物疗效分析结果汇总 (训练集版 4组×3时相)】')
    print('=' * 80)

    print('\n1) 组别×真实时相 的平均概率 (核心疗效表):')
    show = ['group', 'true_phase', 'n_segments',
            'mean_prob_Baseline', 'mean_prob_Ictal', 'mean_prob_Chronic',
            'pred_Baseline_pct', 'pred_Ictal_pct', 'pred_Chronic_pct']
    print(group_phase[show].to_string(index=False))

    print('\n2) 各药物组整体预测分布:')
    print(group_summary[['group', 'n_segments', 'accuracy',
                         'pred_Baseline_pct', 'pred_Ictal_pct', 'pred_Chronic_pct',
                         'mean_prob_Baseline', 'mean_prob_Ictal', 'mean_prob_Chronic']].to_string(index=False))

    print('\n3) 疗效判断 (Δ = drug − pilo, 按 Mann-Whitney):')
    if len(mw_df) > 0:
        sig = mw_df[['true_phase', 'metric', 'drug', 'pilo_mean', 'drug_mean',
                     'delta_drug_minus_pilo', 'p_value', 'efficacy_direction', 'significant']]
        print(sig.to_string(index=False))
        print('\n疗效解读要点:')
        print('  - true Chronic 时相: ΔP(Chronic)<0 且显著 => 抑制癫痫慢性化有效')
        print('                       ΔP(Baseline)>0 且显著 => 慢性期脑电向正常基线回归')
        print('  - true Ictal  时相: ΔP(Ictal)<0 且显著 => 抑制急性发作有效')
        print('                       ΔP(Baseline)>0 且显著 => 发作期脑电向基线回归')
    print('=' * 80)


# ============================ 主程序 ============================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model()
    df = run_inference(model)

    per_seg = os.path.join(OUTPUT_DIR, 'per_segment_predictions.csv')
    df.to_csv(per_seg, index=False, encoding='utf-8-sig')
    print(f'已保存每段预测: {per_seg}  ({len(df)} 段)')

    group_phase = group_phase_aggregate(df)
    gp_path = os.path.join(OUTPUT_DIR, 'group_phase_summary.csv')
    group_phase.to_csv(gp_path, index=False, encoding='utf-8-sig')
    print(f'已保存组×时相核心表: {gp_path}')

    group_summary = group_aggregate(df)
    gs_path = os.path.join(OUTPUT_DIR, 'group_summary.csv')
    group_summary.to_csv(gs_path, index=False, encoding='utf-8-sig')
    print(f'已保存分组汇总: {gs_path}')

    cm_df = confusion_matrix_table(df, group_col='group')
    cm_df.to_csv(os.path.join(OUTPUT_DIR, 'group_confusion_matrix.csv'), index=False, encoding='utf-8-sig')
    cm_phase_df = confusion_matrix_table(df, group_col='group')
    cm_phase_df.to_csv(os.path.join(OUTPUT_DIR, 'group_phase_confusion_matrix.csv'), index=False, encoding='utf-8-sig')
    print('已保存混淆矩阵')

    kw_df = kruskal_test(df)
    kw_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_kruskal.csv'), index=False, encoding='utf-8-sig')
    print(f'已保存 Kruskal-Wallis 检验')

    mw_df = vs_pilo_mannwhitney(df)
    mw_df.to_csv(os.path.join(OUTPUT_DIR, 'stats_vs_pilo_mannwhitney.csv'), index=False, encoding='utf-8-sig')
    print(f'已保存 vs pilo Mann-Whitney 检验')

    plot_phase_comparison(group_phase, os.path.join(OUTPUT_DIR, 'phase_comparison.png'))
    plot_efficacy_vs_pilo(mw_df, os.path.join(OUTPUT_DIR, 'efficacy_vs_pilo.png'))
    plot_group_distribution(group_summary, os.path.join(OUTPUT_DIR, 'group_pred_distribution.png'))
    plot_group_probability(group_summary, os.path.join(OUTPUT_DIR, 'group_mean_probability.png'))

    print_summary(group_phase, group_summary, mw_df)
    print(f'\n所有结果已保存到: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
