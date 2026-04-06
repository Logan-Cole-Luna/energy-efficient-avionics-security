#!/usr/bin/env python3
"""
generate_plots.py
-----------------
Visualization suite for the Satellite CAN Bus Intrusion Detection paper.

Generates the following figures into results/plots/:
  model_comparison.png        - All 5 candidate models across 4 metrics
  tree_structure_nsl.png      - Decision tree structure (NSL-KDD)
  tree_structure_can.png      - Decision tree structure (NSL->CAN)
  confusion_matrices.png      - Confusion matrices across datasets
  threshold_sweep.png         - Precision / Recall / F1 vs threshold
  quantization_metrics.png    - Float32 vs Int16 vs Int8 metrics
  quantization_size.png       - Model size by quantization
  quantization_latency.png    - Inference latency by quantization
  benchmark_can_comparison.png - UNSW->CAN vs NSL->CAN overall comparison
  cross_dataset_heatmap.png   - Performance heatmap across datasets
  feature_importance.png      - Feature importances (NSL + CAN)
  roc_curves.png              - ROC curves across datasets
  radar_summary.png           - Radar chart across deployment scenarios
"""

import json

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parents[1]
RES    = BASE / 'results'
MODELS = BASE / 'models' / 'trained_models'
DATA   = BASE / 'datasets'
PLOTS  = RES / 'plots'
PLOTS.mkdir(exist_ok=True)

# ─── Style ───────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.15)
PALETTE   = sns.color_palette('deep')
HIGHLIGHT = '#e74c3c'   # red accent for recommended model

LIGHT_BLUE   = PALETTE[0]
ORANGE       = PALETTE[1]
GREEN        = PALETTE[2]
RED          = PALETTE[3]
PURPLE       = PALETTE[4]
BROWN        = PALETTE[5]

MODEL_COLORS = {
    'LightRandomForest': LIGHT_BLUE,
    'CompactExtraTrees': ORANGE,
    'MicroXGBoost':      GREEN,
    'TinyDecisionTree':  RED,
    'TinyXGBoost':       PURPLE,
}

MODEL_SELECTION_FILES = {
    'unsw': RES / 'model_selection_unsw.json',
    'nsl': RES / 'model_selection_nsl.json',
}


def save(fig, name, dpi=150):
    path = PLOTS / name
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _load_model_selection(dataset: str):
    return _load_json(MODEL_SELECTION_FILES[dataset])


def _get_model_result(dataset: str, model_name: str = 'TinyDecisionTree'):
    rows = _load_model_selection(dataset)
    for row in rows:
        if row['model'] == model_name:
            return row
    raise ValueError(f"Model '{model_name}' not found in {MODEL_SELECTION_FILES[dataset].name}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_comparison():
    unsw = _load_model_selection('unsw')
    nsl = _load_model_selection('nsl')

    by_model_unsw = {d['model']: d for d in unsw}
    by_model_nsl = {d['model']: d for d in nsl}
    models = [m for m in MODEL_COLORS if m in by_model_unsw and m in by_model_nsl]

    acc = [((by_model_unsw[m]['accuracy'] + by_model_nsl[m]['accuracy']) / 2) * 100 for m in models]
    f1 = [((by_model_unsw[m]['f1'] + by_model_nsl[m]['f1']) / 2) * 100 for m in models]
    size_kb = [(by_model_unsw[m]['size_kb'] + by_model_nsl[m]['size_kb']) / 2 for m in models]
    inf_ms = [((by_model_unsw[m]['inf_us'] + by_model_nsl[m]['inf_us']) / 2) / 1000.0 for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Model Selection – Mean Across UNSW-NB15 and NSL-KDD', fontsize=14, fontweight='bold')

    short_names = [m.replace('Light','L.').replace('Compact','C.').replace('Tiny','T.').replace('Micro','M.') for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    def _bar(ax, vals, title, ylabel, fmt='{:.1f}', annotate=True):
        bars = ax.bar(short_names, vals, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=20)
        if annotate:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                        fmt.format(v), ha='center', va='bottom', fontsize=9)
        # highlight TinyDecisionTree
        idx = models.index('TinyDecisionTree')
        bars[idx].set_edgecolor(HIGHLIGHT)
        bars[idx].set_linewidth(2.5)
        return ax

    _bar(axes[0,0], acc,     'Test Accuracy',       'Accuracy (%)',       '{:.2f}')
    _bar(axes[0,1], f1,      'F1 Score',             'F1 (%)',             '{:.2f}')
    _bar(axes[1,0], size_kb, 'Model Size',           'Size (KB)',          '{:.2f}')
    _bar(axes[1,1], inf_ms,  'Inference Time',       'Time (ms)',          '{:.3f}')

    legend_patch = mpatches.Patch(facecolor='white', edgecolor=HIGHLIGHT, linewidth=2.5, label='TinyDecisionTree (recommended)')
    fig.legend(handles=[legend_patch], loc='lower center', ncol=1, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, 'model_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 02 & 03 — Decision Tree Structures
# ══════════════════════════════════════════════════════════════════════════════
def plot_tree_structure(model_path, feature_names, title, out_name, figsize=(36, 18), fontsize=10):
    clf = joblib.load(model_path)

    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=['Normal', 'Attack'],
        filled=True,
        rounded=True,
        fontsize=fontsize,
        ax=ax,
        impurity=False,
        proportion=True,
        precision=2,
    )
    # Paper-oriented layout: generous canvas and margins for readable leaf labels.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    save(fig, out_name, dpi=320)


def plot_trees():
    # NSL->CAN benchmark model
    with open(MODELS / 'nsl_can' / 'features.json') as f:
        feat2 = json.load(f)
    plot_tree_structure(
        MODELS / 'nsl_can' / 'tree.joblib',
        feat2,
        'TinyDecisionTree - NSL->CAN Benchmark IDS',
        'tree_structure_nsl.png',
        figsize=(40, 20),
        fontsize=10,
    )

    # CAN IDS (UNSW->CAN)
    model_path = MODELS / 'unsw_can' / 'tree.joblib'
    feat_path = MODELS / 'unsw_can' / 'features.json'

    with open(feat_path) as f:
        feat_can = json.load(f)
    plot_tree_structure(
        model_path,
        feat_can,
        'TinyDecisionTree - UNSW->CAN Benchmark IDS',
        'tree_structure_can.png',
        figsize=(36, 16),
        fontsize=10,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Fig 04 — Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices():
    unsw_net = _get_model_result('unsw')
    nsl_net = _get_model_result('nsl')
    nsl_can = _load_json(RES / 'nsl_can_results.json')['overall']

    datasets = [
        ('UNSW-NB15', unsw_net),
        ('NSL-KDD', nsl_net),
        ('NSL->CAN', nsl_can),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Confusion Matrices – TinyDecisionTree', fontsize=14, fontweight='bold')

    for ax, (name, m) in zip(axes, datasets):
        tn, fp, fn, tp = m['tn'], m['fp'], m['fn'], m['tp']
        cm = np.array([[tn, fp], [fn, tp]])
        total = cm.sum()

        # Percentage labels
        annot = np.array([[f'{tn:,}\n({tn/total:.1%})', f'{fp:,}\n({fp/total:.1%})'],
                          [f'{fn:,}\n({fn/total:.1%})', f'{tp:,}\n({tp/total:.1%})']])

        sns.heatmap(
            cm, annot=annot, fmt='', ax=ax,
            cmap='Blues', linewidths=0.5,
            xticklabels=['Predicted\nNormal', 'Predicted\nAttack'],
            yticklabels=['Actual\nNormal', 'Actual\nAttack'],
            cbar=False, annot_kws={'size': 10},
        )
        acc = (tn + tp) / total
        recall = tp / (tp + fn)
        ax.set_title(f'{name}\nAcc={acc:.2%}  Recall={recall:.2%}', fontweight='bold', fontsize=11)

    fig.tight_layout()
    save(fig, 'confusion_matrices.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 05 — Threshold Sweep
# ══════════════════════════════════════════════════════════════════════════════
def plot_threshold_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Reference Metrics – TinyDecisionTree', fontsize=14, fontweight='bold')

    rows = [
        ('UNSW-NB15', _get_model_result('unsw'), axes[0], '#2196F3'),
        ('NSL-KDD', _get_model_result('nsl'), axes[1], '#4CAF50'),
    ]
    metric_names = ['precision', 'recall', 'f1', 'fpr']
    pretty = ['Precision', 'Recall', 'F1', 'FPR']

    for name, row, ax, color in rows:
        vals = [row[m] * 100 for m in metric_names]
        bars = ax.bar(pretty, vals, color=[color, '#e74c3c', '#2ecc71', '#e67e22'], edgecolor='white')
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel('Score (%)')
        ax.set_ylim(0, 105)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f'{v:.1f}', ha='center', fontsize=9)

    fig.tight_layout()
    save(fig, 'threshold_sweep.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 06 — Quantization Trade-offs
# ══════════════════════════════════════════════════════════════════════════════
def plot_quantization():
    datasets = ['unsw', 'nsl']
    labels = ['UNSW->CAN', 'NSL->CAN']
    float_sizes = [
        (MODELS / f'{d}_can' / f'{d}_can_float32.h').stat().st_size / 1024
        for d in datasets
    ]
    int16_sizes = [
        (MODELS / f'{d}_can' / f'{d}_can_int16.h').stat().st_size / 1024
        for d in datasets
    ]
    compression = [100 * (1 - (i / f)) for f, i in zip(float_sizes, int16_sizes)]

    # 6a: compression summary
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('Quantization Compression (Float32 -> Int16)', fontsize=13, fontweight='bold')
    bars = ax.bar(labels, compression, color=['#2196F3', '#FF9800'], edgecolor='white')
    ax.set_ylabel('Size Reduction (%)')
    ax.set_ylim(0, max(compression) * 1.25)
    ax.set_title('Header Size Reduction by Dataset', fontweight='bold')
    for bar, v in zip(bars, compression):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
    fig.tight_layout()
    save(fig, 'quantization_metrics.png')

    # 6b: Header size
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.suptitle('Quantized Header Size', fontsize=13, fontweight='bold')
    width = 0.35
    ax.bar(x - width / 2, float_sizes, width, label='Float32', color='#5DA5DA', edgecolor='white')
    ax.bar(x + width / 2, int16_sizes, width, label='Int16', color='#F17CB0', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Size (KB)')
    ax.set_title('Generated Header Size by Precision', fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, 'quantization_size.png')

    # 6c: Reference inference latency from CAN models
    unsw_can = _load_json(RES / 'unsw_can_results.json')
    nsl_can = _load_json(RES / 'nsl_can_results.json')
    inf = [unsw_can['inference_time_ms'], nsl_can['inference_time_ms']]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.suptitle('CAN Model Inference Latency (Reference)', fontsize=13, fontweight='bold')
    ax.bar(labels, inf, color=['#FAA43A', '#60BD68'], edgecolor='white')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('TinyDecisionTree Inference Latency', fontweight='bold')
    for i, v in enumerate(inf):
        ax.text(i, v + 0.001, f'{v:.4f} ms', ha='center', fontsize=10, fontweight='bold')
    fig.tight_layout()
    save(fig, 'quantization_latency.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 07 — Benchmark-to-CAN Comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_benchmark_can_comparison():
    unsw = _load_json(RES / 'unsw_can_results.json')
    nsl = _load_json(RES / 'nsl_can_results.json')

    labels = ['UNSW->CAN', 'NSL->CAN']
    metrics = {
        'Accuracy': [unsw['overall']['accuracy'] * 100, nsl['overall']['accuracy'] * 100],
        'Recall': [unsw['overall']['recall'] * 100, nsl['overall']['recall'] * 100],
        'Precision': [unsw['overall']['precision'] * 100, nsl['overall']['precision'] * 100],
        'F1': [unsw['overall']['f1'] * 100, nsl['overall']['f1'] * 100],
        'ROC-AUC': [unsw['overall']['roc_auc'] * 100, nsl['overall']['roc_auc'] * 100],
        'Low FPR': [(1 - unsw['overall']['fpr']) * 100, (1 - nsl['overall']['fpr']) * 100],
    }
    model_kb = [unsw['model_size_kb'], nsl['model_size_kb']]
    inf_ms = [unsw['inference_time_ms'], nsl['inference_time_ms']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Benchmark-to-CAN IDS Comparison (UNSW vs NSL)', fontsize=14, fontweight='bold')

    x = np.arange(len(labels))
    width = 0.12
    metric_names = list(metrics.keys())
    colors = sns.color_palette('Set2', len(metric_names))
    for i, (mname, vals) in enumerate(metrics.items()):
        axes[0].bar(x + (i - (len(metric_names)-1)/2) * width, vals, width, label=mname, color=colors[i])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Score (%)')
    axes[0].set_ylim(0, 105)
    axes[0].set_title('Detection Metrics', fontweight='bold')
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].bar(np.array(x) - 0.15, model_kb, 0.3, label='Model size (KB)', color='#5DA5DA')
    axes[1].set_ylabel('Model Size (KB)', color='#2c3e50')
    axes[1].tick_params(axis='y', labelcolor='#2c3e50')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)

    ax2 = axes[1].twinx()
    ax2.bar(np.array(x) + 0.15, inf_ms, 0.3, label='Inference (ms)', color='#FAA43A')
    ax2.set_ylabel('Inference Time (ms)', color='#8e44ad')
    ax2.tick_params(axis='y', labelcolor='#8e44ad')
    axes[1].set_title('Deployment Cost', fontweight='bold')

    for i, v in enumerate(model_kb):
        axes[1].text(i - 0.15, v + 0.03, f'{v:.2f}', ha='center', fontsize=9)
    for i, v in enumerate(inf_ms):
        ax2.text(i + 0.15, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)

    fig.tight_layout()
    save(fig, 'benchmark_can_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 08 — Cross-Dataset Performance Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_cross_dataset_heatmap():
    p2 = _get_model_result('unsw')
    p2b = _get_model_result('nsl')
    ucan = _load_json(RES / 'unsw_can_results.json')['overall']
    ncan = _load_json(RES / 'nsl_can_results.json')['overall']

    rows = {
        'UNSW-NB15\n(Network)': {
            'Accuracy': p2['accuracy'] * 100,
            'Precision': p2['precision'] * 100,
            'Recall': p2['recall'] * 100,
            'F1': p2['f1'] * 100,
            'FPR': p2['fpr'] * 100,
            'ROC-AUC': p2['roc_auc'] * 100,
        },
        'NSL-KDD\n(Network)': {
            'Accuracy': p2b['accuracy'] * 100,
            'Precision': p2b['precision'] * 100,
            'Recall': p2b['recall'] * 100,
            'F1': p2b['f1'] * 100,
            'FPR': p2b['fpr'] * 100,
            'ROC-AUC': p2b['roc_auc'] * 100,
        },
        'UNSW->CAN\n(Benchmark)': {
            'Accuracy': ucan['accuracy'] * 100,
            'Precision': ucan['precision'] * 100,
            'Recall': ucan['recall'] * 100,
            'F1': ucan['f1'] * 100,
            'FPR': ucan['fpr'] * 100,
            'ROC-AUC': ucan['roc_auc'] * 100,
        },
        'NSL->CAN\n(Benchmark)': {
            'Accuracy': ncan['accuracy'] * 100,
            'Precision': ncan['precision'] * 100,
            'Recall': ncan['recall'] * 100,
            'F1': ncan['f1'] * 100,
            'FPR': ncan['fpr'] * 100,
            'ROC-AUC': ncan['roc_auc'] * 100,
        },
    }
    df = pd.DataFrame(rows).T
    metric_order = ['ROC-AUC', 'Accuracy', 'F1', 'Recall', 'Precision', 'FPR']
    df = df[metric_order]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(
        df, annot=True, fmt='.2f', cmap='RdYlGn', linewidths=0.5,
        vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Score (%)'},
        annot_kws={'size': 12, 'weight': 'bold'},
    )
    ax.set_title('TinyDecisionTree – Cross-Dataset Performance Summary (%)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xticklabels(metric_order, fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, rotation=0)
    ax.set_xlabel('')

    fig.tight_layout()
    save(fig, 'cross_dataset_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 09 — Feature Importances
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importances():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Importances – TinyDecisionTree (Gini)', fontsize=14, fontweight='bold')

    configs = [
           (MODELS / 'unsw_can' / 'tree.joblib', MODELS / 'unsw_can' / 'features.json',
            'UNSW->CAN (14 features)', axes[0], '#3498db'),
           (MODELS / 'nsl_can' / 'tree.joblib', MODELS / 'nsl_can' / 'features.json',
            'CAN Bus IDS (NSL->CAN, 14 features)', axes[1], '#e74c3c'),
    ]

    for model_path, feat_path, title, ax, color in configs:
        clf = joblib.load(model_path)
        with open(feat_path) as f:
            feat_names = json.load(f)

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        # Show top-N (all for CAN, top 15 for UNSW)
        top_n = min(15, len(feat_names))
        top_idx = indices[:top_n]

        feat_labels = [feat_names[i] for i in top_idx]
        feat_vals   = importances[top_idx]

        # Horizontal bar chart
        y_pos = np.arange(top_n)
        palette_grad = sns.color_palette('Blues_r', top_n) if color == '#3498db' else sns.color_palette('Reds_r', top_n)
        ax.barh(y_pos, feat_vals[::-1], color=palette_grad, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_labels[::-1], fontsize=9)
        ax.set_xlabel('Gini Importance')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(0, max(feat_vals) * 1.2)
        for i, v in enumerate(feat_vals[::-1]):
            ax.text(v + max(feat_vals)*0.01, i, f'{v:.3f}', va='center', fontsize=8)

    fig.tight_layout()
    save(fig, 'feature_importance.png')


def _load_can_test(dataset: str):
    """Load benchmark-to-CAN feature test set."""
    test_df = pd.read_csv(DATA / 'CAN_FROM_BENCHMARK' / f'{dataset}_can_test_features.csv')
    y = test_df['label'].astype(int).values
    with open(MODELS / f'{dataset}_can' / 'features.json') as f:
        feat_names = json.load(f)
    X = test_df[feat_names].values.astype(np.float32)
    scaler = joblib.load(MODELS / f'{dataset}_can' / 'scaler.joblib')
    X_scaled = scaler.transform(X)
    return X_scaled, y


def plot_roc_curves():
    print("  Loading datasets for ROC curves (this may take a moment)...")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title('ROC Curves – TinyDecisionTree Across Datasets', fontsize=13, fontweight='bold')

    datasets = [
        ('UNSW->CAN\n(Benchmark)',      MODELS / 'unsw_can' / 'tree.joblib', lambda: _load_can_test('unsw'), '#F44336'),
        ('NSL->CAN\n(Benchmark)',       MODELS / 'nsl_can' / 'tree.joblib',  lambda: _load_can_test('nsl'),  '#FF9800'),
    ]

    for label, model_path, loader_fn, color in datasets:
        try:
            clf    = joblib.load(model_path)
            X_test, y_test = loader_fn()
            y_prob = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.2,
                    label=f'{label}  (AUC={roc_auc:.4f})')
        except Exception as e:
            print(f"    Warning: could not load {label}: {e}")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, 'roc_curves.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 11 — Dataset & Model Overview Summary (spider / radar chart)
# ══════════════════════════════════════════════════════════════════════════════
def plot_radar_summary():
    """Radar chart comparing the three deployment scenarios."""
    categories = ['Recall', 'Precision', 'F1', 'Accuracy', 'Low FPR', 'ROC-AUC']

    p2 = _get_model_result('unsw')
    p2b = _get_model_result('nsl')
    ucan = _load_json(RES / 'unsw_can_results.json')
    ncan = _load_json(RES / 'nsl_can_results.json')

    # Normalise FPR to "Low FPR" = 1 - FPR (higher is better)
    data_raw = {
        'UNSW-NB15': [
            p2['recall'] * 100,
            p2['precision'] * 100,
            p2['f1'] * 100,
            p2['accuracy'] * 100,
            (1 - p2['fpr']) * 100,
            p2['roc_auc'] * 100,
        ],
        'NSL-KDD': [
            p2b['recall'] * 100,
            p2b['precision'] * 100,
            p2b['f1'] * 100,
            p2b['accuracy'] * 100,
            (1 - p2b['fpr']) * 100,
            p2b['roc_auc'] * 100,
        ],
        'UNSW->CAN': [
            ucan['overall']['recall'] * 100,
            ucan['overall']['precision'] * 100,
            ucan['overall']['f1'] * 100,
            ucan['overall']['accuracy'] * 100,
            (1 - ucan['overall']['fpr']) * 100,
            ucan['overall']['roc_auc'] * 100,
        ],
        'NSL->CAN': [
            ncan['overall']['recall'] * 100,
            ncan['overall']['precision'] * 100,
            ncan['overall']['f1'] * 100,
            ncan['overall']['accuracy'] * 100,
            (1 - ncan['overall']['fpr']) * 100,
            ncan['overall']['roc_auc'] * 100,
        ],
    }

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors_r = ['#2196F3', '#4CAF50', '#F44336', '#FF9800']

    for (name, vals), color in zip(data_raw.items(), colors_r):
        vals_norm = [v / 100 for v in vals]
        vals_norm += vals_norm[:1]
        ax.plot(angles, vals_norm, color=color, lw=2.2, label=name)
        ax.fill(angles, vals_norm, color=color, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.set_title('TinyDecisionTree – Performance Radar\n(across all deployment scenarios)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=10)

    fig.tight_layout()
    save(fig, 'radar_summary.png')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"\nGenerating plots → {PLOTS}\n")

    steps = [
        ('Model comparison',        plot_model_comparison),
        ('Tree structures',             plot_trees),
        ('Confusion matrices',      plot_confusion_matrices),
        ('Threshold sweep',         plot_threshold_sweep),
        ('Quantization trade-offs', plot_quantization),
        ('Benchmark-CAN compare',   plot_benchmark_can_comparison),
        ('Cross-dataset heatmap',   plot_cross_dataset_heatmap),
        ('Feature importances',     plot_feature_importances),
        ('ROC curves',              plot_roc_curves),
        ('Radar summary',           plot_radar_summary),
    ]

    for name, fn in steps:
        print(f"[{name}]")
        try:
            fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone. All plots saved to {PLOTS}")
