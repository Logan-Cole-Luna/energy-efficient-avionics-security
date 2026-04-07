#!/usr/bin/env python3
"""
generate_plots.py
-----------------
Visualization suite for the NSL-centric Satellite CAN IDS workflow.

Generates figures in results/plots/:
    model_comparison.pdf
    tree_structure_nsl.pdf
    tree_structure_can.pdf
    confusion_matrices.pdf
    threshold_sweep.pdf
    quantization_metrics.pdf
    quantization_size.pdf
    quantization_latency.pdf
    benchmark_can_comparison.pdf
    cross_dataset_heatmap.pdf
    feature_importance.pdf
    roc_curves.pdf
    radar_summary.pdf
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib as mpl
mpl.use('pgf')
mpl.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'text.usetex': True,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parents[1]
RES = BASE / 'results'
MODELS = BASE / 'models' / 'trained_models'
DATA = BASE / 'datasets' / 'CAN_FROM_BENCHMARK'
NSL_DATA = BASE / 'datasets' / 'NSL-KDD'
PLOTS = RES / 'plots'
PLOTS.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.1)


def save(fig, name, dpi=150):
    path = (PLOTS / name).with_suffix('.pdf')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path.name}')


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _load_model_selection_nsl():
    return _load_json(RES / 'model_selection_nsl.json')


def _get_tiny_nsl():
    rows = _load_model_selection_nsl()
    for row in rows:
        if row['model'] == 'TinyDecisionTree':
            return row
    raise RuntimeError('TinyDecisionTree not found in model_selection_nsl.json')


def plot_model_comparison():
    rows = _load_model_selection_nsl()
    model_order = [r['model'] for r in rows]
    short_names = [m.replace('Light', 'L.').replace('Compact', 'C.').replace('Tiny', 'T.').replace('Micro', 'M.') for m in model_order]
    colors = sns.color_palette('deep', len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Model Selection - NSL-KDD Only', fontsize=14, fontweight='bold')

    metrics = [
        ('accuracy', 'Accuracy (%)', 100.0),
        ('recall', 'Recall (%)', 100.0),
        ('f1', 'F1 (%)', 100.0),
        ('fpr', 'FPR (%)', 100.0),
    ]

    for ax, (metric, ylabel, scale) in zip(axes.flatten(), metrics):
        vals = [r[metric] * scale for r in rows]
        bars = ax.bar(short_names, vals, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=20)
        ax.set_ylim(0, 110)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

        idx = model_order.index('TinyDecisionTree')
        bars[idx].set_edgecolor('#e74c3c')
        bars[idx].set_linewidth(2.5)

    fig.tight_layout()
    save(fig, 'model_comparison.pdf')


def _load_can_model():
    with open(MODELS / 'nsl_can' / 'features.json') as f:
        feature_names = json.load(f)
    clf = joblib.load(MODELS / 'nsl_can' / 'tree.joblib')
    return clf, feature_names


def plot_trees():
    clf, feature_names = _load_can_model()

    fig, ax = plt.subplots(figsize=(38, 18))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=['Normal', 'Attack'],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
        impurity=False,
        proportion=True,
        precision=2,
    )
    ax.set_title('TinyDecisionTree - NSL-CAN Benchmark IDS', fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04)
    save(fig, 'tree_structure_nsl.pdf', dpi=320)

    # Alternate render kept for backward compatibility with prior manuscript assets.
    fig, ax = plt.subplots(figsize=(30, 14))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=['Normal', 'Attack'],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
        impurity=False,
        proportion=True,
        precision=2,
    )
    ax.set_title('TinyDecisionTree - CAN IDS Structure', fontsize=14, fontweight='bold')
    save(fig, 'tree_structure_can.pdf', dpi=280)


def plot_confusion_matrices():
    nsl = _get_tiny_nsl()
    nsl_can = _load_json(RES / 'nsl_can_results.json')['overall']

    datasets = [
        ('NSL-KDD (Network)', nsl),
        ('NSL-CAN (Benchmark)', nsl_can),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('Confusion Matrices - TinyDecisionTree', fontsize=14, fontweight='bold')

    for ax, (name, m) in zip(axes, datasets):
        tn, fp, fn, tp = m['tn'], m['fp'], m['fn'], m['tp']
        cm = np.array([[tn, fp], [fn, tp]])
        total = cm.sum()
        annot = np.array([
            [f'{tn:,}\n({tn/total:.1%})', f'{fp:,}\n({fp/total:.1%})'],
            [f'{fn:,}\n({fn/total:.1%})', f'{tp:,}\n({tp/total:.1%})'],
        ])
        sns.heatmap(
            cm,
            annot=annot,
            fmt='',
            ax=ax,
            cmap='Blues',
            linewidths=0.5,
            xticklabels=['Pred Normal', 'Pred Attack'],
            yticklabels=['Actual Normal', 'Actual Attack'],
            cbar=False,
            annot_kws={'size': 10},
        )
        ax.set_title(name, fontweight='bold', fontsize=11)

    fig.tight_layout()
    save(fig, 'confusion_matrices.pdf')


def plot_threshold_sweep():
    nsl = _get_tiny_nsl()
    metrics = ['precision', 'recall', 'f1', 'fpr']
    labels = ['Precision', 'Recall', 'F1', 'FPR']
    vals = [nsl[m] * 100 for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('Reference Metrics - TinyDecisionTree (NSL-KDD)', fontsize=14, fontweight='bold')
    bars = ax.bar(labels, vals, color=['#5DA5DA', '#60BD68', '#F17CB0', '#FAA43A'], edgecolor='white')
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 105)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f'{v:.1f}', ha='center', fontsize=9)
    save(fig, 'threshold_sweep.pdf')


def _load_quantization_results():
    candidates = [
        RES / 'quantization_results.json',
        RES / 'nsl_quantization_results.json',
    ]
    for path in candidates:
        if path.exists():
            return _load_json(path)
    raise RuntimeError('Missing quantization_results.json (or nsl_quantization_results.json)')


def plot_quantization():
    d = _load_quantization_results()
    q = d['quantization_results']

    precisions = [p for p in ['float32', 'int16', 'int8'] if p in q]
    if len(precisions) < 2:
        raise RuntimeError('Quantization results must include at least float32 and int16')

    acc = [q[p]['accuracy'] * 100 for p in precisions]
    rec = [q[p]['recall'] * 100 for p in precisions]
    f1v = [q[p]['f1'] * 100 for p in precisions]
    fpr = [q[p]['fpr'] * 100 for p in precisions]
    size = [q[p]['size_kb'] for p in precisions]
    inf = [q[p]['inference_us'] for p in precisions]

    x = np.arange(len(precisions))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('Quantization Metrics (NSL-CAN)', fontsize=13, fontweight='bold')
    ax.bar(x - 1.5 * w, acc, w, label='Accuracy', color='#3498db')
    ax.bar(x - 0.5 * w, rec, w, label='Recall', color='#e74c3c')
    ax.bar(x + 0.5 * w, f1v, w, label='F1', color='#2ecc71')
    ax.bar(x + 1.5 * w, fpr, w, label='FPR', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in precisions])
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    for i, vals in enumerate(zip(acc, rec, f1v, fpr)):
        for v, offset in zip(vals, [-1.5, -0.5, 0.5, 1.5]):
            ax.text(i + offset * w, v + 1, f'{v:.0f}', ha='center', fontsize=7)
    save(fig, 'quantization_metrics.pdf')

    q_colors = ['#2196F3', '#FF9800', '#F44336'][:len(precisions)]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.suptitle('Quantization Size (NSL-CAN)', fontsize=13, fontweight='bold')
    ax.bar([p.upper() for p in precisions], size, color=q_colors, edgecolor='white')
    ax.set_ylabel('Size (KB)')
    for i, v in enumerate(size):
        ax.text(i, v + 0.01, f'{v:.3f} KB', ha='center', fontsize=10, fontweight='bold')
    save(fig, 'quantization_size.pdf')

    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.suptitle('Quantization Latency (NSL-CAN)', fontsize=13, fontweight='bold')
    ax.bar([p.upper() for p in precisions], inf, color=q_colors, edgecolor='white')
    ax.set_ylabel('Inference Time (us)')
    for i, v in enumerate(inf):
        ax.text(i, v + 0.05, f'{v:.2f} us', ha='center', fontsize=10, fontweight='bold')
    save(fig, 'quantization_latency.pdf')


def plot_benchmark_can_comparison():
    nsl = _get_tiny_nsl()
    nsl_can = _load_json(RES / 'nsl_can_results.json')

    labels = ['NSL-KDD', 'NSL-CAN']
    metrics = {
        'Accuracy': [nsl['accuracy'] * 100, nsl_can['overall']['accuracy'] * 100],
        'Recall': [nsl['recall'] * 100, nsl_can['overall']['recall'] * 100],
        'Precision': [nsl['precision'] * 100, nsl_can['overall']['precision'] * 100],
        'F1': [nsl['f1'] * 100, nsl_can['overall']['f1'] * 100],
        'ROC-AUC': [nsl['roc_auc'] * 100, nsl_can['overall']['roc_auc'] * 100],
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle('NSL Network vs NSL-CAN Benchmark Comparison', fontsize=14, fontweight='bold')

    x = np.arange(len(labels))
    width = 0.15
    colors = sns.color_palette('Set2', len(metrics))
    for i, (mname, vals) in enumerate(metrics.items()):
        ax.bar(x + (i - 2) * width, vals, width, label=mname, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=3)
    save(fig, 'benchmark_can_comparison.pdf')


def plot_cross_dataset_heatmap():
    nsl = _get_tiny_nsl()
    nsl_can = _load_json(RES / 'nsl_can_results.json')['overall']

    df = pd.DataFrame(
        {
            'ROC-AUC': [nsl['roc_auc'] * 100, nsl_can['roc_auc'] * 100],
            'Accuracy': [nsl['accuracy'] * 100, nsl_can['accuracy'] * 100],
            'F1': [nsl['f1'] * 100, nsl_can['f1'] * 100],
            'Recall': [nsl['recall'] * 100, nsl_can['recall'] * 100],
            'Precision': [nsl['precision'] * 100, nsl_can['precision'] * 100],
            'FPR': [nsl['fpr'] * 100, nsl_can['fpr'] * 100],
        },
        index=['NSL-KDD (Network)', 'NSL-CAN (Benchmark)'],
    )

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    sns.heatmap(
        df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        linewidths=0.5,
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={'label': 'Score (%)'},
        annot_kws={'size': 10, 'weight': 'bold'},
    )
    ax.set_title('TinyDecisionTree - NSL Performance Summary (%)', fontsize=13, fontweight='bold', pad=10)
    save(fig, 'cross_dataset_heatmap.pdf')


def plot_feature_importances():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Importances - TinyDecisionTree (Gini)', fontsize=14, fontweight='bold')

    nsl_clf, nsl_feat_names, _, _ = _fit_nsl_tiny_model_from_data()
    can_clf = joblib.load(MODELS / 'nsl_can' / 'tree.joblib')
    with open(MODELS / 'nsl_can' / 'features.json') as f:
        can_feat_names = json.load(f)

    configs = [
        (nsl_clf, nsl_feat_names, 'NSL-KDD (41 features)', axes[0], 'Blues_r', 15),
        (can_clf, can_feat_names, 'NSL-CAN (14 features)', axes[1], 'Reds_r', 14),
    ]

    for clf, feat_names, title, ax, palette_name, max_features in configs:

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(max_features, len(feat_names))
        top_idx = indices[:top_n]

        feat_labels = [feat_names[i] for i in top_idx][::-1]
        feat_vals = importances[top_idx][::-1]

        y_pos = np.arange(top_n)
        palette_grad = sns.color_palette(palette_name, top_n)
        ax.barh(y_pos, feat_vals, color=palette_grad, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_labels, fontsize=9)
        ax.set_xlabel('Gini Importance')
        ax.set_title(title, fontweight='bold')
        x_pad = max(feat_vals) * 0.01 if max(feat_vals) > 0 else 0.001
        for i, v in enumerate(feat_vals):
            ax.text(v + x_pad, i, f'{v:.3f}', va='center', fontsize=8)

    fig.tight_layout()
    save(fig, 'feature_importance.pdf')


def _load_can_test():
    test_df = pd.read_csv(DATA / 'nsl_can_test_features.csv')
    with open(MODELS / 'nsl_can' / 'features.json') as f:
        feat_names = json.load(f)
    X = test_df[feat_names].values.astype(np.float32)
    y = test_df['label'].astype(int).values
    scaler = joblib.load(MODELS / 'nsl_can' / 'scaler.joblib')
    return scaler.transform(X), y


def _nsl_cols():
    return [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty',
    ]


@lru_cache(maxsize=1)
def _fit_nsl_tiny_model_from_data():
    train_file = NSL_DATA / 'KDDTrain+.txt'
    test_file = NSL_DATA / 'KDDTest+.txt'
    if not train_file.exists() or not test_file.exists():
        raise RuntimeError('NSL-KDD train/test files are required for NSL feature importance and ROC plots')

    cols = _nsl_cols()
    train_df = pd.read_csv(train_file, header=None, names=cols)
    test_df = pd.read_csv(test_file, header=None, names=cols)
    for df in [train_df, test_df]:
        if 'difficulty' in df.columns:
            df.drop(columns=['difficulty'], inplace=True)

    combined = pd.concat([train_df, test_df], ignore_index=True)
    split_idx = len(train_df)

    combined['binary_label'] = (combined['label'] != 'normal').astype(int)
    y = combined['binary_label'].values
    X = combined.drop(columns=['label', 'binary_label'])

    for col in X.select_dtypes(include=['object', 'string']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.values[:split_idx])
    X_test = scaler.transform(X.values[split_idx:])

    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)
    clf.fit(X_train, y[:split_idx])

    feature_names = X.columns.tolist()
    return clf, feature_names, X_test, y[split_idx:]


def plot_roc_curves():
    clf = joblib.load(MODELS / 'nsl_can' / 'tree.joblib')
    X_test, y_test = _load_can_test()
    y_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    nsl_clf, _, nsl_X_test, nsl_y_test = _fit_nsl_tiny_model_from_data()
    nsl_y_prob = nsl_clf.predict_proba(nsl_X_test)[:, 1]
    nsl_fpr, nsl_tpr, _ = roc_curve(nsl_y_test, nsl_y_prob)
    nsl_roc_auc = auc(nsl_fpr, nsl_tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title('ROC Curves - TinyDecisionTree Across Datasets', fontsize=13, fontweight='bold')
    ax.plot(nsl_fpr, nsl_tpr, color='#4CAF50', lw=2.2, label=f'NSL-KDD (AUC={nsl_roc_auc:.4f})')
    ax.plot(fpr, tpr, color='#F44336', lw=2.2, label=f'NSL-CAN (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    save(fig, 'roc_curves.pdf')


def plot_radar_summary():
    categories = ['Recall', 'Precision', 'F1', 'Accuracy', 'Low FPR', 'ROC-AUC']

    nsl = _get_tiny_nsl()
    nsl_can = _load_json(RES / 'nsl_can_results.json')['overall']
    rows = {
        'NSL-KDD': [
            nsl['recall'] * 100,
            nsl['precision'] * 100,
            nsl['f1'] * 100,
            nsl['accuracy'] * 100,
            (1 - nsl['fpr']) * 100,
            nsl['roc_auc'] * 100,
        ],
        'NSL-CAN': [
            nsl_can['recall'] * 100,
            nsl_can['precision'] * 100,
            nsl_can['f1'] * 100,
            nsl_can['accuracy'] * 100,
            (1 - nsl_can['fpr']) * 100,
            nsl_can['roc_auc'] * 100,
        ],
    }

    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    colors = ['#4CAF50', '#F44336']
    for (name, vals), color in zip(rows.items(), colors):
        v = [x / 100 for x in vals]
        v += v[:1]
        ax.plot(angles, v, color=color, lw=2.2, label=name)
        ax.fill(angles, v, color=color, alpha=0.14)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.set_title('TinyDecisionTree - NSL Deployment Summary', fontsize=13, fontweight='bold', pad=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), fontsize=10)
    save(fig, 'radar_summary.pdf')


if __name__ == '__main__':
    print(f'\nGenerating plots - {PLOTS}\n')
    steps = [
        ('Model comparison', plot_model_comparison),
        ('Tree structures', plot_trees),
        ('Confusion matrices', plot_confusion_matrices),
        ('Threshold sweep', plot_threshold_sweep),
        ('Quantization trade-offs', plot_quantization),
        ('Benchmark-CAN compare', plot_benchmark_can_comparison),
        ('Cross-dataset heatmap', plot_cross_dataset_heatmap),
        ('Feature importances', plot_feature_importances),
        ('ROC curves', plot_roc_curves),
        ('Radar summary', plot_radar_summary),
    ]

    for name, fn in steps:
        print(f'[{name}]')
        try:
            fn()
        except Exception as e:
            print(f'  ERROR: {e}')

    print(f'\nDone. All plots saved to {PLOTS}')
