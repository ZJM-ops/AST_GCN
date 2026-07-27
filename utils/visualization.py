import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Train', color='steelblue')
    axes[0].plot(val_losses, label='Val', color='tomato', linestyle='--')
    axes[0].set_title(f'{model_name} — Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.4)

    axes[1].plot(train_accs, label='Train', color='steelblue')
    axes[1].plot(val_accs, label='Val', color='tomato', linestyle='--')
    axes[1].set_title(f'{model_name} — Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, model_name, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(f'{model_name} — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison(results_df, save_dir):
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    colors = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']

    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.bar(results_df['model'], results_df[metric], color=color, alpha=0.8,
                      edgecolor='white', linewidth=1.5)
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(results_df['model'], rotation=20, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.4)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle('Model Comparison — All Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()