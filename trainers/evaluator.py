#evaluator.py
import os
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast

from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix


def evaluate_model(model, test_loader, device, model_name, save_dir, num_classes=3, mixed_precision=True):
    model.eval()
    test_probs, test_labels_list = [], []
    with torch.no_grad(), autocast('cuda', enabled=mixed_precision):
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            test_probs.extend(probs.cpu().numpy())
            test_labels_list.extend(labels.numpy())

    test_probs = np.array(test_probs)
    preds = np.argmax(test_probs, axis=1)
    labels_arr = np.array(test_labels_list)

    metrics = compute_metrics(labels_arr, preds, test_probs, num_classes)

    print(f"\n📊 [{model_name}] 测试集结果 (Macro):")
    print(f"  Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | "
          f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
          f"AUC: {metrics['auc']:.4f}")
    print(f"  混淆矩阵:\n{metrics['confusion_matrix']}")

    if num_classes == 2:
        class_names = ['Interictal', 'Ictal']
    else:
        class_names = ['Baseline', 'Ictal', 'Chronic']
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    plot_confusion_matrix(metrics['confusion_matrix'], class_names,
                          model_name, os.path.join(model_dir, f'{model_name}_confusion_matrix.png'))

    result = {
        'model': model_name,
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'auc': metrics['auc']
    }

    pd.DataFrame(result, index=[0]).to_csv(
        os.path.join(model_dir, "test_metrics.csv"), index=False)
    pd.DataFrame(metrics['confusion_matrix']).to_csv(
        os.path.join(model_dir, "confusion_matrix.csv"), index=False)

    return result