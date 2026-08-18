#train.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from utils.early_stopping import EarlyStopping
from utils.visualization import plot_training_curves


def train_one_model(model, model_name, train_loader, val_loader, test_loader,
                    device, run_save_dir, epochs=100, lr=3e-4, weight_decay=1e-4,
                    patience=10, warmup_epochs=15, grad_clip=0.8,
                    class_weights=None, accum_steps=2, mixed_precision=True,
                    cosine_eta_min=1e-6, cosine_restarts=3, optimizer_type='sgd',
                    early_stop_monitor='val_acc'):
    epochs = int(epochs)
    lr = float(lr)
    weight_decay = float(weight_decay)
    patience = int(patience)
    grad_clip = float(grad_clip)
    accum_steps = int(accum_steps)
    warmup_epochs = int(warmup_epochs)
    optimizer_type = optimizer_type.lower()

    if optimizer_type == 'adamw':
        optimizer_config = {
            'type': 'AdamW',
            'lr': lr,
            'weight_decay': weight_decay,
        }
    else:
        optimizer_type = 'sgd'
        optimizer_config = {
            'type': 'SGD',
            'lr': lr,
            'momentum': 0.9,
            'weight_decay': weight_decay,
        }

    print(f"\n{'='*60}")
    print(f"🚀 训练模型: {model_name}")
    print(f"{'='*60}")
    if optimizer_type == 'adamw':
        print(f"  优化器: AdamW")
    else:
        print(f"  优化器: SGD (momentum=0.9)")
    print(f"  训练参数:")
    print(f"    epochs: {epochs}, lr: {lr:.2e}, weight_decay: {weight_decay:.2e}")
    print(f"    patience: {patience}, grad_clip: {grad_clip}")
    print(f"    accum_steps: {accum_steps}, mixed_precision: {mixed_precision}")
    print(f"    scheduler: CosineAnnealingLR (T_max={epochs})")
    print(f"    device: {device}")

    model_dir = os.path.join(run_save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )

    # Setup warmup + cosine annealing
    if warmup_epochs > 0:
        warmup_lr_start = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr_start
        cosine_epochs = epochs - warmup_epochs
    else:
        cosine_epochs = epochs

    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cosine_epochs))

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = GradScaler('cuda', enabled=mixed_precision)

    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    # early_stopping = EarlyStopping(patience=patience, verbose=True, monitor='val_acc')
    early_stop_monitor = early_stop_monitor.lower()
    print(f"  📊 早停监控指标: {early_stop_monitor} (patience={patience})")
    early_stopping = EarlyStopping(patience=patience, verbose=True, monitor=early_stop_monitor)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Warmup: linearly increase LR
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            current_lr = warmup_lr_start + (lr - warmup_lr_start) * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        elif epoch == warmup_epochs:
            # Switch to cosine annealing at full LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        model.train()
        running_loss = 0.0
        train_preds, train_labels_all = [], []
        optimizer.zero_grad()

        for step, (inputs, labels) in enumerate(
                tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast('cuda', enabled=mixed_precision):
                outputs = model(inputs)
                loss = criterion(outputs, labels) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps * inputs.size(0)
            train_preds.extend(torch.argmax(outputs.detach(), dim=1).cpu().numpy())
            train_labels_all.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels_all, train_preds)
        train_losses.append(epoch_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        val_preds, val_labels_all = [], []
        with torch.no_grad(), autocast('cuda', enabled=mixed_precision):
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels_all, val_preds)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Only step scheduler after warmup phase
        if epoch >= warmup_epochs:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        phase = "Warmup" if epoch < warmup_epochs else "Cosine"
        print(f"  Epoch {epoch+1:3d} [{phase}] | LR: {current_lr:.2e} | "
              f"Train Loss: {epoch_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # early_stopping(val_loss, model)  # 旧版：仅用 val_loss
        early_stopping(val_loss, val_acc, model)  # 新版：同时传递 val_loss 和 val_acc
        if early_stopping.early_stop:
            print(f"  🛑 早停触发，在第 {epoch+1} 轮停止训练")
            break

    early_stopping.restore_best_weights(model)

    best_path = os.path.join(model_dir, f'best_{model_name}.pth')
    torch.save(model.state_dict(), best_path)

    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         model_name, os.path.join(model_dir, f'{model_name}_training_curves.png'))

    log_data = {
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses, "val_loss": val_losses,
        "train_acc": train_accs, "val_acc": val_accs
    }
    pd.DataFrame(log_data).to_csv(os.path.join(model_dir, "train_val_log.csv"), index=False)

    import json
    with open(os.path.join(model_dir, "optimizer_config.json"), 'w') as f:
        json.dump(optimizer_config, f, indent=2)

    return model