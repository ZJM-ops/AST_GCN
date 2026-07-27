import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler, CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from utils.early_stopping import EarlyStopping
from utils.visualization import plot_training_curves


def train_one_model(model, model_name, train_loader, val_loader, test_loader,
                    device, run_save_dir, epochs=100, lr=3e-4, weight_decay=1e-4,
                    patience=10, warmup_epochs=15, grad_clip=0.8,
                    class_weights=None, accum_steps=2, mixed_precision=True,
                    cosine_eta_min=1e-6, cosine_restarts=3):
    epochs = int(epochs)
    lr = float(lr)
    weight_decay = float(weight_decay)
    patience = int(patience)
    warmup_epochs = int(warmup_epochs)
    grad_clip = float(grad_clip)
    accum_steps = int(accum_steps)
    cosine_eta_min = float(cosine_eta_min)
    cosine_restarts = int(cosine_restarts)
    
    print(f"\n{'='*60}")
    print(f"🚀 训练模型: {model_name}")
    print(f"{'='*60}")
    print(f"  训练参数:")
    print(f"    epochs: {epochs}, lr: {lr:.2e}, weight_decay: {weight_decay:.2e}")
    print(f"    patience: {patience}, warmup_epochs: {warmup_epochs}, grad_clip: {grad_clip}")
    print(f"    accum_steps: {accum_steps}, mixed_precision: {mixed_precision}")
    print(f"    cosine_eta_min: {cosine_eta_min:.2e}, cosine_restarts: {cosine_restarts}")
    print(f"    device: {device}")

    model_dir = os.path.join(run_save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 差异化学习率：backbone用较低lr微调，GCN/attention用较高lr
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.3, 'weight_decay': weight_decay},
        {'params': other_params, 'lr': lr, 'weight_decay': weight_decay}
    ])
    
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0,
                                     total_iters=warmup_epochs)
        T_mult = 2 if cosine_restarts > 1 else 1
        cos_scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                     T_0=max((epochs - warmup_epochs) // cosine_restarts, 1),
                                                     T_mult=T_mult,
                                                     eta_min=cosine_eta_min)
        scheduler = ChainedScheduler([warmup_scheduler, cos_scheduler])
    else:
        T_mult = 2 if cosine_restarts > 1 else 1
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                 T_0=max(epochs // cosine_restarts, 1),
                                                 T_mult=T_mult,
                                                 eta_min=cosine_eta_min)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = GradScaler('cuda', enabled=mixed_precision)

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
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

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:3d} | LR: {current_lr:.2e} | "
              f"Train Loss: {epoch_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"  🛑 早停触发，在第 {epoch+1} 轮停止训练")
            break

    early_stopping.restore_best_weights(model)

    best_path = os.path.join(model_dir, f'best_{model_name}.pth')
    torch.save(model.state_dict(), best_path)

    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         model_name, os.path.join(model_dir, f'{model_name}_training_curves.png'))

    pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses, "val_loss": val_losses,
        "train_acc": train_accs, "val_acc": val_accs
    }).to_csv(os.path.join(model_dir, "train_val_log.csv"), index=False)

    return model