import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
from collections import Counter
import time
import traceback

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from models import ResNeXt_GCN_Bridge, BaselineResNeXt, BaselineResNet, BaselineEfficientNet, BaselineCNNLSTM, EEGNet, ST_GCN_EEG, EEGConformer1D_GCN_Model
from datasets import CHBMITGraphDataset, CHBMIT1DGraphDataset
from trainers import train_one_model, evaluate_model
from utils.visualization import plot_comparison


def load_config(config_path='config/config_chbmit.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_image_loaders(train_paths, val_paths, test_paths, label_mapping, transform, batch_size, num_channels=18, num_workers=2, pin_memory=False):
    train_dataset = CHBMITGraphDataset(train_paths, transform=transform, label_mapping=label_mapping, num_channels=num_channels)
    val_dataset = CHBMITGraphDataset(val_paths, transform=transform, label_mapping=label_mapping, num_channels=num_channels)
    test_dataset = CHBMITGraphDataset(test_paths, transform=transform, label_mapping=label_mapping, num_channels=num_channels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def build_1d_loaders(train_1d_path, val_1d_path, test_1d_path, label_mapping, batch_size, num_channels=18, num_workers=2, pin_memory=False):
    train_dataset = CHBMIT1DGraphDataset(train_1d_path, training=True, label_mapping=label_mapping, num_channels=num_channels)
    val_dataset = CHBMIT1DGraphDataset(val_1d_path, training=False, label_mapping=label_mapping, num_channels=num_channels)
    test_dataset = CHBMIT1DGraphDataset(test_1d_path, training=False, label_mapping=label_mapping, num_channels=num_channels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def train_single_model_worker(args):
    results_queue, model_name, model_cls, model_kwargs, data_type, data_args, run_save_dir, gpu_id, config = args

    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    torch.cuda.set_per_process_memory_fraction(0.85, gpu_id)

    print(f"🚀 [{model_name}] 使用GPU: {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")

    try:
        model = model_cls(**model_kwargs).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"模型参数量 [{model_name}]: {n_params:.1f}M")

        num_classes = config['model']['num_classes']
        num_workers = config['training'].get('num_workers', 2)
        pin_memory = config['training'].get('pin_memory', False)
        mixed_precision = config['training'].get('mixed_precision', True)

        if data_type == 'image':
            train_cfg = config['training']
            train_paths, val_paths, test_paths, label_mapping, transform = data_args
            num_channels = config['model']['num_nodes']
            train_loader, val_loader, test_loader = build_image_loaders(
                train_paths, val_paths, test_paths, label_mapping, transform, 
                train_cfg['batch_size'], num_channels, num_workers, pin_memory)
        else:
            train_cfg = config.get('training_1d', config['training'])
            train_1d_path, val_1d_path, test_1d_path, label_mapping = data_args
            num_channels = config['model']['num_nodes']
            train_loader, val_loader, test_loader = build_1d_loaders(
                train_1d_path, val_1d_path, test_1d_path, label_mapping, 
                train_cfg['batch_size'], num_channels, num_workers, pin_memory)

        train_labels = train_loader.dataset.get_labels()
        class_counts = Counter(train_labels)
        total = sum(class_counts.values())
        class_weights = torch.tensor(
            [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)],
            dtype=torch.float
        ).to(device)

        trained_model = train_one_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            run_save_dir=run_save_dir,
            epochs=train_cfg['epochs'],
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'],
            patience=train_cfg['patience'],
            warmup_epochs=train_cfg['warmup_epochs'],
            grad_clip=train_cfg['grad_clip'],
            class_weights=class_weights,
            accum_steps=train_cfg['accum_steps'],
            mixed_precision=mixed_precision,
            cosine_eta_min=train_cfg.get('cosine_eta_min', 1e-6),
            cosine_restarts=train_cfg.get('cosine_restarts', 3),
        )

        result = evaluate_model(
            model=trained_model,
            test_loader=test_loader,
            device=device,
            model_name=model_name,
            save_dir=run_save_dir,
            num_classes=num_classes,
            mixed_precision=mixed_precision,
        )
        result['params_M'] = n_params
        result['gpu_id'] = gpu_id

    except Exception as e:
        print(f"❌ [{model_name}] 训练失败: {str(e)}")
        traceback.print_exc()
        result = {
            'model': model_name,
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'auc': 0.0,
            'params_M': 0.0,
            'gpu_id': gpu_id,
            'error': str(e)
        }
    finally:
        import gc
        for var_name in ['model', 'trained_model', 'train_loader', 'val_loader', 'test_loader', 'class_weights']:
            if var_name in locals():
                exec(f'del {var_name}')
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
        torch.cuda.empty_cache()

    if results_queue is not None:
        results_queue.put(result)
    else:
        return result


def main(config, model_filter=None):
    num_gpus = torch.cuda.device_count()
    print(f"📦 可用GPU数量: {num_gpus}")

    if num_gpus == 0:
        print("❌ 没有可用GPU，使用CPU")
    else:
        print(f"✅ GPU列表: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        for i in range(num_gpus):
            mem_info = torch.cuda.mem_get_info(i)
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)} - 可用显存: {mem_info[0]/1e9:.2f} GB / {mem_info[1]/1e9:.2f} GB")

    split_data_dir = config['data']['split_data_dir']
    train_images_path = os.path.join(split_data_dir, config['data']['train_images'])
    val_images_path = os.path.join(split_data_dir, config['data']['val_images'])
    test_images_path = os.path.join(split_data_dir, config['data']['test_images'])
    train_1d_path = os.path.join(split_data_dir, config['data']['train_1d'])
    val_1d_path = os.path.join(split_data_dir, config['data']['val_1d'])
    test_1d_path = os.path.join(split_data_dir, config['data']['test_1d'])

    model_save_dir = config['model']['save_dir']
    os.makedirs(model_save_dir, exist_ok=True)

    batch_size = config['training']['batch_size']
    num_classes = config['model']['num_classes']
    num_nodes = config['model']['num_nodes']
    image_size = config['model']['image_size']
    backbone_name = config['model'].get('backbone_name', 'resnext50_32x4d')
    gradient_checkpointing = config['model'].get('gradient_checkpointing', False)
    label_mapping = config['label_mapping']

    run_dir_name = (datetime.now().strftime("%Y%m%d_%H%M%S")
                    + "_ASTNet_CHBMIT_2Class")
    run_save_dir = os.path.join(model_save_dir, run_dir_name)
    os.makedirs(run_save_dir, exist_ok=True)

    print(f"结果保存目录: {run_save_dir}")

    config_save_path = os.path.join(run_save_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"配置文件已保存: {config_save_path}")

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(train_images_path, 'r') as f:
        train_paths = [l.strip() for l in f if l.strip()]
    with open(val_images_path, 'r') as f:
        val_paths = [l.strip() for l in f if l.strip()]
    with open(test_images_path, 'r') as f:
        test_paths = [l.strip() for l in f if l.strip()]

    image_data_args = (train_paths, val_paths, test_paths, label_mapping, val_transform)
    one_d_data_args = (train_1d_path, val_1d_path, test_1d_path, label_mapping)

    all_tasks = [
        ('AST-Net_GCN_resnet18', ResNeXt_GCN_Bridge, {'num_classes': num_classes, 'backbone_name': 'resnet18', 'gradient_checkpointing': gradient_checkpointing},
         'image', image_data_args),
        ('AST-Net_GCN_resnext50_32x4d', ResNeXt_GCN_Bridge, {'num_classes': num_classes, 'backbone_name': 'resnext50_32x4d', 'gradient_checkpointing': gradient_checkpointing},
         'image', image_data_args),
        ('Baseline_ResNeXt_resnext50_32x4d', BaselineResNeXt, {'num_classes': num_classes, 'backbone_name': 'resnext50_32x4d', 'gradient_checkpointing': gradient_checkpointing},
         'image', image_data_args),
        ('Baseline_ResNet_resnet18', BaselineResNet, {'num_classes': num_classes, 'backbone_name': 'resnet18', 'gradient_checkpointing': gradient_checkpointing},
         'image', image_data_args),
        ('Baseline_EffNet_efficientnet_b0', BaselineEfficientNet, {'num_classes': num_classes, 'backbone_name': 'efficientnet_b0', 'gradient_checkpointing': gradient_checkpointing},
         'image', image_data_args),
        ('Baseline_CNNLSTM_resnet18', BaselineCNNLSTM, {'num_classes': num_classes, 'backbone_name': 'resnet18', 'gradient_checkpointing': gradient_checkpointing},
         'image', image_data_args),
        ('EEGNet', EEGNet, {'num_classes': num_classes, 'num_channels': num_nodes},
         '1d', one_d_data_args),
        ('ST-GCN_EEG', ST_GCN_EEG, {'num_classes': num_classes, 'num_nodes': num_nodes},
         '1d', one_d_data_args),
        ('EEGConformer_GCN', EEGConformer1D_GCN_Model, {'num_classes': num_classes, 'num_nodes': num_nodes},
         '1d', one_d_data_args),
    ]

    if model_filter:
        if isinstance(model_filter, list):
            all_tasks = [t for t in all_tasks if t[0] in model_filter]
            print(f"\n🎯 多模型模式，训练: {', '.join(model_filter)}")
        else:
            all_tasks = [t for t in all_tasks if t[0] == model_filter]
            print(f"\n🎯 单模型模式，仅训练: {model_filter}")
    
    if num_gpus <= 1:
        print("\n⚠️ 单卡模式，串行训练所有模型")
        all_results = []
        for task in all_tasks:
            model_name, model_cls, model_kwargs, data_type, data_args = task
            args = (None, model_name, model_cls, model_kwargs, data_type, data_args,
                    run_save_dir, 0, config)
            result = train_single_model_worker(args)
            if result:
                all_results.append(result)
            torch.cuda.empty_cache()
    else:
        print(f"\n🚀 多卡并行模式 ({num_gpus} GPUs)，同时训练多个模型")
        mp.set_start_method('spawn', force=True)

        results_queue = mp.Queue()
        all_results = []
        processes = []

        for idx, task in enumerate(all_tasks):
            model_name, model_cls, model_kwargs, data_type, data_args = task
            gpu_id = idx % num_gpus

            print(f"\n📦 准备训练 [{model_name}]，分配到 GPU {gpu_id}")
            print(f"   模型参数: {model_kwargs}")

            args = (results_queue, model_name, model_cls, model_kwargs, data_type, data_args,
                    run_save_dir, gpu_id, config)

            p = mp.Process(target=train_single_model_worker, args=(args,))
            processes.append((p, model_name))
            p.start()

        print(f"\n⏳ 等待所有 {len(processes)} 个模型训练完成...")
        for p, model_name in processes:
            p.join()

        while not results_queue.empty():
            result = results_queue.get()
            all_results.append(result)
            if 'error' in result:
                print(f"⚠️ [{result['model']}] 训练失败: {result['error']}")

        torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('f1', ascending=False).reset_index(drop=True)

    print("\n" + "="*70)
    print("🏆 CHB-MIT 模型性能对比汇总 (按 Macro F1 降序)")
    print("="*70)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    results_df.to_csv(os.path.join(run_save_dir, 'model_comparison_results.csv'), index=False)
    plot_comparison(results_df, run_save_dir)

    print(f"\n✅ 所有实验完成。结果保存于: {run_save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AST-Net Training Pipeline for CHB-MIT Dataset')
    parser.add_argument('--config', type=str, default='config/config_chbmit.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                        help='Specify a single model to train (e.g., AST-Net_GCN_resnet18)')
    parser.add_argument('--models', type=str, default=None,
                        help='Specify multiple models to train, separated by comma (e.g., AST-Net_GCN_resnet18,EEGNet)')
    args = parser.parse_args()

    config = load_config(args.config)
    
    model_filter = None
    if args.models:
        model_filter = [m.strip() for m in args.models.split(',')]
    elif args.model:
        model_filter = args.model
    
    main(config, model_filter=model_filter)