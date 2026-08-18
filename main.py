#main.py
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

from models import ResNeXt_GCN_Bridge, BaselineMultiScaleResNeXt, BaselineResNeXt, BaselineResNet, BaselineEfficientNet, BaselineCNNLSTM, EEGNet, ST_GCN_EEG, BraindecodeEEGConformer
from datasets import CompleteGraphDataset, Complete1DGraphDataset, CHBMITGraphDataset, CHBMIT1DGraphDataset
from trainers import train_one_model, evaluate_model
from utils.visualization import plot_comparison


def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def detect_dataset_type(config):
    if 'dataset_type' in config:
        return config['dataset_type']
    label_mapping = config.get('label_mapping', {})
    if 'Interictal' in label_mapping or 'Ictal' in label_mapping and len(label_mapping) <= 2:
        if 'Interictal' in label_mapping:
            return 'chbmit'
    num_classes = config.get('model', {}).get('num_classes', 3)
    if num_classes == 2:
        return 'chbmit'
    return 'standard'


def build_image_loaders(train_paths, val_paths, test_paths, label_mapping, transform,
                        batch_size, dataset_type='standard', num_channels=8,
                        num_workers=2, pin_memory=False):
    if dataset_type == 'chbmit':
        train_dataset = CHBMITGraphDataset(train_paths, transform=transform, label_mapping=label_mapping, num_channels=num_channels)
        val_dataset = CHBMITGraphDataset(val_paths, transform=transform, label_mapping=label_mapping, num_channels=num_channels)
        test_dataset = CHBMITGraphDataset(test_paths, transform=transform, label_mapping=label_mapping, num_channels=num_channels)
    else:
        train_dataset = CompleteGraphDataset(train_paths, transform=transform, label_mapping=label_mapping)
        val_dataset = CompleteGraphDataset(val_paths, transform=transform, label_mapping=label_mapping)
        test_dataset = CompleteGraphDataset(test_paths, transform=transform, label_mapping=label_mapping)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def build_1d_loaders(train_1d_path, val_1d_path, test_1d_path, label_mapping,
                     batch_size, dataset_type='standard', num_channels=8,
                     num_workers=2, pin_memory=False):
    if dataset_type == 'chbmit':
        train_dataset = CHBMIT1DGraphDataset(train_1d_path, training=True, label_mapping=label_mapping, num_channels=num_channels)
        val_dataset = CHBMIT1DGraphDataset(val_1d_path, training=False, label_mapping=label_mapping, num_channels=num_channels)
        test_dataset = CHBMIT1DGraphDataset(test_1d_path, training=False, label_mapping=label_mapping, num_channels=num_channels)
    else:
        train_dataset = Complete1DGraphDataset(train_1d_path, training=True, label_mapping=label_mapping)
        val_dataset = Complete1DGraphDataset(val_1d_path, training=False, label_mapping=label_mapping)
        test_dataset = Complete1DGraphDataset(test_1d_path, training=False, label_mapping=label_mapping)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def train_single_model_worker(args):
    results_queue, model_name, model_cls, model_kwargs, data_type, data_args, run_save_dir, gpu_id, config = args

    import os
    
    # 直接使用传入的 gpu_id 参数
    actual_gpu_id = gpu_id
    
    if actual_gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(actual_gpu_id)
        print(f"🔧 [{model_name}] 设置CUDA_VISIBLE_DEVICES={actual_gpu_id}")
        
        torch.cuda.set_device(0)  # 设置可见设备后，物理GPU对应逻辑GPU 0
        device = torch.device('cuda:0')
        torch.cuda.set_per_process_memory_fraction(0.85, 0)
        print(f"🚀 [{model_name}] 使用GPU: {actual_gpu_id} ({torch.cuda.get_device_name(0)})")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device('cpu')
        print(f"⚠️ [{model_name}] 没有可用GPU，使用CPU")

    try:
        model = model_cls(**model_kwargs).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"模型参数量 [{model_name}]: {n_params:.1f}M (可训练 {n_trainable:.1f}M)")

        num_classes = config['model']['num_classes']
        num_nodes = config['model']['num_nodes']
        num_workers = config['training'].get('num_workers', 2)
        pin_memory = config['training'].get('pin_memory', False)
        dataset_type = detect_dataset_type(config)

        if data_type == 'image':
            train_cfg = dict(config['training'])
            train_paths, val_paths, test_paths, label_mapping, transform = data_args
        else:
            train_cfg = dict(config.get('training_1d', config['training']))
            train_1d_path, val_1d_path, test_1d_path, label_mapping = data_args

        model_overrides = config.get('model_configs', {}).get(model_name, {})
        if model_overrides:
            print(f"  ⚙️ 使用 [{model_name}] 专属配置: {model_overrides}")
            train_cfg.update(model_overrides)

        mixed_precision = train_cfg.get('mixed_precision', config['training'].get('mixed_precision', True))

        if data_type == 'image':
            train_loader, val_loader, test_loader = build_image_loaders(
                train_paths, val_paths, test_paths, label_mapping, transform, 
                train_cfg['batch_size'], dataset_type, num_nodes, num_workers, pin_memory)
        else:
            train_loader, val_loader, test_loader = build_1d_loaders(
                train_1d_path, val_1d_path, test_1d_path, label_mapping, 
                train_cfg['batch_size'], dataset_type, num_nodes, num_workers, pin_memory)

        train_labels = train_loader.dataset.get_labels()
        class_counts = Counter(train_labels)
        total = sum(class_counts.values())
        # Softened class weights (power 0.5) to avoid over-predicting minority class
        raw_weights = torch.tensor(
            [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)],
            dtype=torch.float
        )
        class_weights = torch.pow(raw_weights, 0.5).to(device)
        print(f"  类权重 (软化后): {class_weights.cpu().numpy()}")

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
            optimizer_type=train_cfg.get('optimizer_type', 'sgd'),
            early_stop_monitor=train_cfg.get('early_stop_monitor', 'val_acc'),
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
        # 正确清理变量以释放显存
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
    dataset_type = detect_dataset_type(config)
    num_classes = config['model']['num_classes']
    dataset_label = "CHB-MIT" if dataset_type == 'chbmit' else "Standard"
    
    print(f"\n{'='*60}")
    print(f"📊 数据集类型: {dataset_label} (dataset_type={dataset_type})")
    print(f"   类别数: {num_classes}, 通道数: {config['model']['num_nodes']}")
    print(f"   标签映射: {config.get('label_mapping', {})}")
    print(f"{'='*60}\n")

    num_gpus = torch.cuda.device_count()
    print(f"📦 torch.cuda.device_count() 返回: {num_gpus}")
    
    available_gpus = []
    for i in range(num_gpus):
        try:
            torch.cuda.get_device_name(i)
            available_gpus.append(i)
        except Exception as e:
            print(f"⚠️  GPU {i} 不可用: {e}")
    
    num_gpus = len(available_gpus)
    print(f"✅ 实际可用GPU数量: {num_gpus}")

    if num_gpus == 0:
        print("❌ 没有可用GPU，使用CPU")
    else:
        print(f"✅ GPU列表: {[torch.cuda.get_device_name(i) for i in available_gpus]}")
        for i in available_gpus:
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
    num_nodes = config['model']['num_nodes']
    image_size = config['model']['image_size']

    # Auto-detect n_times from data for 1D models
    n_times = config['model'].get('n_times', None)
    if n_times is None:
        try:
            import numpy as _np
            with open(train_1d_path, 'r') as f:
                _first_line = f.readline().strip()
            _sample_data = _np.load(_first_line)
            if _sample_data.shape[0] == num_nodes:
                n_times = _sample_data.shape[1]
            else:
                n_times = _sample_data.shape[0]
            print(f"✅ 自动检测 1D 数据时间点数: n_times={n_times}")
        except Exception as e:
            n_times = 2000
            print(f"⚠️ 无法自动检测 n_times，使用默认值 2000: {e}")
    gradient_checkpointing = config['model'].get('gradient_checkpointing', False)
    label_mapping = config['label_mapping']

    class_label = f"{num_classes}Class"
    run_dir_name = (datetime.now().strftime("%Y%m%d_%H%M%S")
                    + f"_ASTNet_{dataset_label}_{class_label}")
    run_save_dir = os.path.join(model_save_dir, run_dir_name)
    os.makedirs(run_save_dir, exist_ok=True)

    print(f"结果保存目录: {run_save_dir}")

    config['optimizer'] = {
        'type': 'SGD',
        'momentum': 0.9,
        'scheduler': 'CosineAnnealingLR',
    }

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
        ('AST-Net_GCN', ResNeXt_GCN_Bridge, {'num_classes': num_classes, 'num_nodes': num_nodes, 'backbone_name': 'resnext50_32x4d', 'gradient_checkpointing': gradient_checkpointing, 'use_anatomical_prior': True},
         'image', image_data_args),
        ('Baseline_MultiScaleResNeXt', BaselineMultiScaleResNeXt, {'num_classes': num_classes, 'num_nodes': num_nodes, 'backbone_name': 'resnext50_32x4d', 'gradient_checkpointing': gradient_checkpointing},
         'image', image_data_args),
        ('Baseline_ResNeXt_resnext50_32x4d', BaselineResNeXt, {'num_classes': num_classes, 'num_nodes': num_nodes, 'backbone_name': 'resnext50_32x4d', 'gradient_checkpointing': gradient_checkpointing},
         'image', image_data_args),
        ('Baseline_ResNet_resnet18', BaselineResNet, {'num_classes': num_classes, 'num_nodes': num_nodes, 'backbone_name': 'resnet18', 'gradient_checkpointing': gradient_checkpointing},
         'image', image_data_args),
        ('Baseline_EffNet_efficientnet_b0', BaselineEfficientNet, {'num_classes': num_classes, 'num_nodes': num_nodes, 'backbone_name': 'efficientnet_b0', 'gradient_checkpointing': gradient_checkpointing},
         'image', image_data_args),
        ('Baseline_CNNLSTM_resnet18', BaselineCNNLSTM, {'num_classes': num_classes, 'num_nodes': num_nodes, 'backbone_name': 'resnet18', 'gradient_checkpointing': gradient_checkpointing},
         'image', image_data_args),
        # 1D models
        ('EEGNet', EEGNet, {'num_classes': num_classes, 'num_channels': num_nodes},
         '1d', one_d_data_args),
        ('ST-GCN_EEG', ST_GCN_EEG, {'num_classes': num_classes, 'num_nodes': num_nodes},
         '1d', one_d_data_args),
        ('EEGConformer', BraindecodeEEGConformer, {'num_classes': num_classes, 'num_channels': num_nodes, 'n_times': n_times},
         '1d', one_d_data_args),
    ]

    if model_filter:
        available_names = [t[0] for t in all_tasks]
        if isinstance(model_filter, list):
            all_tasks = [t for t in all_tasks if t[0] in model_filter]
            print(f"\n🎯 多模型模式，请求: {', '.join(model_filter)}")
        else:
            all_tasks = [t for t in all_tasks if t[0] == model_filter]
            print(f"\n🎯 单模型模式，请求: {model_filter}")
        
        if not all_tasks:
            print(f"⚠️ 没有匹配的模型！可用模型:")
            for name in available_names:
                print(f"     - {name}")
            return
    
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
        import time
        from collections import deque

        print(f"\n🚀 动态GPU调度模式 ({num_gpus} GPUs)")
        print(f"   优先模型: AST-Net_GCN, Baseline_MultiScaleResNeXt")
        print(f"   总模型数: {len(all_tasks)}, 可用GPU: {num_gpus}")

        # Sort tasks: priority models first, then the rest
        priority_models = {'AST-Net_GCN', 'Baseline_MultiScaleResNeXt'}
        sorted_tasks = sorted(all_tasks, key=lambda t: 0 if t[0] in priority_models else 1)

        task_queue = deque(sorted_tasks)
        all_results = []
        mp.set_start_method('spawn', force=True)
        results_queue = mp.Queue()

        # Track running processes: {process: (gpu_id, model_name)}
        running = {}
        gpu_available = list(range(num_gpus))

        print(f"\n{'='*60}")
        print(f"🚀 开始动态调度，共 {len(sorted_tasks)} 个模型，{num_gpus} 张GPU")
        print(f"{'='*60}")

        while task_queue or running:
            # Start new tasks on available GPUs
            while task_queue and gpu_available:
                task = task_queue.popleft()
                model_name, model_cls, model_kwargs, data_type, data_args = task
                gpu_id = gpu_available.pop(0)

                print(f"\n📦 准备训练 [{model_name}]，分配到 GPU {gpu_id}")
                print(f"   模型参数: {model_kwargs}")

                args = (results_queue, model_name, model_cls, model_kwargs, data_type, data_args,
                        run_save_dir, gpu_id, config)

                p = mp.Process(target=train_single_model_worker, args=(args,))
                p.start()
                running[p] = (gpu_id, model_name)

                remaining = len(task_queue)
                if remaining > 0:
                    print(f"   队列中还有 {remaining} 个模型等待调度")

            # Check for finished processes
            if running:
                time.sleep(2)  # Small sleep to avoid busy waiting
                finished = []
                for p, (gpu_id, model_name) in list(running.items()):
                    if not p.is_alive():
                        p.join()
                        finished.append(p)
                        gpu_available.append(gpu_id)
                        print(f"\n  ✅ [{model_name}] 训练完成，GPU {gpu_id} 已释放")

                for p in finished:
                    del running[p]

        # Collect all results
        while not results_queue.empty():
            result = results_queue.get()
            all_results.append(result)
            if 'error' in result:
                print(f"⚠️ [{result['model']}] 训练失败: {result['error']}")

        torch.cuda.empty_cache()

    if not all_results:
        print("\n⚠️ 没有任何模型训练成功，跳过结果汇总。")
        print("   请检查模型名称是否正确，可用模型名称:")
        for task in all_tasks:
            print(f"     - {task[0]}")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('f1', ascending=False).reset_index(drop=True)

    print("\n" + "="*70)
    print(f"🏆 {dataset_label} 模型性能对比汇总 (按 Macro F1 降序)")
    print("="*70)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    results_df.to_csv(os.path.join(run_save_dir, 'model_comparison_results.csv'), index=False)
    plot_comparison(results_df, run_save_dir)

    print(f"\n✅ 所有实验完成。结果保存于: {run_save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AST-Net Training Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                        help='Specify a single model to train (e.g., AST-Net_GCN_resnet18)')
    parser.add_argument('--models', type=str, default=None,
                        help='Specify multiple models to train, separated by comma (e.g., AST-Net_GCN_resnet18,EEGNet,ST-GCN_EEG)')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 确定模型过滤条件
    model_filter = None
    if args.models:
        model_filter = [m.strip() for m in args.models.split(',')]
    elif args.model:
        model_filter = args.model
    
    main(config, model_filter=model_filter)