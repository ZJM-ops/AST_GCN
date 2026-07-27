import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MultiChannelDataset(Dataset):
    def __init__(self, file_paths, transform=None, single_channel_index=None, label_mapping=None):
        self.segs = {}
        for p in file_paths:
            parts = p.split(os.sep)
            label_folder = parts[-2]
            filename = parts[-1]

            seg_id_parts = filename.split('_seg')
            if len(seg_id_parts) > 1:
                base_name = '_seg'.join(seg_id_parts[:-1])
                seg_num = seg_id_parts[-1].split('_')[0]
                seg_id = f"{label_folder}_{base_name}_seg{seg_num}"
            else:
                seg_id = filename.split('_ch')[0]

            if seg_id not in self.segs:
                self.segs[seg_id] = []
            self.segs[seg_id].append(p)

        self.seg_ids = [k for k, v in self.segs.items() if len(v) == 8]
        for k in self.seg_ids:
            self.segs[k] = sorted(self.segs[k], key=lambda x: int(x.split("ch")[-1].split(".")[0]))

        self.transform = transform
        self.label_mapping = label_mapping or {'Baseline': 0, 'Ictal': 1, 'Chronic': 2}
        self.single_channel_index = single_channel_index

    def __len__(self):
        return len(self.seg_ids)

    def __getitem__(self, idx):
        seg_id = self.seg_ids[idx]
        img_paths = self.segs[seg_id]

        if len(img_paths) != 8:
            raise ValueError(f"Sample {seg_id} does not have 8 channels, found {len(img_paths)}")

        if self.single_channel_index is not None:
            img_path = img_paths[self.single_channel_index - 1]
            imgs = Image.open(img_path).convert("RGB")
            if self.transform:
                imgs = self.transform(imgs)
        else:
            imgs = [Image.open(p).convert("RGB") for p in img_paths]
            if self.transform:
                imgs = [self.transform(img) for img in imgs]
            imgs = torch.cat(imgs, dim=0)

        label_folder = img_paths[0].split(os.sep)[-2]
        label = self.label_mapping.get(label_folder, 0)
        return imgs, label


class CompleteGraphDataset(Dataset):
    def __init__(self, file_paths, transform=None, label_mapping=None):
        self.segs = {}
        for p in file_paths:
            parts = p.split(os.sep)
            label_folder = parts[-2]
            filename = parts[-1]
            seg_id_parts = filename.split('_seg')
            if len(seg_id_parts) > 1:
                base_name = '_seg'.join(seg_id_parts[:-1])
                seg_num = seg_id_parts[-1].split('_')[0]
                seg_id = f"{label_folder}_{base_name}_seg{seg_num}"
            else:
                seg_id = filename.split('_ch')[0]
            if seg_id not in self.segs:
                self.segs[seg_id] = []
            self.segs[seg_id].append(p)

        self.seg_ids = [k for k, v in self.segs.items() if len(v) == 8]
        for k in self.seg_ids:
            self.segs[k] = sorted(self.segs[k], key=lambda x: int(x.split("ch")[-1].split(".")[0]))

        self.transform = transform
        self.label_mapping = label_mapping or {'Baseline': 0, 'Ictal': 1, 'Chronic': 2}

    def __len__(self):
        return len(self.seg_ids)

    def get_labels(self):
        labels = []
        for seg_id in self.seg_ids:
            label_folder = self.segs[seg_id][0].split(os.sep)[-2]
            labels.append(self.label_mapping.get(label_folder, 0))
        return labels

    def __getitem__(self, idx):
        seg_id = self.seg_ids[idx]
        img_paths = self.segs[seg_id]
        imgs = [Image.open(p).convert("RGB") for p in img_paths]
        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)
        label_folder = img_paths[0].split(os.sep)[-2]
        return imgs, self.label_mapping.get(label_folder, 0)


class EEG_FullChannel_Dataset(Dataset):
    def __init__(self, file_paths, fs=2000.0, apply_filter=True, num_channels=8, label_mapping=None):
        self.file_paths = file_paths
        self.label_mapping = label_mapping or {'Baseline': 0, 'Ictal': 1}
        self.fs = fs
        self.apply_filter = apply_filter
        self.num_channels = num_channels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = np.load(path)
        data = data[:self.num_channels, :]

        if self.apply_filter:
            from scipy.signal import butter, filtfilt
            nyq = 0.5 * self.fs
            lowcut, highcut = 250.0, 490.0
            if highcut >= nyq:
                lowcut, highcut = 1.0, min(40.0, nyq - 1.0)
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(4, [low, high], btype='band')
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered_data[ch, :] = filtfilt(b, a, data[ch, :])
            data = filtered_data

        for ch in range(data.shape[0]):
            ch_std = np.std(data[ch, :])
            if ch_std > 1e-8:
                data[ch, :] = (data[ch, :] - np.mean(data[ch, :])) / ch_std
            else:
                data[ch, :] = data[ch, :] - np.mean(data[ch, :])

        tensor_data = torch.from_numpy(data.copy()).float()
        label_str = path.split(os.sep)[-2]
        label = self.label_mapping.get(label_str, 0)
        return tensor_data, label


class Complete1DGraphDataset(Dataset):
    def __init__(self, txt_path, global_mean=None, global_std=None, training=False, label_mapping=None):
        with open(txt_path, 'r') as f:
            self.npy_paths = [line.strip() for line in f if line.strip()]
        self.label_mapping = label_mapping or {'Baseline': 0, 'Ictal': 1, 'Chronic-28d': 2}
        self.global_mean = global_mean
        self.global_std = global_std
        self.training = training

    def __len__(self):
        return len(self.npy_paths)

    def get_labels(self):
        labels = []
        for path in self.npy_paths:
            path_parts = path.split(os.sep)
            label_folder = path_parts[-2]
            labels.append(self.label_mapping.get(label_folder, 0))
        return labels

    def __getitem__(self, idx):
        path = self.npy_paths[idx]
        signal = np.load(path).astype(np.float32)
        signal = np.nan_to_num(signal)

        if signal.shape[0] != 8 and signal.shape[1] == 8:
            signal = signal.T
        if signal.shape[0] != 8:
            raise ValueError(f"通道不匹配：{path} shape={signal.shape}")

        if self.global_mean is not None and self.global_std is not None:
            signal = (signal - self.global_mean) / (self.global_std + 1e-6)

        signal_tensor = torch.tensor(signal)
        if self.training:
            signal_tensor += torch.randn_like(signal_tensor) * 0.015

        path_parts = path.split(os.sep)
        label_folder = path_parts[-2]
        label = self.label_mapping.get(label_folder, 0)
        return signal_tensor, label