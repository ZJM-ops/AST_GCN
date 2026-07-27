import argparse
import os
import yaml
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import mne


def main(config):
    fif_root_dir = config['data']['preprocessed_dir']
    output_root_dir = config['data']['segmented_dir']
    labels_csv = config['data']['labels_csv']
    window_length_sec = config['preprocessing']['window_length_sec']
    overlap_ratio = config['preprocessing']['overlap_ratio']

    label_mapping = {k: k for k in config['label_mapping'].keys()}

    os.makedirs(output_root_dir, exist_ok=True)

    try:
        labels_df = pd.read_csv(labels_csv)
    except FileNotFoundError:
        print(f"错误: CSV文件未在 {labels_csv} 找到。")
        return

    original_labels_to_process = list(label_mapping.keys())
    relevant_records = labels_df[labels_df['label'].isin(original_labels_to_process)]

    matched_metadata = []
    print("正在检查CSV记录与文件系统的一致性...")

    for _, row in tqdm(relevant_records.iterrows(), total=len(relevant_records), desc="文件检查进度"):
        animal_id = str(row['animal_id']).strip()
        group = str(row['group']).strip()
        label = str(row['label']).strip()
        original_filename_base = os.path.basename(row['filepath']).replace('.edf', '')
        expected_fif_name = f"{animal_id}_{original_filename_base}-raw.fif".replace(' ', '_')
        expected_fif_path = os.path.join(fif_root_dir, group, label, expected_fif_name)

        if os.path.exists(expected_fif_path):
            matched_metadata.append({
                'fif_filepath': expected_fif_path,
                'animal_id': animal_id,
                'group': group,
                'original_label': label
            })
        else:
            tqdm.write(f"\n[警告] 未找到匹配的 .fif 文件。预期路径: '{expected_fif_path}'")

    if not matched_metadata:
        print("错误: 未找到任何符合条件的文件。")
        return

    print(f"\n匹配成功 {len(matched_metadata)} 个文件，准备进行分段。")
    processed_count = 0
    skipped_count = 0

    for meta in tqdm(matched_metadata, desc="分段处理进度"):
        fif_filepath = meta['fif_filepath']
        animal_id = meta['animal_id']
        group = meta['group']
        original_label = meta['original_label']

        new_label = label_mapping[original_label]
        output_dir = os.path.join(output_root_dir, group, new_label)

        output_file_pattern = f"{animal_id}_{group}_{new_label}_seg*.npy"
        existing_segments = glob.glob(os.path.join(output_dir, output_file_pattern))

        if existing_segments:
            tqdm.write(f"\n  [跳过] '{os.path.basename(fif_filepath)}' 的分段文件已存在。")
            skipped_count += 1
            continue

        try:
            os.makedirs(output_dir, exist_ok=True)
            raw = mne.io.read_raw_fif(fif_filepath, preload=True, verbose=False)
            sfreq = raw.info['sfreq']

            window_samples = int(window_length_sec * sfreq)
            step_samples = int(window_samples * (1 - overlap_ratio))

            if step_samples <= 0:
                raise ValueError("步长计算错误，请检查窗口长度和重叠率。")

            num_segments = 0
            for start_sample in range(0, raw.n_times - window_samples + 1, step_samples):
                segment_data = raw.get_data(start=start_sample, stop=start_sample + window_samples)
                segment_filename = f"{animal_id}_{group}_{new_label}_seg{num_segments:04d}.npy"
                output_path = os.path.join(output_dir, segment_filename)
                np.save(output_path, segment_data)
                num_segments += 1

            processed_count += 1
            tqdm.write(f"\n  [成功] 文件 '{os.path.basename(fif_filepath)}' 已切分并保存 {num_segments} 个数据段到 '{output_dir}'")

        except Exception as e:
            tqdm.write(f"\n  [错误] 处理文件 '{fif_filepath}' 失败：{e}")

    print("\n--- 分段处理完成 ---")
    print(f"总文件数 (来自labels.csv): {len(relevant_records)}")
    print(f"本次新处理并分段: {processed_count} 个文件")
    print(f"因已存在而被跳过: {skipped_count} 个文件")
    print(f"总计成功（新处理+已存在）: {processed_count + skipped_count} 个文件")

    errors_count = len(matched_metadata) - processed_count - skipped_count
    if errors_count > 0:
        print(f"因错误处理失败: {errors_count} 个文件")
        print("\n请检查上面打印的错误信息，以定位问题文件。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG Data Segmentation')
    parser.add_argument('--config', type=str, default='../config/config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)