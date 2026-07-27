import argparse
import os
import yaml
import pandas as pd
import mne


def main(config):
    labels_csv = config['data']['labels_csv']
    output_root_dir = config['data']['preprocessed_dir']
    l_freq = config['preprocessing']['l_freq']
    h_freq = config['preprocessing']['h_freq']
    notch_freqs = config['preprocessing']['notch_freqs']

    channels_with_prefix = config['channels']['with_prefix']
    channels_without_prefix = config['channels']['without_prefix']

    try:
        labels_df = pd.read_csv(labels_csv)
    except FileNotFoundError:
        print(f"错误：labels.csv 文件未在 {labels_csv} 找到。")
        return

    required_cols = ['filepath', 'animal_id', 'group', 'label']
    if not all(col in labels_df.columns for col in required_cols):
        print(f"错误：CSV文件必须包含以下所有列: {required_cols}")
        return

    os.makedirs(output_root_dir, exist_ok=True)
    failed_files = []

    print(f"\n开始预处理 {len(labels_df)} 个文件...")
    for index, row in labels_df.iterrows():
        filepath = row['filepath']
        animal_id = row['animal_id']
        group = row['group']
        label = row['label']
        original_filename = os.path.basename(filepath)
        print(f"--- 正在处理: {group}/{label}/{original_filename} ---")

        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError("文件路径不存在。")

            target_dir = os.path.join(output_root_dir, group, label)
            os.makedirs(target_dir, exist_ok=True)

            new_filename = f"{animal_id}_{original_filename}".replace('.edf', '-raw.fif').replace(' ', '_')
            output_filepath = os.path.join(target_dir, new_filename)

            if os.path.exists(output_filepath):
                print("  [跳过] 已存在处理过的文件。")
                continue

            raw_info = mne.io.read_raw_edf(filepath, preload=False, verbose=False)

            if channels_with_prefix[0] in raw_info.ch_names:
                channels_to_use = channels_with_prefix
                print("  [信息] 检测到带 'EEG ' 前缀的通道名格式。")
            else:
                channels_to_use = channels_without_prefix
                print("  [信息] 检测到不带 'EEG ' 前缀的通道名格式。")

            raw = raw_info.load_data(verbose=False)
            current_channels_to_keep = [ch for ch in channels_to_use if ch in raw.ch_names]

            if not current_channels_to_keep:
                raise ValueError("根据检测到的格式，仍然在文件中找不到任何指定的EEG通道。")

            raw.pick_channels(current_channels_to_keep)
            raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
            raw.notch_filter(freqs=notch_freqs, verbose=False)
            raw.save(output_filepath, overwrite=True)
            print(f"  [成功] 已保存至 -> {output_filepath}")

        except Exception as e:
            error_message = str(e).replace('\n', ' ').strip()
            print(f"  [错误] {error_message}")
            failed_files.append({
                'filepath': filepath,
                'animal_id': animal_id,
                'group': group,
                'label': label,
                'reason': error_message
            })

    print("\n\n--- 所有文件预处理和保存操作完成！ ---")
    if failed_files:
        print("\n以下文件在处理过程中发生错误：")
        for i, fail_row in enumerate(failed_files):
            print(f"  - 文件: {fail_row['filepath']} (Animal ID: {fail_row['animal_id']})")
            print(f"    原因: {fail_row['reason']}")
    else:
        print("\n所有文件均已成功处理！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG Data Preprocessing')
    parser.add_argument('--config', type=str, default='../config/config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)