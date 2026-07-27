import argparse
import os
import yaml
import json
import numpy as np
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import concurrent.futures
import glob


def process_single_file(args):
    filepath, global_max_energy, label_mapping, group_mapping, output_root_dir, wavelet_name, scales, sampling_period, duration = args
    try:
        path_parts = filepath.split(os.sep)
        group_chinese = path_parts[-3]
        label_chinese = path_parts[-2]
        group_english = group_mapping.get(group_chinese, group_chinese)
        label_english = label_mapping.get(label_chinese, label_chinese)
        base_filename = os.path.basename(filepath).replace('.npy', '')
        seg_index_start = base_filename.rfind('_seg')

        if seg_index_start == -1:
            return f"错误: 文件名格式不正确，无法找到 '_seg'：{os.path.basename(filepath)}"

        animal_id_part = base_filename[:seg_index_start]
        segment_index = base_filename[seg_index_start+1:]
        parts_to_keep = [
            p for p in animal_id_part.split('_')
            if p not in group_mapping and p not in label_mapping
        ]
        final_animal_id = '_'.join(parts_to_keep)
        new_base_filename = f"{group_english}_{label_english}_{final_animal_id}_{segment_index}"

        segment_data = np.load(filepath)
        processed_count_for_file = 0
        skipped_count_for_file = 0

        for i, channel_data in enumerate(segment_data):
            output_dir = os.path.join(output_root_dir, group_english, label_english)
            output_path = os.path.join(output_dir, f"{new_base_filename}_ch{i+1}.png")

            if os.path.exists(output_path):
                skipped_count_for_file += 1
                continue

            coefficients, _ = pywt.cwt(channel_data, scales, wavelet_name, sampling_period=sampling_period)
            fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
            plt.imshow(np.abs(coefficients), cmap='jet', aspect='auto',
                       extent=(0.0, duration, f_min, f_max),
                       vmin=0, vmax=global_max_energy)
            plt.axis('off')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            processed_count_for_file += 1

        if processed_count_for_file > 0:
            return "processed"
        elif skipped_count_for_file > 0:
            return "skipped"
        else:
            return f"错误: 处理文件 {filepath} 失败，可能数据无效或所有文件已存在。"

    except Exception as e:
        return f"错误: 处理文件 {filepath} 失败：{e}"


def calculate_file_max(args):
    filepath, wavelet_name, scales, sampling_period = args
    try:
        segment_data = np.load(filepath)
        coefficients, _ = pywt.cwt(segment_data, scales, wavelet_name, sampling_period=sampling_period)
        file_max = np.max(np.abs(coefficients))
        return file_max
    except Exception as e:
        print(f"警告：计算文件 {filepath} 最大值时出错：{e}")
        return 0.0


def main(config):
    global f_min, f_max

    segmented_data_root = config['data']['segmented_dir']
    output_root_dir = config['data']['cwt_images_dir']

    wavelet_name = config['cwt']['wavelet']
    f_min = config['cwt']['f_min']
    f_max = config['cwt']['f_max']
    num_scales = config['cwt']['num_scales']
    sampling_rate = config['cwt']['sampling_rate']
    duration = config['cwt']['duration']
    sampling_period = 1.0 / sampling_rate

    group_mapping = config['group_mapping']
    label_mapping = {k: k for k in config['label_mapping'].keys()}

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
        print(f"已创建输出根目录: {output_root_dir}")

    npy_files = glob.glob(os.path.join(segmented_data_root, '**', '*.npy'), recursive=True)
    if not npy_files:
        print(f"错误: 未在 {segmented_data_root} 找到任何 .npy 文件。")
        return

    print(f"找到 {len(npy_files)} 个 .npy 分段文件。")

    central_frequency = pywt.central_frequency(wavelet_name)
    scale_min = (central_frequency * sampling_rate) / f_max
    scale_max = (central_frequency * sampling_rate) / f_min
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num=num_scales)

    num_cores = cpu_count()
    max_value_file = os.path.join(output_root_dir, "global_max_energy.json")
    global_max_energy = 0.0

    if os.path.exists(max_value_file):
        try:
            with open(max_value_file, 'r') as f:
                max_value_info = json.load(f)
                global_max_energy = max_value_info.get("global_max_energy", 0.0)

            if global_max_energy > 0:
                print(f"\n成功从 {max_value_file} 读取已保存的全局最大能量值: {global_max_energy:.4f}")
            else:
                print(f"警告: 从 {max_value_file} 读取的最大值为0或无效，将重新计算。")
                global_max_energy = 0.0

        except (json.JSONDecodeError, KeyError) as e:
            print(f"警告: 读取 {max_value_file} 文件失败 ({e})，将重新计算。")
            global_max_energy = 0.0

    if global_max_energy == 0.0:
        print("\n第一步：未找到或无法读取有效的最大值文件，开始计算所有文件的全局最大能量值...")

        calc_args = [(fp, wavelet_name, scales, sampling_period) for fp in npy_files]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            future_to_file = {executor.submit(calculate_file_max, args): fp for args in calc_args}

            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(npy_files), desc="计算全局最大能量值"):
                file_max = future.result()
                if file_max > global_max_energy:
                    global_max_energy = file_max

        max_value_info = {
            "global_max_energy": global_max_energy,
            "global_max_energy_formatted": f"{global_max_energy:.4f}"
        }

        with open(max_value_file, 'w') as f:
            json.dump(max_value_info, f, indent=4)

        print(f"计算完成，所有文件的全局最大能量值为: {global_max_energy:.4f}")
        print(f"全局最大值已保存到: {max_value_file}")

    if global_max_energy == 0.0:
        print("警告：全局最大能量值为0，可能数据无效。程序中止。")
        return

    print("\n第二步：开始生成时频图，并使用全局最大值进行颜色归一化...")

    process_args = [(fp, global_max_energy, label_mapping, group_mapping,
                     output_root_dir, wavelet_name, scales, sampling_period, duration)
                    for fp in npy_files]

    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_file, process_args),
                            total=len(process_args), desc="生成时频图进度"))

    processed_count = results.count("processed")
    skipped_count = results.count("skipped")
    error_results = [r for r in results if r.startswith("错误")]
    error_count = len(error_results)

    print("\n--- 所有时频图生成完成！ ---")
    print(f"成功处理并生成新文件的数量: {processed_count}")
    print(f"因文件已存在而跳过的数量: {skipped_count}")
    if error_count > 0:
        print(f"处理出错的数量: {error_count}")
        for res in error_results:
            print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CWT Time-Frequency Images')
    parser.add_argument('--config', type=str, default='../config/config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)