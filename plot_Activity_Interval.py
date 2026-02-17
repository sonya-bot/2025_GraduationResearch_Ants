# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import calculate_thresholds

# データ入力・前処理関数群

def data_input(velocity_csv_path):
    """
    入力データの読み込み、FPS定義、時間軸の計算、平滑化を行う。
    """
    try:
        df_vel = pd.read_csv(velocity_csv_path)
        # print(f"データ読み込み成功: '{os.path.basename(velocity_csv_path)}'")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {velocity_csv_path}")
        return None, None, None, None, None, None

    output_dir = os.path.dirname(velocity_csv_path)

    # FPSの定義
    FPS = 2.0  # 1フレーム=1/2秒
    total_frames = len(df_vel)
    frames = df_vel['position']
        
    # 時間計算
    time_minutes = frames / FPS / 60
    
    # 個体IDの特定
    speed_cols = [col for col in df_vel.columns if col.startswith('speed_')]
    individual_ids = [col.replace('speed_', '') for col in speed_cols]

    # 平滑化処理 (60秒間の移動平均)
    smoothing_window = int(60 * FPS)
    if smoothing_window > 1:
        for col in speed_cols:
            df_vel[col] = pd.to_numeric(df_vel[col], errors='coerce').fillna(0)
            df_vel[col] = df_vel[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()
            df_vel[col] = df_vel[col].fillna(0)

    return df_vel, speed_cols, output_dir, time_minutes, individual_ids, FPS

def calculate_speed(df_vel, individual_ids, remove_outliers):
    """
    速度データの数値変換と外れ値処理を行う。
    """
    speed_cols = [f'speed_{uid}' for uid in individual_ids]
    
    for col in speed_cols:
        df_vel[col] = pd.to_numeric(df_vel[col], errors='coerce')
        df_vel[col].fillna(0, inplace=True)
    
    if remove_outliers:
        predefined_threshold = 200.0
        for col in speed_cols:
            df_vel.loc[df_vel[col] > predefined_threshold, col] = 0
    
    return df_vel

def calculate_states(velocity_csv_path, remove_outliers, velocity_threshold_key):
    """
    データ読み込みから状態(0/1)の判定までを行う。
    """
    df_vel, speed_cols, output_dir, time_minutes, individual_ids, FPS = data_input(velocity_csv_path)
    if df_vel is None:
        return None, None, None, None, None

    df_vel = calculate_speed(df_vel, individual_ids, remove_outliers)

    # 閾値取得 (calculate_thresholdsモジュールを使用)
    _, individual_results_list = calculate_thresholds.get_threshold_values(velocity_csv_path)
    
    if individual_results_list is None:
        print("閾値の計算に失敗しました。")
        return None, None, None, None, None

    thresh_map = {res['id']: res for res in individual_results_list}

    activity_states = {}
    for uid in individual_ids:
        ind_res = thresh_map.get(uid)
        speed_col = f'speed_{uid}'
        
        th = 5.0 # デフォルト
        if ind_res and velocity_threshold_key in ind_res:
            th = ind_res.get(velocity_threshold_key)
        else:
            print(f"警告: ID {uid} の閾値が見つかりません。デフォルト値 {th} を使用します。")

        # 閾値を超えたら1、それ以外は0
        activity_states[uid] = (df_vel[speed_col] > th).astype(int)
            
    return activity_states, output_dir, time_minutes, individual_ids, FPS

def calculate_interval_data(velocity_csv_path, display_name, remove_outliers, velocity_threshold_key):
    """
    個体ごとのインターバルデータ（分）を計算してリストで返す。
    """
    activity_states, output_dir, _, individual_ids, FPS = calculate_states(velocity_csv_path, remove_outliers, velocity_threshold_key)
    
    if activity_states is None:
        return None, None

    colony_name = display_name 
    results = []

    for uid in individual_ids:
        states = activity_states[uid].values

        # 状態が 0 -> 1 に変わる瞬間（活動開始）を探す
        diffs = np.diff(states, prepend=0)
        onset_indices = np.where(diffs == 1)[0]
        
        if len(onset_indices) < 2:
            continue

        # インターバル計算 (分単位)
        interval_frames = np.diff(onset_indices)
        interval_minutes = interval_frames / FPS / 60
        
        results.append({
            'colony': colony_name,
            'id': uid,
            'intervals': interval_minutes,
            'output_dir': output_dir
        })
        
    return results, output_dir

# グラフ描画関数

def plot_intervals_analysis(target_colonies_dict, mode_label, base_path, remove_outliers, velocity_threshold_key, fig_size, save_path, auto_save):
    """
    1つの条件（Isolated/Paired/Trio）について解析し、グラフを作成する。
    """
    all_data = []
    print(f"\n解析開始: {mode_label} ({len(target_colonies_dict)} コロニー)")

    # --- 1. データ収集 ---
    for display_name, folder_id in target_colonies_dict.items():
        # ファイルパスの構築
        input_csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position_velocity.csv")
        
        # print(f"処理中: {display_name} ({folder_id}) ...")
        colony_results, output_dir = calculate_interval_data(input_csv_path, display_name, remove_outliers, velocity_threshold_key)
        
        if colony_results:
            all_data.extend(colony_results)

    if not all_data:
        print("有効なデータが見つかりませんでした。")
        return

    # --- 2. 重ね合わせグラフ (コロニー平均 + 全体平均) の作成 ---
    print(f"グラフ作成中: {mode_label}")
    plot_combined_graph_by_colony(all_data, mode_label, fig_size, save_path, auto_save)


def plot_combined_graph_by_colony(all_data, mode_label, fig_size, save_path, auto_save):
    """
    コロニーごとに全個体の平均をプロットし、さらに全体の平均を黒線で表示する。
    """
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    
    # ビンの設定 (0, 2, 4, ... 30)
    bins = np.arange(0, 32, 2)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # コロニーごとのデータ整理
    # colony_data[colony_name] = [prob_dist_ind1, prob_dist_ind2, ...]
    colony_data_map = {}
    unique_colonies = sorted(list(set(d['colony'] for d in all_data)))
    
    # カラーマップ設定 (コロニーごとに色を固定)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_colonies)))
    colony_color_map = {col: color for col, color in zip(unique_colonies, colors)}

    # 1. 個体ごとの確率分布を計算し、コロニーごとにまとめる
    for res in all_data:
        intervals = res['intervals']
        colony = res['colony']
        
        # ヒストグラム計算
        hist_vals, _ = np.histogram(intervals, bins=bins, density=False)
        if np.sum(hist_vals) > 0:
            prob = hist_vals / np.sum(hist_vals)
            if colony not in colony_data_map:
                colony_data_map[colony] = []
            colony_data_map[colony].append(prob)

    # 2. コロニーごとの平均分布を計算してプロット
    all_colony_avgs = [] # 全体平均計算用

    for colony in unique_colonies:
        if colony not in colony_data_map:
            continue
            
        probs_list = colony_data_map[colony]
        # そのコロニー内の個体の平均分布を計算
        colony_avg_prob = np.mean(probs_list, axis=0)
        all_colony_avgs.append(colony_avg_prob)
        
        # プロット (コロニーの色)
        ax.plot(bin_centers, colony_avg_prob, 
                marker='o', markersize=3, 
                linestyle='-', linewidth=1.5, alpha=0.3,
                color=colony_color_map[colony], label=colony)

    # 3. 全体平均 (全コロニーの平均) を計算してプロット
    if all_colony_avgs:
        grand_avg_prob = np.mean(all_colony_avgs, axis=0)
        
        # 太い黒線で平均をプロット
        ax.plot(bin_centers, grand_avg_prob, 
                marker='o', markersize=4, 
                linestyle='-', linewidth=3.0, color='black', 
                label='Average', zorder=10)

    # グラフの体裁
    ax.set_title(f'Activity Interval({mode_label})', fontsize=20)
    ax.set_xlabel('Interval Time [min]', fontsize=20)
    ax.set_ylabel('Probability [%]', fontsize=20)
    ax.set_xlim(0, 30)
    ax.set_ylim(0,0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 凡例
    ax.legend(loc='upper right', title="Colony", fontsize=12, ncol=2)

    # 保存または表示
    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")
        plt.close()
    else:
        plt.show()


# メイン実行ブロック

if __name__ == '__main__':

    # 1. データ定義
    ISO_DICT = {
        "Colony A": "20251030_01", "Colony B": "20251104_02", "Colony C": "20251106_01",
        "Colony D": "20251113_01", "Colony E": "20251117_02", "Colony G": "20251119_01",
        "Colony H": "20251120_02", "Colony I": "20251127_01",
    }
    PAIR_DICT = {
        "Colony A": "20251030_02", "Colony B": "20251105_01", "Colony C": "20251107_01",
        "Colony D": "20251113_03", "Colony E": "20251118_01", "Colony G": "20251119_02",
        "Colony H": "20251121_01", "Colony I": "20251127_02",
    }
    TRIO_DICT = {
        "Colony A": "20251101_01", "Colony B": "20251105_02", "Colony C": "20251110_01",
        "Colony D": "20251117_01", "Colony E": "20251118_02", "Colony G": "20251120_01",
        "Colony H": "20251126_01", "Colony I": "20251128_01",
    }   
    
    # パス設定 (環境に合わせて変更してください)
    BASE_PATH = "/Users/sonya/卒論データ/analysis_data"
    OUTPUT_DIR = "/Users/sonya/卒論データ/analysis_data/graph_output"

    # パラメータ設定
    REMOVE_OUTLIERS = True
    VELOCITY_THRESHOLD_KEY = "avg_half" 
    FIG_SIZE = (6, 4)

    # 保存設定 (Trueならファイル保存、Falseなら画面表示)
    AUTO_SAVE = False

    if AUTO_SAVE and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    DATA_SETS = [
        (ISO_DICT, "Isolate", "Isolate"),
        (PAIR_DICT, "Pair", "Pair"),
        (TRIO_DICT, "Trio", "Trio")
    ]

    # 4. 実行ループ
    for source_dict, mode_label, file_suffix in DATA_SETS:
        process_dict = source_dict
        
        # 保存ファイル名
        save_name = f"Activity_Interval_{file_suffix}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)

        # 解析実行
        plot_intervals_analysis(
            process_dict, 
            mode_label, 
            BASE_PATH, 
            REMOVE_OUTLIERS, 
            VELOCITY_THRESHOLD_KEY, 
            FIG_SIZE, 
            save_path, 
            AUTO_SAVE
        )

    print("\nAll processes completed.")