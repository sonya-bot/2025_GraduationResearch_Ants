# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import LogLocator, ScalarFormatter
from scipy import stats
import calculate_thresholds
import math

# 1. データ読み込み・前処理 (共通関数)

def data_input(velocity_csv_path):
    try:
        df_vel = pd.read_csv(velocity_csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {velocity_csv_path}")
        return None, None, None, None, None, None

    output_dir = os.path.dirname(velocity_csv_path)
    FPS = 2.0
    
    speed_cols = [col for col in df_vel.columns if col.startswith('speed_')]
    individual_ids = [col.replace('speed_', '') for col in speed_cols]

    smoothing_window = int(60 * FPS)
    if smoothing_window > 1:
        for col in speed_cols:
            df_vel[col] = df_vel[col].rolling(window=smoothing_window, center=True, min_periods=1).mean().fillna(0)
    
    frames = df_vel['position']
    time_minutes = frames / FPS / 60
    return df_vel, speed_cols, output_dir, time_minutes, individual_ids, FPS

def calculate_speed(df_vel, individual_ids, remove_outliers):
    speed_cols = [f'speed_{uid}' for uid in individual_ids]
    for col in speed_cols:
        df_vel[col] = pd.to_numeric(df_vel[col], errors='coerce').fillna(0)

    if remove_outliers:
        limit = 200.0
        for col in speed_cols:
            mask = df_vel[col] > limit
            if mask.any(): df_vel.loc[mask, col] = np.nan
    return df_vel

def calculate_states(velocity_csv_path, remove_outliers, velocity_threshold_key):
    res = data_input(velocity_csv_path)
    if res is None or res[0] is None: return None
    df_vel, _, output_dir, _, individual_ids, FPS = res
    df_vel = calculate_speed(df_vel, individual_ids, remove_outliers)

    _, ind_results = calculate_thresholds.get_threshold_values(velocity_csv_path)
    if not ind_results: return None
    thresh_map = {r['id']: r[velocity_threshold_key] for r in ind_results}

    activity_states = {}
    for uid in individual_ids:
        th = thresh_map.get(uid)
        if th is not None:
            activity_states[uid] = (df_vel[f'speed_{uid}'].fillna(0) > th).astype(int)
            
    return activity_states, output_dir, individual_ids, FPS

# 2. 計算ロジック (CCDF, 回帰分析)

def calculate_ccdf_coords(durations):
    """
    持続時間リストからCCDF座標を算出。
    Y軸 = Frequency (%)
    """
    if not durations: return [], []
    sorted_durs = np.sort(durations)[::-1]
    n = len(sorted_durs)
    ranks = np.arange(1, n + 1)
    probs_percent = (ranks / n) * 100
    return sorted_durs, probs_percent

def perform_regression(x, y_percent):
    """
    片対数回帰分析。
    """
    if len(x) < 2: return 0.0, 0.0
    mask = y_percent > 0
    x_v, y_v = x[mask], np.log(y_percent[mask]) # log(Frequency)
    if len(x_v) < 2: return 0.0, 0.0
    
    slope, _, r_val, _, _ = stats.linregress(x_v, y_v)
    return -slope, r_val**2

# =========================================================
# 3. 解析実行関数 (生データの抽出)
# =========================================================

def analyze_file_and_get_raw(csv_path, remove_outliers, threshold_key):
    """
    指定ファイルの解析を行い、生の持続時間リストを返す。
    """
    res = calculate_states(csv_path, remove_outliers, threshold_key)
    
    if not res:
        return None

    states_map, _, ids, FPS = res
    pool = {0: [], 1: []} # 0:Inactive, 1:Active

    # 全個体の持続時間をプール
    for uid in ids:
        states = states_map[uid].values
        # RLE
        diffs = np.diff(states)
        indices = np.concatenate(([0], np.where(diffs != 0)[0] + 1, [len(states)]))
        lengths = np.diff(indices) / FPS
        values = states[indices[:-1]]
        
        for v, l in zip(values, lengths):
            pool[v].append(l)

    return pool

# 4. グラフ描画関数 (コロニーごとの比較ダッシュボード)

def get_layout_params(n_plots, single_fig_size):
    if n_plots <= 1:
        return 1, 1, single_fig_size
    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    total_width = single_fig_size[0] * n_cols
    total_height = single_fig_size[1] * n_rows
    return n_rows, n_cols, (total_width, total_height)

def plot_state_duration_ccdf(colony_dataset, output_dir, file_name, fig_size, use_y_log, auto_save, stats_list):
    """
    コロニーごとに Isolated/Paired/Trio を重ね書きしたダッシュボードを作成
    colony_dataset: { "Colony A": {"Isolated": pool, "Paired": pool, ...}, ... }
    """
    n_plots = len(colony_dataset)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()
    
    # 色設定
    colors = {"Isolate": "blue", "Pair": "orange", "Trio": "green"}
    # 描画順序
    order = ["Isolate", "Pair", "Trio"]
    # 線種設定
    plot_styles = [(0, "-", "Inactive"), (1, "--", "Active")]

    sorted_colonies = sorted(colony_dataset.keys())

    for i, colony_name in enumerate(sorted_colonies):
        if i >= len(axes): break
        ax = axes[i]
        
        conditions_data = colony_dataset[colony_name]
        
        for condition in order:
            if condition not in conditions_data: continue
            pool = conditions_data[condition]
            c = colors.get(condition, "gray")
            
            for state_val, line_style, state_label in plot_styles:
                durs = pool[state_val]
                
                # CCDF計算
                x, y = calculate_ccdf_coords(durs)
                k, r2 = perform_regression(x, y)
                
                if len(x) > 0:
                    ax.plot(x, y, linestyle=line_style, color=c, alpha=0.8, linewidth=1.5,
                            label=f"{condition} ({state_label})")
                    
                    # 統計リストに追加
                    stats_list.append({
                        "Colony": colony_name,
                        "Condition": condition,
                        "State": state_label,
                        "k (Slope)": round(k, 4),
                        "R2": round(r2, 4),
                        "Count": len(durs)
                    })

        ax.set_title(colony_name, fontsize=20)
        ax.grid(True, which="both", linestyle=':', alpha=0.5)
        ax.set_xlim(0,4500)


        if use_y_log:
            ax.set_yscale('log')
            ax.set_ylim(1, 110)
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)

        if i >= (n_rows - 1) * n_cols: # 最下段
            ax.set_xlabel("Duration Time [sec]", fontsize=20)
        if i % n_cols == 0: # 左端
            ax.set_ylabel('Probability [%]', fontsize=20)
        

        # 凡例はグラフ外、または最後のグラフのみなどの調整が可能だが
        # ここでは視認性のため各グラフの右上に配置（枠外）
        ax.legend(loc='upper right', fontsize=15)

    # 余白の調整と不要なAxesの削除
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if n_plots > 1:
        plt.suptitle("Activity State Duration", fontsize=20)
        plt.tight_layout()
    else:
        plt.tight_layout()

    # plt.subplots_adjust(wspace=0.3, hspace=0.4, right=0.85)

    save_path = os.path.join(output_dir, file_name)
    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph Saved: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_state_duration_ccdf_All_avarage(global_pool, output_dir, file_name, fig_size, use_y_log, auto_save, stats_list):
    """
    【全体平均グラフ】
    全コロニーのデータを条件ごとにプールして、1枚に重ね書きする
    """
    if not global_pool: return

    fig, ax = plt.subplots(figsize=(8,4))
    
    colors = {"Isolate": "blue", "Pair": "orange", "Trio": "green"}
    order = ["Isolate", "Pair", "Trio"]
    plot_styles = [(0, "-", "Inactive"), (1, "--", "Active")]

    for condition in order:
        if condition not in global_pool: continue
        pool = global_pool[condition]
        c = colors.get(condition, "gray")
        
        for state_val, line_style, state_label in plot_styles:
            durs = pool[state_val]
            
            # CCDF計算
            x, y = calculate_ccdf_coords(durs)
            k, r2 = perform_regression(x, y)
            
            if len(x) > 0:
                ax.plot(x, y, linestyle=line_style, color=c, alpha=0.8, linewidth=1.5,
                        label=f"{condition} ({state_label})")
                
                stats_list.append({
                    "Type": "Pooled_All",
                    "Colony": "ALL",
                    "Condition": condition,
                    "State": state_label,
                    "k (Slope)": round(k, 4),
                    "R2": round(r2, 4),
                    "Count": len(durs)
                })

    ax.set_title("Activity State Duration (All Colonies Average)", fontsize=15)
    ax.set_xlabel("Duration Time $τ$ [sec]", fontsize=15)
    ax.set_ylabel("$1 - P(X<τ)×100$ [%]", fontsize=15)
    ax.grid(True, which="both", linestyle=':', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlim(0,4500)

    if use_y_log:
        ax.set_yscale('log')
        ax.set_ylim(1, 110)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)

    ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(output_dir, file_name)
    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph Saved: {save_path}")
        plt.close()
    else:
        plt.show()

# メイン処理
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

    # 2. ユーザー設定
    BASE_PATH = "/Users/sonya/卒論データ/analysis_data"
    OUTPUT_DIR = "/Users/sonya/卒論データ/analysis_data/graph_output"
    
    # パラメータ
    REMOVE_OUTLIERS = True
    VELOCITY_THRESHOLD = "avg_half"
    USE_Y_LOG = True
    FIG_SIZE = (6, 4) # 個別グラフの基本サイズ
    AUTO_SAVE = False

    if AUTO_SAVE and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # データセット構造の定義
    CONDITIONS = [
        (ISO_DICT, "Isolate"),
        (PAIR_DICT, "Pair"),
        (TRIO_DICT, "Trio")
    ]
    
    # コロニーごとのデータ格納庫
    # Structure: {"Colony A": {"Isolate": pool, "Pair": pool, ...}, ...}
    COLONY_DATASET = {}
    # 全個体平均用
    GLOBAL_POOL = {
        "Isolate": {0: [], 1: []},
        "Pair":   {0: [], 1: []},
        "Trio":     {0: [], 1: []}
    }
    
    # 全コロニー名のリスト作成
    all_colonies = set()
    for d, _ in CONDITIONS:
        all_colonies.update(d.keys())
    
    for c_name in all_colonies:
        COLONY_DATASET[c_name] = {}

    ALL_STATS = []

    # 3. データ収集
    print("データ解析開始...")
    
    for source_dict, condition_label in CONDITIONS:
        for colony_name, folder_id in source_dict.items():
            velocity_csv = os.path.join(BASE_PATH, folder_id, f"{folder_id}-position_velocity.csv")
            
            # 各ファイルの生データを取得
            pool = analyze_file_and_get_raw(velocity_csv, REMOVE_OUTLIERS, VELOCITY_THRESHOLD)
            
            if pool:
                # 1. コロニー別データセットに追加
                COLONY_DATASET[colony_name][condition_label] = pool
                
                # 2. 全体集計プールに追加
                GLOBAL_POOL[condition_label][0].extend(pool[0])
                GLOBAL_POOL[condition_label][1].extend(pool[1])
                
                print(f" - Loaded: {colony_name} ({condition_label})")

    # 4. グラフ描画 (コロニーごとの比較ダッシュボード)
    # print("\n - 状態持続時間グラフ作成")
    # plot_state_duration_ccdf(COLONY_DATASET, OUTPUT_DIR, "State_Duration_CCDF.png", 
    #                        FIG_SIZE, USE_Y_LOG, AUTO_SAVE, ALL_STATS)
    
    # 4. 全体平均 (Pooled) グラフ描画
    print("\n - 全体平均グラフ作成中")
    plot_state_duration_ccdf_All_avarage(GLOBAL_POOL, OUTPUT_DIR, "State_Duration_CCDF_All_avarage.png", 
                      FIG_SIZE, USE_Y_LOG, AUTO_SAVE, ALL_STATS)
    
    # 5. 統計データ保存
    if ALL_STATS and AUTO_SAVE:
        df_stats = pd.DataFrame(ALL_STATS)
        # カラム順序を整える
        cols = ["Colony", "Condition", "State", "k (Slope)", "R2", "Count"]
        df_stats = df_stats[cols]
        
        csv_path = os.path.join(OUTPUT_DIR, "state_duration_stats_by_colony.csv")
        df_stats.to_csv(csv_path, index=False)
        print(f"\n統計データを保存しました: {csv_path}")
    
    print("\nAll processes completed.")