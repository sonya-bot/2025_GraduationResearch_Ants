# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import LogLocator, ScalarFormatter
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import calculate_thresholds
import math

# 1. データ読み込み・前処理

def load_data_pair(folder_path, folder_id, fps=2.0):
    """
    速度データと位置データを読み込み、同期させて返す
    """
    vel_path = os.path.join(folder_path, f"{folder_id}-position_velocity.csv")
    pos_path = os.path.join(folder_path, f"{folder_id}-position.csv")
    
    # 1. 速度データの読み込み
    try:
        df_vel = pd.read_csv(vel_path)
    except FileNotFoundError:
        print(f"エラー: 速度ファイルが見つかりません - {vel_path}")
        return None, None, None

    # 2. 位置データの読み込み
    try:
        df_pos = pd.read_csv(pos_path)
    except FileNotFoundError:
        print(f"エラー: 位置ファイルが見つかりません - {pos_path}")
        return None, None, None

    # ID抽出
    speed_cols = [col for col in df_vel.columns if col.startswith('speed_')]
    ids = [col.replace('speed_', '') for col in speed_cols]
    
    # 位置データのカラム名正規化 (x_1, y_1形式へ)
    # df_posのカラムが x1, y1 または position_x_1 などの可能性があるため調整
    # ここでは plot_Trajectry.py の data_input ロジックを簡易化して適用
    pos_map = {}
    for col in df_pos.columns:
        if col.startswith('x') and col[1:].isdigit():
            pos_map[col] = f"x_{col[1:]}"
        elif col.startswith('y') and col[1:].isdigit():
            pos_map[col] = f"y_{col[1:]}"
    if pos_map:
        df_pos.rename(columns=pos_map, inplace=True)

    # 3. 前処理 (速度スムージング)
    smoothing_window = int(60 * fps)
    if smoothing_window > 1:
        for col in speed_cols:
            df_vel[col] = df_vel[col].rolling(window=smoothing_window, center=True, min_periods=1).mean().fillna(0)
            
    # 数値変換と外れ値処理
    for col in speed_cols:
        df_vel[col] = pd.to_numeric(df_vel[col], errors='coerce').fillna(0)
        df_vel.loc[df_vel[col] > 200.0, col] = np.nan

    # 行数合わせ (positionとvelocityで行数がずれる場合があるため、短い方に合わせる)
    min_len = min(len(df_vel), len(df_pos))
    df_vel = df_vel.iloc[:min_len].reset_index(drop=True)
    df_pos = df_pos.iloc[:min_len].reset_index(drop=True)

    return df_vel, df_pos, ids

def calculate_social_states(df_vel, df_pos, ids, velocity_csv_path, threshold_key, contact_threshold_px=50.0):
    """
    各フレームの状態を判定する
    0: Solitary Inactive (離れて休息)
    1: Social Inactive (集まって休息)
    2: Active (活動中) -> 今回は解析対象外だが判定は行う
    """
    # 閾値取得
    _, ind_results = calculate_thresholds.get_threshold_values(velocity_csv_path)
    if not ind_results: return None
    thresh_map = {r['id']: r[threshold_key] for r in ind_results}

    fps = 2.0
    n_frames = len(df_vel)
    
    # 結果格納用辞書: {uid: [state_array]}
    states_dict = {}
    
    # 座標データの準備 (計算高速化のためnumpy配列化)
    # coords shape: (n_frames, n_ids, 2)
    coords = np.zeros((n_frames, len(ids), 2))
    id_to_idx = {uid: i for i, uid in enumerate(ids)}
    
    for i, uid in enumerate(ids):
        col_x = f"x_{uid}"
        col_y = f"y_{uid}"
        if col_x in df_pos.columns and col_y in df_pos.columns:
            coords[:, i, 0] = df_pos[col_x].values
            coords[:, i, 1] = df_pos[col_y].values
        else:
            coords[:, i, :] = np.nan # データなし

    # 全フレームの距離計算
    # idsが1匹(Isolated)の場合は距離計算不要
    social_contact_mask = np.zeros((n_frames, len(ids)), dtype=bool)
    
    if len(ids) >= 2:
        for f in range(n_frames):
            frame_coords = coords[f] # (n_ids, 2)
            # 全ペア距離行列
            if np.any(np.isnan(frame_coords)):
                continue
            
            dists = squareform(pdist(frame_coords)) # (n_ids, n_ids)
            np.fill_diagonal(dists, np.inf) # 自分自身との距離は除外
            
            # 各個体について、距離が閾値以下の相手が1匹でもいるか
            has_contact = np.any(dists <= contact_threshold_px, axis=1)
            social_contact_mask[f] = has_contact

    # 状態判定ループ
    for i, uid in enumerate(ids):
        th = thresh_map.get(uid)
        if th is None: continue
        
        # 1. 速度判定 (Active vs Inactive)
        speed_vals = df_vel[f'speed_{uid}'].fillna(0).values
        is_active = speed_vals >= th  # 閾値以上ならActive
        
        # 2. 接触判定 (Social vs Solitary)
        # Isolatedの場合は常に False (Solitary)
        is_contact = social_contact_mask[:, i]
        
        # 状態割り当て
        # Default: 0 (Solitary Inactive)
        states = np.zeros(n_frames, dtype=int)
        
        # Active: 2
        states[is_active] = 2
        
        # Social Inactive: Inactive(=Not Active) AND Contact
        # (states == 0) の場所 (=Inactive) かつ is_contact が True の場所を 1 に
        is_inactive = ~is_active
        states[is_inactive & is_contact] = 1
        
        states_dict[uid] = states

    return states_dict

def extract_durations(states_dict, fps=2.0):
    """
    状態配列から持続時間を抽出してプールする
    Target States: 
      0: Solitary Inactive
      1: Social Inactive
    (2: Active は無視)
    """
    pool = {0: [], 1: []}
    
    for uid, states in states_dict.items():
        # RLE
        diffs = np.diff(states)
        indices = np.concatenate(([0], np.where(diffs != 0)[0] + 1, [len(states)]))
        lengths = np.diff(indices) / fps
        values = states[indices[:-1]]
        
        for v, l in zip(values, lengths):
            if v in [0, 1]: # Active(2)は除外
                pool[v].append(l)
                
    return pool

# 2. 計算・描画ロジック

def calculate_ccdf_coords(durations):
    if not durations: return [], []
    sorted_durs = np.sort(durations)[::-1]
    n = len(sorted_durs)
    ranks = np.arange(1, n + 1)
    probs_percent = (ranks / n) * 100
    return sorted_durs, probs_percent

def perform_regression(x, y_percent):
    if len(x) < 2: return 0.0, 0.0
    mask = y_percent > 0
    x_v, y_v = x[mask], np.log(y_percent[mask])
    if len(x_v) < 2: return 0.0, 0.0
    slope, _, r_val, _, _ = stats.linregress(x_v, y_v)
    return -slope, r_val**2

def get_layout_params(n_plots, single_fig_size):
    if n_plots <= 1: return 1, 1, single_fig_size
    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    return n_rows, n_cols, (single_fig_size[0] * n_cols * 1.1, single_fig_size[1] * n_rows)

def plot_social_rest_dashboard(colony_dataset, output_dir, file_name, fig_size, auto_save, stats_list):
    """
    コロニーごとに Social vs Solitary を比較するダッシュボード
    """
    n_plots = len(colony_dataset)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()
    
    colors = {"Isolate": "blue", "Pair": "orange", "Trio": "green"}
    order = ["Isolate", "Pair", "Trio"]
    
    # 線種定義
    # 0: Isolated (点線), 1: Social (実線)
    line_styles = {0: "--", 1: "-"}
    labels = {0: "Isolated", 1: "United"}

    sorted_colonies = sorted(colony_dataset.keys())

    for i, colony_name in enumerate(sorted_colonies):
        if i >= len(axes): break
        ax = axes[i]
        
        conditions_data = colony_dataset[colony_name]
        
        for condition in order:
            if condition not in conditions_data: continue
            pool = conditions_data[condition]
            c = colors.get(condition, "gray")
            
            for state_val in [0, 1]: # Solitary, Social
                durs = pool[state_val]
                if not durs: continue # Socialがない(Isolatedなど)場合はスキップ
                
                x, y = calculate_ccdf_coords(durs)
                k, r2 = perform_regression(x, y)
                
                if len(x) > 0:
                    ls = line_styles[state_val]
                    lbl = labels[state_val]
                    ax.plot(x, y, linestyle=ls, color=c, alpha=0.8, linewidth=1.5,
                            label=f"{condition} ({lbl})")
                    
                    stats_list.append({
                        "Type": "Individual_Colony",
                        "Colony": colony_name,
                        "Condition": condition,
                        "Rest_Type": lbl,
                        "k (Slope)": round(k, 4),
                        "R2": round(r2, 4),
                        "Count": len(durs)
                    })

        ax.set_title(colony_name, fontsize=20)
        ax.grid(True, which="both", linestyle=':', alpha=0.5)
        
        ax.set_yscale('log')
        ax.set_ylim(1, 110)
        ax.set_xlim(0, 4500)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)

        if i >= (n_rows - 1) * n_cols: # 最下段
            ax.set_xlabel("Duration Time [sec]", fontsize=20)
        if i % n_cols == 0: # 左端
            ax.set_ylabel('Probability [%]', fontsize=20)

        ax.legend(loc='upper right', fontsize=15)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if n_plots > 1:
        plt.suptitle("Rest State Duration", fontsize=20)
        plt.tight_layout()
    else:
        plt.tight_layout()
    
    # plt.subplots_adjust(wspace=0.3, hspace=0.4, right=0.85)

    if auto_save:
        plt.savefig(os.path.join(output_dir, file_name), dpi=300, bbox_inches='tight')
        print(f"Graph Saved: {file_name}")
        plt.close()
    else:
        plt.show()

def plot_social_rest_pooled(global_pool, output_dir, file_name, fig_size, auto_save, stats_list):
    """
    全データをプールした平均グラフ
    """
    if not global_pool: return

    fig, ax = plt.subplots(figsize=fig_size)
    
    colors = {"Pair": "orange", "Trio": "green"}
    order = ["Pair", "Trio"]
    line_styles = {0: "-", 1: "-."}
    labels = {0: "Isolated", 1: "United"}

    for condition in order:
        if condition not in global_pool: continue
        pool = global_pool[condition]
        c = colors.get(condition, "gray") 
        
        for state_val in [0, 1]:
            durs = pool[state_val]
            if not durs: continue
            
            x, y = calculate_ccdf_coords(durs)
            k, r2 = perform_regression(x, y)
            
            if len(x) > 0:
                ls = line_styles[state_val]
                lbl = labels[state_val]
                ax.plot(x, y, linestyle=ls, color=c, alpha=0.8, linewidth=1.5,
                        label=f"{condition} ({lbl})")
                
                stats_list.append({
                    "Type": "Pooled_All",
                    "Colony": "ALL",
                    "Condition": condition,
                    "Rest_Type": lbl,
                    "k (Slope)": round(k, 4),
                    "R2": round(r2, 4),
                    "Count": len(durs)
                })

    ax.set_title("Inactive State Duration (All Colonies Average)", fontsize=15)
    ax.set_xlabel("Duration Time $τ$ [sec]", fontsize=15)
    ax.set_ylabel("$1 - P(X<τ)×100$ [%]", fontsize=15)
    ax.grid(True, which="both", linestyle=':', alpha=0.5)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1, 110)
    ax.set_xlim(0,4500)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)

    ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    if auto_save:
        plt.savefig(os.path.join(output_dir, file_name), dpi=600, bbox_inches='tight')
        print(f"Graph Saved: {file_name}")
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
    VELOCITY_THRESHOLD = "avg_half"
    CONTACT_THRESHOLD_PX = 50.0
    FIG_SIZE = (6, 4)
    POOLED_FIG_SIZE = (8, 4)
    AUTO_SAVE = False

    if AUTO_SAVE and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # データ構造初期化
    CONDITIONS = [
        (ISO_DICT, "Isolate"),
        (PAIR_DICT, "Pair"),
        (TRIO_DICT, "Trio")
    ]
    
    COLONY_DATASET = {}
    all_colonies = set()
    for d, _ in CONDITIONS: all_colonies.update(d.keys())
    for c_name in all_colonies: COLONY_DATASET[c_name] = {}

    GLOBAL_POOL = {
        "Isolate": {0: [], 1: []},
        "Pair":   {0: [], 1: []},
        "Trio":     {0: [], 1: []}
    }
    
    ALL_STATS = []

    # 3. 解析実行
    print("データ解析開始")
    
    for source_dict, condition_label in CONDITIONS:
        for colony_name, folder_id in source_dict.items():
            folder_path = os.path.join(BASE_PATH, folder_id)
            velocity_csv = os.path.join(folder_path, f"{folder_id}-position_velocity.csv")
            
            # データ読み込み
            df_vel, df_pos, ids = load_data_pair(folder_path, folder_id)
            if df_vel is None: continue
            
            # 状態判定
            states_dict = calculate_social_states(
                df_vel, df_pos, ids, velocity_csv, 
                VELOCITY_THRESHOLD, CONTACT_THRESHOLD_PX
            )
            if states_dict is None: continue
            
            # 持続時間抽出
            pool = extract_durations(states_dict)
            
            # 保存
            COLONY_DATASET[colony_name][condition_label] = pool
            GLOBAL_POOL[condition_label][0].extend(pool[0])
            GLOBAL_POOL[condition_label][1].extend(pool[1])
            
            print(f" - Processed: {colony_name} ({condition_label})")

    # 4. グラフ描画
    # print("\n - コロニー別ダッシュボード作成")
    # plot_social_rest_dashboard(
    #     COLONY_DATASET, OUTPUT_DIR, "Social_Rest_CCDF.png",
    #     FIG_SIZE, AUTO_SAVE, ALL_STATS
    # )
    
    print("\n - 全体平均グラフ作成")
    plot_social_rest_pooled(
        GLOBAL_POOL, OUTPUT_DIR, "Social_Rest_CCDF_All_avarage.png",
        POOLED_FIG_SIZE, AUTO_SAVE, ALL_STATS
    )

    # 5. 統計保存
    if ALL_STATS and AUTO_SAVE:
        df_stats = pd.DataFrame(ALL_STATS)
        cols = ["Type", "Colony", "Condition", "Rest_Type", "k (Slope)", "R2", "Count"]
        df_stats = df_stats[cols]
        csv_path = os.path.join(OUTPUT_DIR, "social_rest_stats.csv")
        df_stats.to_csv(csv_path, index=False)
        print(f"\n統計データを保存しました: {csv_path}")

    print("\nAll processes completed.")