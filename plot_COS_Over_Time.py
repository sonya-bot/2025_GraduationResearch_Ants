# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
from itertools import combinations
import calculate_thresholds
import math
from matplotlib.lines import Line2D


# 1. データ読み込み・前処理

def load_data_sync(folder_path, folder_id, fps=2.0):
    """
    速度データと位置データを読み込み、フレーム数を合わせて返す
    """
    vel_path = os.path.join(folder_path, f"{folder_id}-position_velocity.csv")
    pos_path = os.path.join(folder_path, f"{folder_id}-position.csv")
    
    # 読み込み
    try:
        df_vel = pd.read_csv(vel_path)
        df_pos = pd.read_csv(pos_path)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません - {e}")
        return None, None, None

    # ID抽出
    speed_cols = [col for col in df_vel.columns if col.startswith('speed_')]
    ids = [col.replace('speed_', '') for col in speed_cols]
    
    # 位置データのカラム名正規化
    pos_map = {}
    for col in df_pos.columns:
        if col.startswith('x') and col[1:].isdigit():
            pos_map[col] = f"x_{col[1:]}"
        elif col.startswith('y') and col[1:].isdigit():
            pos_map[col] = f"y_{col[1:]}"
    if pos_map:
        df_pos.rename(columns=pos_map, inplace=True)

    # 速度の前処理 (平滑化)
    smoothing_window = int(60 * fps)
    if smoothing_window > 1:
        for col in speed_cols:
            df_vel[col] = pd.to_numeric(df_vel[col], errors='coerce').fillna(0)
            df_vel[col] = df_vel[col].rolling(window=smoothing_window, center=True, min_periods=1).mean().fillna(0)
            
    # 外れ値処理 (200以上を0に)
    for col in speed_cols:
        df_vel.loc[df_vel[col] > 200.0, col] = 0

    # 行数合わせ
    min_len = min(len(df_vel), len(df_pos))
    df_vel = df_vel.iloc[:min_len].reset_index(drop=True)
    df_pos = df_pos.iloc[:min_len].reset_index(drop=True)

    return df_vel, df_pos, ids

# 2. 計算ロジック

def calculate_activity_count(df_vel, ids, velocity_csv_path, threshold_key):
    """
    各フレームでの同時活動個体数 (0 ~ N) を計算
    """
    _, ind_results = calculate_thresholds.get_threshold_values(velocity_csv_path)
    if not ind_results: return None
    thresh_map = {r['id']: r[threshold_key] for r in ind_results}

    n_frames = len(df_vel)
    active_matrix = np.zeros((n_frames, len(ids)), dtype=int)

    for i, uid in enumerate(ids):
        th = thresh_map.get(uid, 5.0) # デフォルト5.0
        speed_vals = df_vel[f'speed_{uid}'].values
        # 活動=1, 非活動=0
        active_matrix[:, i] = (speed_vals > th).astype(int)

    # 行ごとの合計 (=同時活動数)
    total_active_count = np.sum(active_matrix, axis=1)
    
    return total_active_count

def calculate_contacts(df_pos, ids, contact_threshold_px):
    """
    各フレームでの接触状態を判定
    Returns:
        contact_events: { 'label': [frames...], ... }
    """
    n_frames = len(df_pos)
    coords = {}
    for uid in ids:
        if f'x_{uid}' in df_pos.columns and f'y_{uid}' in df_pos.columns:
            coords[uid] = df_pos[[f'x_{uid}', f'y_{uid}']].values
        else:
            return np.array([])

    is_contact_frame = np.zeros(n_frames, dtype=bool)

    # 組み合わせごとに距離を計算し、接触があればフラグを立てる
    from itertools import combinations
    for id1, id2 in combinations(ids, 2):
        dist = np.sqrt(np.sum((coords[id1] - coords[id2])**2, axis=1))
        is_contact_frame |= (dist <= contact_threshold_px)

    return np.where(is_contact_frame)[0]

# 3. グラフ描画 (ダッシュボード)

def get_layout_params(n_plots, single_fig_size):
    if n_plots <= 1: return 1, 1, single_fig_size
    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    # 幅を少し広めに取る
    total_w = single_fig_size[0] * n_cols
    total_h = single_fig_size[1] * n_rows
    return n_rows, n_cols, (total_w, total_h)

def plot_activity_count_dashboard(target_dict, mode_label, base_path, fig_size, thresh_key, contact_thresh, auto_save, save_path):
    """
    指定されたコロニー群の同時活動数グラフをダッシュボード形式で作成
    凡例に活動数の構成比（%）を表示する
    """
    print(f"\n - 解析開始: {mode_label} ")
    
    n_plots = len(target_dict)
    if n_plots == 0: return

    # レイアウト計算
    n_rows, n_cols, figsize = get_layout_params(n_plots, fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    fps = 2.0
    
    # コロニーごとの処理
    sorted_colonies = sorted(target_dict.keys())
    
    for i, colony_name in enumerate(sorted_colonies):
        if i >= len(axes): break
        ax = axes[i]
        folder_id = target_dict[colony_name]
        folder_path = os.path.join(base_path, folder_id)
        
        # データ読み込み
        df_vel, df_pos, ids = load_data_sync(folder_path, folder_id, fps)
        if df_vel is None: 
            ax.text(0.5, 0.5, 'Data Load Error', ha='center')
            continue

        # 計算
        count_data = calculate_activity_count(
            df_vel, ids, 
            os.path.join(folder_path, f"{folder_id}-position_velocity.csv"), 
            thresh_key
        )
        
        # 接触計算
        contact_frames = calculate_contacts(df_pos, ids, contact_thresh)

        # 時間軸 (分)
        time_min = np.arange(len(count_data)) / fps / 60.0

        # --- 構成比の計算 ---
        # count_data の各値 (0, 1, 2...) の出現割合を計算
        counts = pd.Series(count_data).value_counts(normalize=True).sort_index()
        
        # --- プロット 1: 活動数 (ステップグラフ) ---
        ax.step(time_min, count_data, linewidth=1.0, alpha=0.8, where='mid')

        # --- プロット 2: 接触イベント (Y=0) ---
        has_contact = len(contact_frames) > 0
        if has_contact:
            t_events = contact_frames / fps / 60.0
            y_events = np.zeros_like(t_events) # Y=0上にプロット
            # 接触点プロット (赤色統一)
            ax.scatter(t_events, y_events-0.1, color='red', s=10, alpha=0.3, zorder=2)

        # --- 凡例の作成 ---
        # 線やマーカーを含まない、テキスト情報のみの凡例を作成
        handles = []
        
        for n in range(len(ids) + 1): # 0匹 ~ N匹
            pct = counts.get(n, 0.0) * 100
            label_text = f"{n} : {pct:.1f}%"
            # マーカーなし、線なし（透明）、ハンドル自体も透明にする
            handles.append(Line2D([0], [0], color='w', marker='None', label=label_text, alpha=0))

        # 装飾
        ax.set_title(f"{colony_name}", fontsize=20)
        ax.set_xlim(0, time_min[-1])
        
        # Y軸目盛り (整数のみ)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(-0.2, len(ids) + 0.9) # 0~N個まで入るように

        if i >= (n_rows - 1) * n_cols: # 最下段
            ax.set_xlabel('Time [min]', fontsize=20)
        if i % n_cols == 0: # 左端
            ax.set_ylabel("Active Workers", fontsize=20)

        ax.grid(True, linestyle=':', alpha=0.5)
        
        # 凡例表示 
        # handlelength=0, handletextpad=0 でシンボルスペースを詰める
        ax.legend(handles=handles, loc='upper right', ncol=2,
                  handlelength=0, handletextpad=0, fontsize=15, framealpha=0.8, title="Activity Workers", title_fontsize=15)

    # 余白処理
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Active Workers Count ({mode_label})", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフ保存完了: {save_path}")
        plt.close()
    else:
        plt.show()


# メイン処理

if __name__ == "__main__":
    
    # 1. データ定義
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
    
    # 2. 設定
    BASE_PATH = "/Users/sonya/卒論データ/analysis_data"
    OUTPUT_DIR = "/Users/sonya/卒論データ/analysis_data/graph_output"

    TARGET_SPECIFIC = ["Colony"]  # 空リストで全実行、特定コロニーのみ実行する場合はリストに追加
    
    # パラメータ設定 (辞書ではなく個別定義)
    FIG_SIZE = (6, 4)
    VELOCITY_THRESHOLD_KEY = 'avg_half'
    CONTACT_THRESHOLD_PX = 50.0
    AUTO_SAVE = False

    if AUTO_SAVE and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 3. 実行ループ
    DATA_SETS = [
        (PAIR_DICT, "Pair", "Pair"),
        (TRIO_DICT, "Trio", "Trio")
    ]
    
    for target_dict, label, suffix in DATA_SETS:

        # 特定コロニーのみ実行する場合のフィルタリング
        if TARGET_SPECIFIC:
            process_dict = {k: v for k, v in target_dict.items() if k in TARGET_SPECIFIC}
            if not process_dict: continue
            is_specific_mode = True
        else:
            process_dict = target_dict
            is_specific_mode = False

        # 保存ファイル名
        if is_specific_mode:
            save_name = f"Activity_Count_{suffix}_{'_'.join(TARGET_SPECIFIC)}.png"
            plot_activity_count_dashboard(
                process_dict, 
                label, 
                BASE_PATH, 
                FIG_SIZE, 
                VELOCITY_THRESHOLD_KEY, 
                CONTACT_THRESHOLD_PX, 
                AUTO_SAVE, 
                os.path.join(OUTPUT_DIR, save_name)
            )

        else:
            save_name = f"Activity_Count_{suffix}.png"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            
            # 個別の引数として渡す
            plot_activity_count_dashboard(
                target_dict, 
                label, 
                BASE_PATH, 
                FIG_SIZE, 
                VELOCITY_THRESHOLD_KEY, 
                CONTACT_THRESHOLD_PX, 
                AUTO_SAVE, 
                save_path
            )

    print("\nAll processes completed.")