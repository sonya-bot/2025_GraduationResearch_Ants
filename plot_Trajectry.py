# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import math
from itertools import combinations
from scipy import stats  # 統計検定用
import matplotlib.cm as cm

# 1. 計算・前処理関数

def data_input(position_csv_path):
    """
    入力データの読み込み、ロング形式への変換。
    """
    try:
        df_wide = pd.read_csv(position_csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return None, None
    
    df_temp = df_wide.copy()
    if 'position' in df_temp.columns:
        df_temp = df_temp.rename(columns={'position': 'frame'})
    
    # カラム名の正規化
    new_columns = {}
    for col in df_temp.columns:
        if col == 'frame': continue
        if col.startswith('x') and col[1:].isdigit():
            new_columns[col] = f"x_{col[1:]}"
        elif col.startswith('y') and col[1:].isdigit():
            new_columns[col] = f"y_{col[1:]}"
    if new_columns:
        df_temp.rename(columns=new_columns, inplace=True)

    # ID抽出
    x_cols = [c for c in df_temp.columns if c.startswith('x_')]
    ids = [c.replace('x_', '') for c in x_cols]
    
    # ロング形式へ変換
    frames = []
    for uid in ids:
        col_x = f'x_{uid}'
        col_y = f'y_{uid}'
        if col_x in df_temp.columns and col_y in df_temp.columns:
            sub_df = df_temp[['frame', col_x, col_y]].copy()
            sub_df.columns = ['frame', 'position_x', 'position_y']
            sub_df['id'] = uid
            frames.append(sub_df)
    
    if not frames:
        return None, None
        
    df_long = pd.concat(frames)
    df_long['id'] = pd.to_numeric(df_long['id'])

    # 5.時間軸の計算
    fps = 2.0
    time_minutes = df_temp.index / fps / 60.0  # 分単位の時間軸
    total_time_in_minutes = len(df_temp) / fps / 60.0

    return df_long, ids, time_minutes, fps

def estimate_arena_center(df_long):
    """
    全データの最大・最小値からアリーナの中心座標(cx, cy)のみを自動推定する。
    """
    x = df_long['position_x'].dropna().values
    y = df_long['position_y'].dropna().values
    
    if len(x) == 0: return 0, 0
    
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    return cx, cy

def calculate_contacts(df_long, ids, contact_threshold_px):
    """
    接触した座標のリストを返す (ピクセル単位)
    """
    contact_points_x = []
    contact_points_y = []
    
    ids_int = sorted([int(i) for i in ids])
    if len(ids_int) < 2:
        return [], []
    
    pairs = list(combinations(ids_int, 2))
    
    for p1, p2 in pairs:
        df1 = df_long[df_long['id'] == p1].set_index('frame')
        df2 = df_long[df_long['id'] == p2].set_index('frame')
        
        common = df1.index.intersection(df2.index)
        if len(common) == 0: continue
        
        pos1 = df1.loc[common, ['position_x', 'position_y']].values
        pos2 = df2.loc[common, ['position_x', 'position_y']].values
        
        dist = np.sqrt(np.sum((pos1 - pos2)**2, axis=1))
        is_contact = dist <= contact_threshold_px
        
        if np.any(is_contact):
            contact_points_x.extend(pos1[is_contact, 0])
            contact_points_y.extend(pos1[is_contact, 1])
            contact_points_x.extend(pos2[is_contact, 0])
            contact_points_y.extend(pos2[is_contact, 1])
            
    return contact_points_x, contact_points_y

def calculate_spatial_stats(df_long, ids, cx, cy, radius_px):
    """
    空間統計量を計算
    (ヒートマップと共通のグリッドサイズを使用)
    """
    # ★パラメータ固定
    FIELD_DIAMETER_PX = 1000.0
    FIELD_DIAMETER_MM = 380.0
    GRID_SIZE_PX = 20.0  # ヒートマップと統一

    diameter_px = FIELD_DIAMETER_PX
    diameter_mm = FIELD_DIAMETER_MM
    grid_size_px = GRID_SIZE_PX

    mm_per_px = diameter_mm / diameter_px

    stats_result = {}
    wall_zone_radius_px = radius_px * 0.85
    
    # 空間エントロピー正規化用の最大エントロピー
    # 円の中に含まれるおよそのグリッド数を計算
    valid_grid_area = np.pi * (radius_px**2)
    single_grid_area = grid_size_px**2
    approx_n_grids = valid_grid_area / single_grid_area
    max_entropy = np.log(approx_n_grids)

    for uid in ids:
        uid_num = int(uid)
        df_ind = df_long[df_long['id'] == uid_num]
        x_px = df_ind['position_x'].values
        y_px = df_ind['position_y'].values
        
        # 1. 総移動距離 (mm変換)
        dist_diff_px = np.sqrt(np.diff(x_px)**2 + np.diff(y_px)**2)
        total_dist_px = np.nansum(dist_diff_px)
        total_dist_mm = total_dist_px * mm_per_px
        
        # 2. 壁際滞在率 (Thigmotaxis)
        dist_from_center = np.sqrt((x_px - cx)**2 + (y_px - cy)**2)
        valid_mask = ~np.isnan(dist_from_center)
        total_frames = np.sum(valid_mask)
        if total_frames > 0:
            wall_frames = np.sum(dist_from_center[valid_mask] > wall_zone_radius_px)
            thigmotaxis_idx = wall_frames / total_frames
        else:
            thigmotaxis_idx = np.nan
            
        # 3. 空間エントロピー (正規化)
        if len(x_px) > 0:
            # 中心を原点としてグリッド化
            x_idx = np.floor((x_px - (cx - radius_px)) / grid_size_px).astype(int)
            y_idx = np.floor((y_px - (cy - radius_px)) / grid_size_px).astype(int)
            
            coords = np.column_stack((x_idx, y_idx))
            # 滞在したグリッドとその頻度
            _, counts = np.unique(coords, axis=0, return_counts=True)
            probs = counts / np.sum(counts)
            
            # Shannon entropy
            raw_entropy = -np.sum(probs * np.log(probs))
            
            # Normalize (0-1)
            norm_entropy = raw_entropy / max_entropy
        else:
            norm_entropy = np.nan
            
        stats_result[uid] = {
            'Total_Distance_px': total_dist_px,
            'Total_Distance_mm': total_dist_mm,
            'Thigmotaxis_Index': thigmotaxis_idx,
            'Spatial_Entropy': norm_entropy
        }
    return stats_result

def calculate_exploration_timeseries(df_long, ids, cx, cy):
    """
    時系列ごとの累積探索率を計算する
    """
    # ★パラメータ固定
    FIELD_DIAMETER_PX = 1000.0
    GRID_SIZE_PX = 20.0
    radius_px = FIELD_DIAMETER_PX / 2.0

    # 1. アリーナ内の有効グリッド総数を計算 (分母)
    x_range = np.arange(cx - radius_px, cx + radius_px, GRID_SIZE_PX)
    y_range = np.arange(cy - radius_px, cy + radius_px, GRID_SIZE_PX)
    
    valid_grid_count = 0
    for gx in x_range:
        for gy in y_range:
            g_center_x = gx + GRID_SIZE_PX / 2
            g_center_y = gy + GRID_SIZE_PX / 2
            if (g_center_x - cx)**2 + (g_center_y - cy)**2 <= radius_px**2:
                valid_grid_count += 1
    
    if valid_grid_count == 0: valid_grid_count = 1

    # 2. グリッド座標への変換とフィルタリング
    df = df_long.copy()
    # アリーナ外のデータを除外
    df['dist'] = np.sqrt((df['position_x'] - cx)**2 + (df['position_y'] - cy)**2)
    df = df[df['dist'] <= radius_px].copy()
    
    # グリッドインデックスに変換
    df['gx_idx'] = np.floor((df['position_x'] - (cx - radius_px)) / GRID_SIZE_PX)
    df['gy_idx'] = np.floor((df['position_y'] - (cy - radius_px)) / GRID_SIZE_PX)
    
    results = {}
    
    min_frame = df['frame'].min()
    max_frame = df['frame'].max()
    all_frames = np.arange(min_frame, max_frame + 1)
    
    # --- A. コロニー全体 (Total) ---
    total_visited = df.groupby(['gx_idx', 'gy_idx'])['frame'].min().sort_values()
    counts = np.searchsorted(total_visited.values, all_frames, side='right')
    rates = (counts / valid_grid_count) * 100.0
    results['Total'] = (all_frames, rates)
    
    # --- B. 個体別 (Individual) ---
    for uid in ids:
        uid_num = int(uid)
        df_ind = df[df['id'] == uid_num]
        
        if len(df_ind) == 0:
            results[uid] = (all_frames, np.zeros_like(all_frames))
            continue
            
        ind_visited = df_ind.groupby(['gx_idx', 'gy_idx'])['frame'].min().sort_values()
        counts_ind = np.searchsorted(ind_visited.values, all_frames, side='right')
        rates_ind = (counts_ind / valid_grid_count) * 100.0
        results[uid] = (all_frames, rates_ind)
        
    return results

def calculate_overlap_stats(df_long, ids, cx, cy, radius_px):
    """
    【新規追加】重複度 (Overlap Score) を計算する関数
    エントロピー計算と同じグリッド定義(20px)を使用。
    """
    # ★パラメータ固定 (エントロピーと統一)
    GRID_SIZE_PX = 20.0
    
    # 全個体の滞在確率分布を作成
    prob_distributions = {} # {uid: {grid_tuple: prob}}
    
    for uid in ids:
        uid_num = int(uid)
        df_ind = df_long[df_long['id'] == uid_num]
        x_px = df_ind['position_x'].values
        y_px = df_ind['position_y'].values
        
        if len(x_px) == 0:
            prob_distributions[uid] = {}
            continue

        # グリッド座標計算
        x_idx = np.floor((x_px - (cx - radius_px)) / GRID_SIZE_PX).astype(int)
        y_idx = np.floor((y_px - (cy - radius_px)) / GRID_SIZE_PX).astype(int)
        coords = np.column_stack((x_idx, y_idx))
        
        # ユニークなグリッドとカウント (n_A,i)
        unique_grids, counts = np.unique(coords, axis=0, return_counts=True)
        total_counts = np.sum(counts) # N_A
        
        # 確率分布 (p_A,i)
        dist_map = {}
        for grid, count in zip(unique_grids, counts):
            dist_map[tuple(grid)] = count / total_counts
        prob_distributions[uid] = dist_map

    # ペアワイズ重複度 (Schoener's Index)
    ids_list = list(ids)
    n_ids = len(ids_list)
    overlap_results = {uid: np.nan for uid in ids} # デフォルトNaN

    if n_ids < 2:
        return overlap_results # Isolateなどは計算不可

    # 全ペアの重複度を計算
    for i in range(n_ids):
        uid_i = ids_list[i]
        my_scores = []
        
        for j in range(n_ids):
            if i == j: continue
            uid_j = ids_list[j]
            
            # ペア (i, j) の分布取得
            dist_i = prob_distributions.get(uid_i, {})
            dist_j = prob_distributions.get(uid_j, {})
            
            # 和集合グリッドについて差分を計算
            union_grids = set(dist_i.keys()) | set(dist_j.keys())
            diff_sum = 0.0
            for g in union_grids:
                p_i = dist_i.get(g, 0.0)
                p_j = dist_j.get(g, 0.0)
                diff_sum += abs(p_i - p_j)
            
            # O_ij = 1 - 0.5 * Σ |p_i - p_j|
            o_ij = 1.0 - 0.5 * diff_sum
            my_scores.append(o_ij)
        
        # 平均スコア (Trioの場合は2匹との平均、Pairの場合は1匹との値)
        if my_scores:
            overlap_results[uid_i] = np.mean(my_scores)
            
    return overlap_results

def collect_spatial_statistics(target_dict, mode, base_path, stats_list):
    """
    【修正】重複度の計算と格納ロジックを追加
    """
    print(f" - 統計データ収集中 ({mode})...")
    
    FIELD_DIAMETER_PX = 1000.0
    radius_px = FIELD_DIAMETER_PX / 2.0

    for name, folder_id in target_dict.items():
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        df_long, ids, _, _ = data_input(csv_path)
        
        if df_long is None: continue
        
        cx, cy = estimate_arena_center(df_long)
        
        # 1. 基本統計
        basic_stats = calculate_spatial_stats(df_long, ids, cx, cy, radius_px)
        
        # 2. 探索率
        exp_results = calculate_exploration_timeseries(df_long, ids, cx, cy)
        
        # 3. 重複度 (Overlap Score) ★ここを追加
        overlap_stats = calculate_overlap_stats(df_long, ids, cx, cy, radius_px)

        # 最終的なTotal探索率
        final_total_rate = 0.0
        if 'Total' in exp_results:
            _, t_rates = exp_results['Total']
            if len(t_rates) > 0:
                final_total_rate = t_rates[-1]
        
        for uid, val in basic_stats.items():
            # 個体の最終探索率
            final_indiv_rate = 0.0
            key_uid = str(uid) # 文字列型でキー検索
            if key_uid not in exp_results and int(uid) in exp_results:
                key_uid = int(uid)
            
            if key_uid in exp_results:
                _, i_rates = exp_results[key_uid]
                if len(i_rates) > 0:
                    final_indiv_rate = i_rates[-1]

            # 個体の重複度 ★ここを追加
            overlap_val = np.nan
            if str(uid) in overlap_stats: overlap_val = overlap_stats[str(uid)]
            elif int(uid) in overlap_stats: overlap_val = overlap_stats[int(uid)]

            stats_list.append({
                "Group_Type": mode,
                "Colony_Name": name,
                "ID": uid,
                "Total_Distance_px": round(val['Total_Distance_px'], 2),
                "Total_Distance_mm": round(val['Total_Distance_mm'], 2),
                "Thigmotaxis_Index": round(val['Thigmotaxis_Index'], 4),
                "Spatial_Entropy": round(val['Spatial_Entropy'], 4),
                "Final_Exploration_Rate_Indiv": round(final_indiv_rate, 2),
                "Final_Exploration_Rate_Total": round(final_total_rate, 2),
                "Overlap_Score": round(overlap_val, 4) # ★カラム追加
            })

# 2. グラフ描画関数

def get_layout_params(n_plots, single_fig_size):
    if n_plots <= 1:
        return 1, 1, single_fig_size
    n_cols = min(4, n_plots) # 最大4列
    n_rows = math.ceil(n_plots / n_cols)
    total_width = single_fig_size[0] * n_cols * 1.1
    total_height = single_fig_size[1] * n_rows
    return n_rows, n_cols, (total_width, total_height)

def plot_trajectory(target_dict, mode, base_path, 
                    contact_threshold_px, single_fig_size, save_path, auto_save):
    """
    軌跡図 (Trajectory Plot)
    """
    # 描画時間範囲(分)
    START_TIME = 0 # Noneの場合、最初から
    END_TIME = 30  # Noneの場合、全フレーム

    PLOT_CONTACT_SPOTS = False

    FIELD_DIAMETER_PX = 1000.0
    radius_px = FIELD_DIAMETER_PX / 2.0

    n_plots = len(target_dict)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, single_fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    print(f"\n - 軌跡図作成: {mode}")

    for i, (name, folder_id) in enumerate(target_dict.items()):
        if i >= len(axes): break
        ax = axes[i]
        
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        df_long, ids, _, _= data_input(csv_path)

        if df_long is None:
            ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center')
            continue

        cx, cy = estimate_arena_center(df_long)

        # 描画範囲のフィルタリング
        if START_TIME is not None:
            df_long = df_long[df_long['frame'] >= START_TIME*60*2] #時間→フレーム変換
        if END_TIME is not None:
            df_long = df_long[df_long['frame'] <= END_TIME*60*2] #時間→フレーム変換

        # 描画フレーム数の確認
        print(f"  - {name}: 個体数={len(ids)}, フレーム数={df_long['frame'].nunique()}")

        # 1. アリーナ境界
        circle = plt.Circle((cx, cy), radius_px, fill=False, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
        ax.add_patch(circle)

        # 2. 軌跡プロット
        for uid in ids:
            uid_num = int(uid)
            df_ind = df_long[df_long['id'] == uid_num].sort_values('frame')
            x = df_ind['position_x'].values
            y = df_ind['position_y'].values
            
            ax.plot(x, y, linewidth=0.8, alpha=0.6, label=f'ID:{uid}')
        
        # 3. 接触点
        if PLOT_CONTACT_SPOTS and len(ids) > 1:
            if len(ids) >= 2:
                cx_pts, cy_pts = calculate_contacts(df_long, ids, contact_threshold_px)
                if len(cx_pts) > 0:
                    ax.scatter(cx_pts, cy_pts, s=10, c='black', alpha=0.9, zorder=10, label=None)

        ax.set_title(name, fontsize=20)
        ax.set_aspect('equal')
        
        # 軸範囲
        margin = radius_px * 0.1
        ax.set_xlim(cx - radius_px - margin, cx + radius_px + margin)
        ax.set_ylim(cy - radius_px - margin, cy + radius_px + margin)
        ax.grid(True, linestyle=':', alpha=0.3)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if len(ids) <= 10:
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=15)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if n_plots > 1:
        plt.suptitle(f"Trajectory Map ({mode})", fontsize=20, y=0.98)
        plt.tight_layout()
    else:
        plt.tight_layout()
        
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_heatmap_overview(target_dict, mode, base_path, single_fig_size, save_path, auto_save):
    """
    滞在ヒートマップ (Stay Heatmap)
    """
    FIELD_DIAMETER_PX = 1000.0
    GRID_SIZE_PX = 20.0
    radius_px = FIELD_DIAMETER_PX / 2.0

    n_plots = len(target_dict)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, single_fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    print(f"\n - 滞在ヒートマップ作成: {mode}")

    for i, (name, folder_id) in enumerate(target_dict.items()):
        if i >= len(axes): break
        ax = axes[i]
        
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        df_long, ids, _, _= data_input(csv_path)

        if df_long is None:
            ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center')
            continue

        cx, cy = estimate_arena_center(df_long)
        all_x = df_long['position_x'].dropna().values
        all_y = df_long['position_y'].dropna().values
        
        if len(all_x) == 0: continue

        margin = radius_px * 0.1
        x_min, x_max = cx - radius_px - margin, cx + radius_px + margin
        y_min, y_max = cy - radius_px - margin, cy + radius_px + margin
        
        nx = int((x_max - x_min) / GRID_SIZE_PX)
        ny = int((y_max - y_min) / GRID_SIZE_PX)
        if nx <= 0: nx = 10
        if ny <= 0: ny = 10

        weights = np.ones_like(all_x) / len(all_x) * 100
        
        try:
            h = ax.hist2d(all_x, all_y, bins=[nx, ny], range=[[x_min, x_max], [y_min, y_max]],
                          weights=weights, cmap='coolwarm', 
                          norm=mcolors.LogNorm(vmin=1e-2, vmax=1e1))
        except ValueError:
            continue
            
        circle = plt.Circle((cx, cy), radius_px, fill=False, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
        ax.add_patch(circle)

        ax.set_title(name, fontsize=20)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04).set_label(label='Freqency [%]', fontsize=12)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if n_plots > 1:
        plt.suptitle(f"Stay Heatmap ({mode})", fontsize=20, y=0.98)
        plt.tight_layout()
    else:
        plt.tight_layout()
        
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_contact_heatmap(target_dict, mode, base_path, contact_threshold_px, single_fig_size, save_path, auto_save):
    """
    接触ヒートマップ (Contact Heatmap)
    """
    FIELD_DIAMETER_PX = 1000.0
    GRID_SIZE_PX = 20.0
    radius_px = FIELD_DIAMETER_PX / 2.0

    n_plots = len(target_dict)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, single_fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    print(f"\n - 接触ヒートマップ作成: {mode}")

    for i, (name, folder_id) in enumerate(target_dict.items()):
        if i >= len(axes): break
        ax = axes[i]
        
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        df_long, ids, _, _= data_input(csv_path)

        if df_long is None:
            ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center')
            continue

        cx, cy = estimate_arena_center(df_long)
        contact_x, contact_y = calculate_contacts(df_long, ids, contact_threshold_px)
        
        if len(contact_x) == 0:
            ax.text(0.5, 0.5, 'No Contact Data', ha='center', va='center')
            continue

        margin = radius_px * 0.1
        x_min, x_max = cx - radius_px - margin, cx + radius_px + margin
        y_min, y_max = cy - radius_px - margin, cy + radius_px + margin
        
        nx = int((x_max - x_min) / GRID_SIZE_PX)
        ny = int((y_max - y_min) / GRID_SIZE_PX)
        if nx <= 0: nx = 10
        if ny <= 0: ny = 10

        weights = np.ones_like(contact_x) / len(contact_x) * 100
        
        try:
            h = ax.hist2d(contact_x, contact_y, bins=[nx, ny], range=[[x_min, x_max], [y_min, y_max]],
                          weights=weights, cmap='coolwarm', 
                          norm=mcolors.LogNorm(vmin=1e-2, vmax=1e1))
        except ValueError:
            continue 
        circle = plt.Circle((cx, cy), radius_px, fill=False, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
        ax.add_patch(circle)
        ax.set_title(name, fontsize=20)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04).set_label(label='Freqency [%]', fontsize=12)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    if n_plots > 1:
        plt.suptitle(f"Contact Heatmap ({mode})", fontsize=20, y=0.98)
        plt.tight_layout()
    else:
        plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_exploration_rate(target_dict, mode, base_path, single_fig_size, save_path, auto_save):
    """
    探索率時系列グラフ (Cumulative Exploration Rate)
    """
    n_plots = len(target_dict)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, (6,4))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    print(f"\n - 探索率グラフ作成: {mode}")

    for i, (name, folder_id) in enumerate(target_dict.items()):
        if i >= len(axes): break
        ax = axes[i]
        
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        df_long, ids, time_minutes, fps= data_input(csv_path)

        if df_long is None:
            ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center')
            continue

        cx, cy = estimate_arena_center(df_long)
        results = calculate_exploration_timeseries(df_long, ids, cx, cy)
        
        final_total_rate = 0.0
        
        for uid in ids:
            if uid in results:
                frames, rates = results[uid]
                time_min = frames / (fps * 60.0)
                ax.plot(time_min, rates, linewidth=1.5, alpha=0.7, label=f'ID:{uid}')
        
        if 'Total' in results:
            frames, rates = results['Total']
            time_min = frames / (fps * 60.0)
            ax.plot(time_min, rates, linewidth=2.5, alpha=1.0, color='black', label='Total')
            if len(rates) > 0:
                final_total_rate = rates[-1]

        ax.set_title(name, fontsize=20)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper left', fontsize=15)

        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Time [min]', fontsize=20)
        if i % n_cols == 0:
            ax.set_ylabel('Exploration Rate [%]', fontsize=20)

        # テキストボックスの表示 (現在のコロニーのTotal Rateのみを表示)
        text_str = f"Total Exploration Rate: {final_total_rate:.2f}%"
        ax.text(0.10, 0.05, text_str, transform=ax.transAxes, fontsize=20,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='lightgray'))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if n_plots > 1:
        plt.suptitle(f"Exploration Rate ({mode})", fontsize=20, y=0.98)
        plt.tight_layout()
    else:
        plt.tight_layout()
        
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")
        plt.close()
    else:
        plt.show()
    
def plot_total_exploration_rate(target_dict, mode, base_path, output_dir, fig_size, save_path, auto_save):
    """
    【新規】コロニーごとの合計探索率(Total)の時系列変化を一覧で比較するグラフ
    - 各コロニーのTotal線を色分けしてプロット
    - 全体の平均推移を黒太線でプロット
    """
    import scipy.interpolate as interp

    # プロットするコロニーがない場合は終了
    if not target_dict:
        return

    print(f"\n - コロニー別Total探索率比較グラフ作成 ({mode})...")

    # 1. 配色設定 (他のグラフと統一するためコロニー名でソート)
    # target_dictのキー(コロニー名)を取得してソート
    sorted_colonies = sorted(target_dict.keys())
    # カラーマップ (jet)
    colors = plt.cm.jet(np.linspace(0, 1, len(sorted_colonies)))
    colony_colors = {name: color for name, color in zip(sorted_colonies, colors)}
    
    # データを蓄積するリスト
    interp_rates_list = []

    fig, ax = plt.subplots(figsize=(6,4))

    # 3. 各コロニーのデータを処理
    for name, folder_id in target_dict.items():
        # CSVパス
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        
        # データ読み込み
        df_long, ids, _, fps = data_input(csv_path)
        if df_long is None: continue

        # 探索率計算
        cx, cy = estimate_arena_center(df_long)
        results = calculate_exploration_timeseries(df_long, ids, cx, cy)

        # 'Total' のデータを取得
        if 'Total' in results:
            frames, rates = results['Total']
            if len(frames) == 0: continue
            
            # 時間(分)に変換
            times_min = frames / (fps * 60.0)
            
            # A. 個別コロニーのプロット
            color = colony_colors.get(name, 'gray')
            ax.plot(times_min, rates, color=color, linewidth=1.5, alpha=0.5, label=name)
            
            # B. 平均計算用の補間 (Interpolation)
            # 線形補間関数を作成
            f = interp.interp1d(times_min, rates, kind='linear', bounds_error=False, fill_value=(0, rates[-1]))
            # 共通時間軸上の値を取得
            interp_rates_list.append(rates)

    # 4. 平均線の計算とプロット
    if interp_rates_list:
        # 縦方向に平均をとる (axis=0)
        all_rates_arr = np.array(interp_rates_list)
        mean_rates = np.mean(all_rates_arr, axis=0)
        
        # 黒太線でプロット
        ax.plot(times_min, mean_rates, color='black', alpha=0.8, linewidth=3.0, label='Average')

    # 5. グラフの体裁
    ax.set_title(f"Total Exploration Rate ({mode})", fontsize=20)
    ax.set_xlabel("Time [min]", fontsize=20)
    ax.set_ylabel("Exploration Rate [%]", fontsize=20)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.5)

    # 凡例を枠外(右側)に配置
    # ax.legend(title="Colony", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=12, title_fontsize=12)

    plt.tight_layout()

    # 6. 保存または表示
    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")
        plt.close()
    else:
        plt.show()
    

def plot_entropy_comparison(all_stats, output_dir, file_name, fig_size, auto_save):
    """
    条件ごとのエントロピー分布を箱ひげ図で比較
    """
    # データを整形
    # ここでのキーは DATA_SETS で指定した "Isolate", "Pair", "Trio" に合わせる
    data_dict = {"Isolate": [], "Pair": [], "Trio": []}
    
    for rec in all_stats:
        gtype = rec["Group_Type"]
        val = rec["Spatial_Entropy"]
        
        # グループ名正規化 (Trioed -> Trio)
        if gtype == "Trioed": gtype = "Trio"
        
        if gtype in data_dict:
            if not np.isnan(val):
                data_dict[gtype].append(val)

    # データを取得するためのキーリスト (data_dictのキーと一致させる)
    keys = ["Isolate", "Pair", "Trio"]
    
    # グラフのX軸に表示するラベルリスト
    labels = ["Isolate", "Pair", "Trio"]
    
    # 正しいキーを使ってデータを抽出
    plot_data = [data_dict[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6,6))
    
    # 箱ひげ図
    box = ax.boxplot(plot_data, labels=labels, patch_artist=True, showfliers=False)
    colors = ['lightblue', 'orange', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # 散布図
    for i, data in enumerate(plot_data):
        y = data
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.6, s=15, color='black', zorder=3)

    # 統計検定 (Kruskal-Wallis)
    # データが存在する群だけで検定を行う
    valid_plot_data = [d for d in plot_data if len(d) > 0]
    if len(valid_plot_data) > 1:
        stat, p_val = stats.kruskal(*valid_plot_data)
        p_text = f"(p={p_val:.2e})"
        # 小数表記に変換(小数第2位まで)
        if p_val >= 0.01:
            p_text = f"(p={p_val:.4f})"
        title_text = f"Spatial Entropy\n{p_text}"
    else:
        title_text = "Spatial Entropy "

    ax.set_title(title_text, fontsize=20)
    ax.set_ylabel("Entropy", fontsize=20)
    ax.set_xlabel("Group Size", fontsize=20)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()

    if auto_save:
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフ保存完了: {save_path}")
        plt.close()
    else:
        plt.show()
    
    print("\n--- Spatial Entropy ---")
    for label, data in zip(labels, plot_data):
        if data:
            print(f"{label}: Mean={np.mean(data):.4f}, Std={np.std(data):.4f}, N={len(data)}")

def plot_exploration_boxplot(all_stats, output_path, auto_save):
    """
    ハイブリッド箱ひげ図 (探索率用)
    Individual (点) vs Colony Total (箱)
    """
    # データ収集
    # indiv_data: 個体ごとのスコア (点用)
    indiv_data = {"Isolate": [], "Pair": [], "Trio": []}
    # total_data: コロニーごとのスコア (箱用, 重複排除)
    total_data_map = {"Isolate": {}, "Pair": {}, "Trio": {}} # ColonyName -> Rate

    for rec in all_stats:
        gtype = rec["Group_Type"]
        cname = rec["Colony_Name"]
        if gtype == "Trioed": gtype = "Trio"
        
        if gtype in indiv_data:
            # 個体データ
            if not np.isnan(rec["Final_Exploration_Rate_Indiv"]):
                indiv_data[gtype].append(rec["Final_Exploration_Rate_Indiv"])
            # コロニーデータ (上書きでユニーク化)
            if not np.isnan(rec["Final_Exploration_Rate_Total"]):
                total_data_map[gtype][cname] = rec["Final_Exploration_Rate_Total"]

    keys = ["Isolate", "Pair", "Trio"]
    labels = ["Isolate", "Pair", "Trio"]
    
    indiv_plot_list = [indiv_data[k] for k in keys]
    total_plot_list = [list(total_data_map[k].values()) for k in keys]

    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. Individual Data (Scatter - Back Layer)
    for i, data in enumerate(indiv_plot_list):
        x = np.random.normal(i + 1, 0.05, size=len(data)) # ジッター
        ax.scatter(x, data, s=15, color='black', zorder=3, label='Individual' if i==0 else "")

    # 2. Colony Total Data (Box Plot - Front Layer)
    box = ax.boxplot(total_plot_list, labels=labels, patch_artist=True, showfliers=False)
    colors = ['lightblue', 'orange', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7) # 少し濃く

    # 3. Colony Total Data (Scatter - Marker Overlay)
    for i, data in enumerate(total_plot_list):
        x = np.random.normal(i + 1, 0.02, size=len(data))
        # 箱と同じ色相で濃い枠線などをつけると見やすい
        ax.scatter(x, data, alpha=0.6, s=50, marker='D', edgecolor='black', zorder=2,
        facecolor=colors[i], label='Colony Total' if i==0 else "")

    # 検定 (Colony Totalに対して)
    valid_data = [d for d in total_plot_list if len(d) > 0]
    p_text = ""
    if len(valid_data) > 1:
        try:
            _, p_val = stats.kruskal(*valid_data)
            p_text = f"\n(Colony : p={p_val:.2e})"
            # 小数表記に変換(小数第2位まで)
            if p_val >= 0.01:
                p_text = f"(Colony : p={p_val:.4f})"

        except ValueError: pass

    ax.set_title(f"Total Exploration Rate\n{p_text}", fontsize=20)
    ax.set_ylabel("Exploration Rate [%]", fontsize=20)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Group Size", fontsize=20)
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticklabels(labels=[0, 20, 40, 60, 80, 100], fontsize=15)
    ax.grid(True, axis='y', ls='--', alpha=0.5)
    
    # 凡例 (ダミープロットで作成)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='Individual', markersize=6),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markeredgecolor='black', label='Colony', markersize=6, alpha=0.6)
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=12)

    plt.tight_layout()
    if auto_save: plt.savefig(output_path, dpi=300); plt.close()
    else: plt.show()

def plot_overlap_exploration_scatter(all_stats, output_path, auto_save):
    """
    【修正版】重複度 vs 探索率 散布図
    - 配色: plot_Activity_Interval.py と完全統一 (jetカラーマップ)
    - 凡例: コロニーと色の対応を枠外に表示
    """
    # 1. データのフィルタリング (Pair, Trioのみ)
    target_data = [r for r in all_stats if r["Group_Type"] in ["Pair", "Trio", "Trioed"]]
    if not target_data:
        print("No Pair/Trio data for Overlap Scatter plot.")
        return

    # 2. カラーマップの生成 (plot_Activity_Interval.pyと同一ロジック)
    # コロニー名をソートして固定することで、他グラフと色を一致させる
    unique_colonies = sorted(list(set(r["Colony_Name"] for r in target_data)))
    n_colonies = len(unique_colonies)
    
    # jetカラーマップから等間隔に色を取得
    colors = plt.cm.jet(np.linspace(0, 1, n_colonies))
    colony_color_map = {col: color for col, color in zip(unique_colonies, colors)}

    # 3. グラフ描画
    # 凡例スペース確保のため横幅を広げる
    fig, ax = plt.subplots(figsize=(6, 4))

    # マーカー定義
    markers = {"Pair": "^", "Trio": "o", "Trio": "o"}
    
    # プロット実行
    for rec in target_data:
        x_val = rec["Overlap_Score"]
        y_val = rec["Final_Exploration_Rate_Indiv"]
        gtype = rec["Group_Type"]
        cname = rec["Colony_Name"]
        
        if np.isnan(x_val) or np.isnan(y_val): continue
        
        color = colony_color_map.get(cname, 'gray')
        marker = markers.get(gtype, "o")
        
        ax.scatter(x_val, y_val, color=color, marker=marker, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

    # 4. 軸とタイトルの設定
    ax.set_title("Overlap vs Exploration Rate", fontsize=20)
    ax.set_xlabel("Individual Overlap Score", fontsize=20)
    ax.set_ylabel("Exploration Rate [%]", fontsize=20)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.5)

    # 5. 凡例の作成 (2部構成: 形状と色)
    from matplotlib.lines import Line2D

    # (A) 形状の凡例 (Group Size)
    shape_legend_elems = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', label='Pair', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Trio', markersize=10, markeredgecolor='k'),
    ]
    # 左上に配置
    legend1 = ax.legend(handles=shape_legend_elems, loc='upper left', title="Group Size", framealpha=0.8)
    ax.add_artist(legend1)

    # (B) 色の凡例 (Colony) - 枠外右側に配置
    color_legend_elems = []
    for cname in unique_colonies:
        color = colony_color_map[cname]
        # 丸いマーカーで色を示す
        color_legend_elems.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=cname, markersize=8, markeredgecolor='none')
        )
    
    # 枠外(bbox_to_anchor)に配置
    ax.legend(handles=color_legend_elems, title="Colony", 
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=12)

    plt.tight_layout()

    # 6. 保存または表示
    if auto_save:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"散布図を保存しました: {output_path}")
        plt.close()
    else:
        plt.show()

def plot_entropy_exploration_scatter(all_stats, output_path, auto_save):
    """
    【新規追加】空間エントロピー vs 探索率 散布図
    X: Spatial Entropy
    Y: Exploration Rate
    """
    # 1. データフィルタリング (Isolateも比較対象として面白いので含めるか、
    #    あるいはPair/Trioに限定するか。ここでは全条件プロットしますが、
    #    マーカーで区別します)
    target_data = [r for r in all_stats if not np.isnan(r["Spatial_Entropy"]) and not np.isnan(r["Final_Exploration_Rate_Indiv"])]
    
    if not target_data:
        print("No valid data for Entropy vs Exploration plot.")
        return

    # 2. カラーマップ (コロニー統一)
    unique_colonies = sorted(list(set(r["Colony_Name"] for r in target_data)))
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_colonies)))
    colony_color_map = {col: color for col, color in zip(unique_colonies, colors)}

    fig, ax = plt.subplots(figsize=(10, 6))

    # マーカー定義 (Isolateも含める場合)
    markers = {"Isolate": "s", "Pair": "^", "Trio": "o"}
    
    for rec in target_data:
        x_val = rec["Spatial_Entropy"]
        y_val = rec["Final_Exploration_Rate_Indiv"]
        gtype = rec["Group_Type"]
        cname = rec["Colony_Name"]
        
        color = colony_color_map.get(cname, 'gray')
        marker = markers.get(gtype, "o")
        
        # プロット
        ax.scatter(x_val, y_val, color=color, marker=marker, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

    # 軸設定
    ax.set_title("Spatial Entropy vs Exploration Rate", fontsize=20)
    ax.set_xlabel("Spatial Entropy", fontsize=20)
    ax.set_ylabel("Exploration Rate [%]", fontsize=20)
    ax.set_xlim(0.0, 1.0) # エントロピーは0-1 (正規化済みの場合)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.5)

    # 凡例 (Group Size + Colony)
    from matplotlib.lines import Line2D
    
    # 形状
    shape_legend_elems = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', label='Isolate', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', label='Pair', markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Trio', markersize=10, markeredgecolor='k'),
    ]
    legend1 = ax.legend(handles=shape_legend_elems, loc='upper left', title="Group Size", framealpha=0.8)
    ax.add_artist(legend1)

    # 色 (枠外配置)
    color_legend_elems = []
    for cname in unique_colonies:
        color = colony_color_map[cname]
        color_legend_elems.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=cname, markersize=8, markeredgecolor='none')
        )
    ax.legend(handles=color_legend_elems, title="Colony", 
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=10)

    plt.tight_layout()

    if auto_save:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# メイン処理
if __name__ == "__main__":
    
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

    TARGET_SPECIFIC = [] # 空リストで全実行(例: []), 特定コロニーのみ実行(例: ["Colony A", "Colony B"])
    BASE_PATH = "/Users/sonya/卒論データ/analysis_data"
    OUTPUT_DIR = "/Users/sonya/卒論データ/analysis_data/graph_output"
    CONTACT_THRESHOLD = 50.0
    FIG_SIZE = (4, 4)
    AUTO_SAVE  = True

    if AUTO_SAVE and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_spatial_stats = []

    DATA_SETS = [
        (ISO_DICT, "Isolate", "Isolate"),
        (PAIR_DICT, "Pair", "Pair"),
        (TRIO_DICT, "Trio", "Trio") # ラベルを Trio に統一
    ]

    # 4. 実行ループ
    for source_dict, mode_label, file_suffix in DATA_SETS:
        
        if TARGET_SPECIFIC:
            process_dict = {k: v for k, v in source_dict.items() if k in TARGET_SPECIFIC}
            if not process_dict: continue
            is_specific_mode = True
        else:
            process_dict = source_dict
            is_specific_mode = False
# 
        # 統計データの収集
        collect_spatial_statistics(process_dict, mode_label, BASE_PATH, all_spatial_stats)

        # グラフ描画
        if is_specific_mode:
            for target_colony, folder_id in process_dict.items():
                single_dict = {target_colony: folder_id}
                print(f"\nProcessing Specific Target: {target_colony} ({mode_label})")
                
                # 各種グラフ作成
                plot_trajectory(single_dict, f"{target_colony} - {mode_label}", BASE_PATH, CONTACT_THRESHOLD, FIG_SIZE, os.path.join(OUTPUT_DIR, f"Trajectory_{target_colony}_{file_suffix}.png"), AUTO_SAVE)
                plot_contact_heatmap(single_dict, f"{target_colony} - {mode_label}", BASE_PATH, CONTACT_THRESHOLD, FIG_SIZE, os.path.join(OUTPUT_DIR, f"Contact_Heatmap_{target_colony}_{file_suffix}.png"), AUTO_SAVE)
                plot_heatmap_overview(single_dict, f"{target_colony} - {mode_label}", BASE_PATH, FIG_SIZE, os.path.join(OUTPUT_DIR, f"Heatmap_Stay_{target_colony}_{file_suffix}.png"), AUTO_SAVE)
                plot_exploration_rate(single_dict, f"{target_colony} - {mode_label}", BASE_PATH, FIG_SIZE, os.path.join(OUTPUT_DIR, f"Exploration_Rate_{target_colony}_{file_suffix}.png"), AUTO_SAVE)
        else:
            print(f"\nProcessing All Targets ({mode_label})")
            
            # plot_trajectory(process_dict, mode_label, BASE_PATH, CONTACT_THRESHOLD, FIG_SIZE, os.path.join(OUTPUT_DIR, f"Trajectory_All_{file_suffix}.png"), AUTO_SAVE)
            # plot_heatmap_overview(process_dict, mode_label, BASE_PATH, FIG_SIZE, os.path.join(OUTPUT_DIR, f"Heatmap_Stay_All_{file_suffix}.png"), AUTO_SAVE)
            # plot_contact_heatmap(process_dict, mode_label, BASE_PATH, CONTACT_THRESHOLD, FIG_SIZE, os.path.join(OUTPUT_DIR, f"Contact_Heatmap_All_{file_suffix}.png"), AUTO_SAVE)
            plot_exploration_rate(process_dict, mode_label, BASE_PATH, FIG_SIZE, os.path.join(OUTPUT_DIR, f"Exploration_Rate_All_{file_suffix}.png"), AUTO_SAVE)
            plot_total_exploration_rate(process_dict, mode_label, BASE_PATH, OUTPUT_DIR, FIG_SIZE, os.path.join(OUTPUT_DIR, f"Total_Exploration_Rate_All_{file_suffix}.png"), AUTO_SAVE)

    # 5. 統計処理 & CSV保存
    if all_spatial_stats:
        # 空間エントロピー比較グラフの作成
        # plot_entropy_comparison(all_spatial_stats, OUTPUT_DIR, "Spatial_Entropy_Comparison.png", FIG_SIZE, AUTO_SAVE)
        # ハイブリッド箱ひげ図 (探索率)
        plot_exploration_boxplot(all_spatial_stats, os.path.join(OUTPUT_DIR, "Comparison_Exploration_Hybrid.png"), AUTO_SAVE)

        # 重複度 vs 探索率 散布図
        # plot_overlap_exploration_scatter(all_spatial_stats, os.path.join(OUTPUT_DIR, "Overlap_vs_Exploration_ColonyColor.png"), AUTO_SAVE)

        # 空間エントロピー vs 探索率 散布図
        # plot_entropy_exploration_scatter(all_spatial_stats, os.path.join(OUTPUT_DIR, "Entropy_vs_Exploration_ColonyColor.png"), AUTO_SAVE)

        if AUTO_SAVE:
            df_stats = pd.DataFrame(all_spatial_stats)
            # カラム順序定義に Overlap_Score を追加
            cols_order = [
                "Group_Type", "Colony_Name", "ID", 
                "Total_Distance_px", "Total_Distance_mm", "Thigmotaxis_Index", "Spatial_Entropy",
                "Overlap_Score", # ★追加
                "Final_Exploration_Rate_Indiv", "Final_Exploration_Rate_Total"
            ]
            final_cols = [c for c in cols_order if c in df_stats.columns]
            df_stats = df_stats[final_cols]
            
            csv_save_path = os.path.join(OUTPUT_DIR, "spatial_summary.csv")
            df_stats.to_csv(csv_save_path, index=False)
            print(f"\n空間統計データを保存しました: {csv_save_path}")