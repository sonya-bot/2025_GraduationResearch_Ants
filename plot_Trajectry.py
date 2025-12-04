# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os
from itertools import combinations

def plot_trajectry(position_csv_path, contact_threshold, fig_size, auto_save, plot_contact, grid_size):
    """
    入力データの確認、個体数のカウントを行い
    実行する関数を呼び出す
    位置データから個体の移動軌跡を描画し、接触した地点にマーカーをプロットする。
    Hayashi et al. (2015) Fig. 5 の形式に倣う。
    """
    # 1.データの読み込み
    try:
        df_pos = pd.read_csv(position_csv_path)
        print(f"'{position_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return
    
    # 2.個体IDの特定と個体ごとの反復処理
    x_cols = [col for col in df_pos.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]

    n_individuals = len(individual_ids)
    if n_individuals < 1:
        print(f"エラー: 個体を検出できません")
        print("プログラムを終了します")
        return
    elif n_individuals >= 1:
        print(f"{n_individuals}匹の個体が検出されました。軌跡の描画を行います。")
        plot_trajectory_hist(position_csv_path, contact_threshold, fig_size, auto_save, plot_contact)
        if n_individuals >= 2:
            print(f"2匹以上の個体が検出されました。軌跡の描画とヒストグラムの作成を行います。")
            plot_stay_heatmap(position_csv_path, grid_size, fig_size, auto_save)
            plot_contact_heatmap(position_csv_path, contact_threshold, grid_size, fig_size, auto_save)
    

    


def calculate_field_boundary(all_x, all_y):
    """
    全個体の全座標から、フィールドの境界（中心と半径）を推定する。
    """
    valid_mask = ~np.isnan(all_x) & ~np.isnan(all_y)
    x = all_x[valid_mask]
    y = all_y[valid_mask]

    if len(x) == 0:
        return 0, 0, 0

    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # 半径はデータの広がりから推定
    radius_x = (max_x - min_x) / 2
    radius_y = (max_y - min_y) / 2
    radius = max(radius_x, radius_y)
    
    return center_x, center_y, radius

def draw_heatmap(x_data, y_data, title, save_path, grid_size, field_boundary, fig_size, auto_save):
    """
    共通のヒートマップ描画関数
    """
    plt.figure(figsize=fig_size)
    ax = plt.gca()

    # フィールド境界（円）の描画
    cx, cy, r = field_boundary
    circle = plt.Circle((cx, cy), r, fill=False, linestyle='--', linewidth=1.5, alpha=0.8, zorder=10)
    ax.add_patch(circle)

    # グリッド計算
    margin = r * 0.1
    x_min, x_max = cx - r - margin, cx + r + margin
    y_min, y_max = cy - r - margin, cy + r + margin
    
    nx = int((x_max - x_min) / grid_size)
    ny = int((y_max - y_min) / grid_size)
    
    if nx <= 0 or ny <= 0:
        print("エラー: グリッドサイズ設定が不適切です。")
        return

    if len(x_data) > 0:
        # ヒストグラム計算 (%表示)
        weights = np.ones_like(x_data) / len(x_data) * 100
        # normで色の範囲をパーセンテージに合わせて調整 (例: 0.01% から 10%)
        h = ax.hist2d(x_data, y_data, bins=[nx, ny], range=[[x_min, x_max], [y_min, y_max]],
                      weights=weights, cmap='coolwarm', norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1e1))
        H = h[0] # ヒストグラムのデータ
        
        # カラーバー
        cbar = plt.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Frequency (%)', rotation=270, labelpad=15)

        # # 等高線の表示
        # ax.contour(H.T, extent=[x_min, x_max, y_min, y_max], colors='black', linewidths=0.5, alpha=0.7, zorder=5)
    else:
        print(f"警告: {title} の描画データがありません。")

    # 装飾
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X position [pixels]', fontsize=12)
    ax.set_ylabel('Y position [pixels]', fontsize=12)
    ax.set_aspect('equal')
    # 目盛は表示しない
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

    # 保存または表示
    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" - ヒートマップを保存しました: {save_path}")
        plt.close()
    else:
        print(f" - ヒートマップを表示します: {title}")
        plt.show()

def plot_trajectory_hist(position_csv_path, contact_threshold, fig_size, auto_save, plot_contact):
    """
    位置データから個体の移動軌跡を描画し、接触した地点にマーカーをプロットする。
    Hayashi et al. (2015) Fig. 5 の形式に倣う。

    Args:
        position_csv_path (str): 位置情報CSVファイルのパス。
        contact_threshold (float): 接触とみなす距離（ピクセル）。
        fig_size (tuple): グラフのサイズ (幅, 高さ)。
        auto_save (bool): グラフを自動保存するかどうか。
    """

# 1.データの読み込み
    try:
        df_pos = pd.read_csv(position_csv_path)
        print(f"'{position_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return
    
# 2.個体IDの特定と個体ごとの反復処理
    x_cols = [col for col in df_pos.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]

    n_individuals = len(individual_ids)
    if n_individuals == 0:
        print(f"エラー: 個体を検出できません")
        print("プログラムを終了します")
        return
    
# 3. 座標データの取得 (高速化のためNumpy配列化)
    coords_dict = {}
    for uid in individual_ids:
        # 欠損値(NaN)がある場合はそのフレームをスキップするか、線が途切れるようにする
        coords_dict[uid] = df_pos[[f'x{uid}', f'y{uid}']].to_numpy()
    print(f" - 座標データを取得しました。")

# 4. 接触フレームの特定
    # 各個体が「どのフレーム」で「誰か」と接触していたかを記録するフラグ配列を作成
    # shape: (フレーム数, )
    total_frames = len(df_pos)
    contact_flags = {uid: np.zeros(total_frames, dtype=bool) for uid in individual_ids}

    pair_combinations = list(combinations(individual_ids, 2))
    
    for id_a, id_b in pair_combinations:
        pos_a = coords_dict[id_a]
        pos_b = coords_dict[id_b]
        
        # ユークリッド距離を計算
        dists = np.sqrt(np.sum((pos_a - pos_b)**2, axis=1))
        
        # 閾値以下のフレームを特定
        is_contacting = dists <= contact_threshold
        
        # 両方の個体のフラグをTrueにする (OR演算)
        contact_flags[id_a] |= is_contacting
        contact_flags[id_b] |= is_contacting

# 5. グラフの描画
    # 正方形に近い比率で描画（軌跡の歪みを防ぐため）
    plt.figure(figsize=fig_size) 
    ax = plt.gca()

    # フィールドの境界目安（データの最大最小から自動設定）
    all_x = np.concatenate([coords_dict[uid][:, 0] for uid in individual_ids])
    all_y = np.concatenate([coords_dict[uid][:, 1] for uid in individual_ids])
    # nanを除去
    all_x = all_x[~np.isnan(all_x)]
    all_y = all_y[~np.isnan(all_y)]
    
    if len(all_x) > 0:
        min_x, max_x = np.min(all_x), np.max(all_x)
        min_y, max_y = np.min(all_y), np.max(all_y)
        
        # 余白を持たせる
        margin_x = (max_x - min_x) * 0.05
        margin_y = (max_y - min_y) * 0.05
        ax.set_xlim(min_x - margin_x, max_x + margin_x)
        ax.set_ylim(min_y - margin_y, max_y + margin_y)
        
        # アスペクト比を1:1に固定（重要: 移動空間を正しく表現するため）
        ax.set_aspect('equal')

    # プロットループ
    for i, uid in enumerate(individual_ids):
        pos = coords_dict[uid]
        x, y = pos[:, 0], pos[:, 1]
        
        # 5_1. 軌跡の線 (Trajectory)
        # alpha=0.5 で薄くして、重なりやドットを見やすくする
        ax.plot(x, y, label=f'ID: {uid}', linewidth=0.8, alpha=0.5, zorder=1)

        # 5_2. 接触点 (Contact Spots)
        if plot_contact == True:
            # 接触フラグがTrueの座標のみを抽出
            contact_mask = contact_flags[uid]
            contact_x = x[contact_mask]
            contact_y = y[contact_mask]
            
            if len(contact_x) > 0:
                # 論文の形式に倣い、接触点をドットでプロット
                # zorder=2 で線の上に描画
                ax.scatter(contact_x, contact_y, s=15, color="black", 
                        edgecolors='none', alpha=0.9, zorder=2, label=None)
            ax.set_title(f'Trajectory and Contact Points (N={n_individuals})', fontsize=14)

        else:
            ax.set_title(f'Trajectory (N={n_individuals})', fontsize=14)

    # 装飾
    ax.set_xlabel('X position (pixels)', fontsize=12)
    ax.set_ylabel('Y position (pixels)', fontsize=12)
    
    # 凡例 (線の色＝個体)
    ax.legend(loc='upper right', fontsize=10)
    
    # グリッド (控えめに)
    ax.grid(True, linestyle=':', alpha=0.3)

    # 目盛は表示しない
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

    # 保存または表示
    if auto_save:
        output_filename = f"Trajectory_Map(N={n_individuals}).png"
        output_directory = os.path.dirname(position_csv_path)
        save_path = os.path.join(output_directory, output_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" - 軌跡グラフを保存しました: {save_path}")
    else:
        print(" - 軌跡グラフを表示します")
        plt.show()
    


def plot_stay_heatmap(position_csv_path, grid_size, fig_size, auto_save):
    """
    個体の滞在頻度（どこに長くいたか）をヒートマップで可視化する。
    """
    # データの読み込みとID特定
    try:
        df_pos = pd.read_csv(position_csv_path)
    except FileNotFoundError:
        return
    x_cols = [col for col in df_pos.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]
    n_individuals = len(individual_ids)
    
    if n_individuals == 0: return

    print(f"{n_individuals} 個体の滞在分布を解析します...")

    # 全座標データの収集
    all_x_list = []
    all_y_list = []
    for uid in individual_ids:
        x = df_pos[f'x{uid}'].to_numpy()
        y = df_pos[f'y{uid}'].to_numpy()
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        all_x_list.append(x[valid_mask])
        all_y_list.append(y[valid_mask])
    
    flat_x = np.concatenate(all_x_list)
    flat_y = np.concatenate(all_y_list)
    
    # 境界推定と描画
    field_boundary = calculate_field_boundary(flat_x, flat_y)
    title = f"Stay Distribution Heatmap (N={n_individuals})"
    output_dir = os.path.dirname(position_csv_path)
    save_path = os.path.join(output_dir, f"Spatial_Distribution_Stay(N={n_individuals}).png")
    
    draw_heatmap(flat_x, flat_y, title, save_path, grid_size, field_boundary, fig_size, auto_save)


def plot_contact_heatmap(position_csv_path, contact_threshold, grid_size, fig_size, auto_save):
    """
    接触が発生した場所（社会的ホットスポット）をヒートマップで可視化する。
    """
    # データの読み込み
    try:
        df_pos = pd.read_csv(position_csv_path)
    except FileNotFoundError:
        return 
    x_cols = [col for col in df_pos.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]
    n_individuals = len(individual_ids)
    
    if n_individuals < 2:
        print("接触ヒートマップ: 個体数が2未満のためスキップします。")
        return

    print(f"{n_individuals} 個体の接触分布を解析します...")

    coords_dict = {}
    for uid in individual_ids:
        coords_dict[uid] = df_pos[[f'x{uid}', f'y{uid}']].to_numpy()

    # 境界推定用（全データ）
    all_x_bound = np.concatenate([coords_dict[uid][:, 0] for uid in individual_ids])
    all_y_bound = np.concatenate([coords_dict[uid][:, 1] for uid in individual_ids])
    field_boundary = calculate_field_boundary(all_x_bound, all_y_bound)

    # 接触座標の抽出
    contact_x_list = []
    contact_y_list = []
    pair_combinations = list(combinations(individual_ids, 2))
    
    for id_a, id_b in pair_combinations:
        pos_a = coords_dict[id_a]
        pos_b = coords_dict[id_b]
        dists = np.sqrt(np.sum((pos_a - pos_b)**2, axis=1))
        contact_indices = np.where(dists <= contact_threshold)[0]
        
        if len(contact_indices) > 0:
            contact_x_list.append(pos_a[contact_indices, 0])
            contact_y_list.append(pos_a[contact_indices, 1])
            contact_x_list.append(pos_b[contact_indices, 0])
            contact_y_list.append(pos_b[contact_indices, 1])

    if len(contact_x_list) == 0:
        print("接触データがありませんでした。")
        return

    flat_contact_x = np.concatenate(contact_x_list)
    flat_contact_y = np.concatenate(contact_y_list)
    
    # 描画
    title = f"Contact Distribution Heatmap (N={n_individuals})"
    output_dir = os.path.dirname(position_csv_path)
    save_path = os.path.join(output_dir, f"Spatial_Distribution_Contact(N={n_individuals}).png")
    
    draw_heatmap(flat_contact_x, flat_contact_y, title, save_path, grid_size, field_boundary, fig_size, auto_save)

if __name__ == "__main__":
    INPUT_CSV = "20251117_01"
    INPUT_POSITION_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position.csv"
    CONTACT_THRESHOLD = 50.0
    FIG_SIZE = (10, 5)
    AUTO_SAVE = False
    PLOT_CONTACT = True # 接触点の表示

    # ヒートマップの粒度(接触分布、滞在分布)
    GRID_SIZE = 20

    plot_trajectry(INPUT_POSITION_CSV, CONTACT_THRESHOLD, FIG_SIZE, AUTO_SAVE, PLOT_CONTACT, GRID_SIZE)

    # # 各機能のON/OFF
    # DO_PLOT_TRAJECTORY = True
    # DO_PLOT_STAY_HEATMAP = True
    # DO_PLOT_CONTACT_HEATMAP = True

    # if DO_PLOT_TRAJECTORY:
    #     plot_trajectory_hist(INPUT_POSITION_CSV, CONTACT_THRESHOLD, FIG_SIZE, AUTO_SAVE, PLOT_CONTACT)

    # if DO_PLOT_STAY_HEATMAP:
    #     plot_stay_heatmap(INPUT_POSITION_CSV, GRID_SIZE, FIG_SIZE, AUTO_SAVE)

    # if DO_PLOT_CONTACT_HEATMAP:
    #     plot_contact_heatmap(INPUT_POSITION_CSV, CONTACT_THRESHOLD, GRID_SIZE, FIG_SIZE, AUTO_SAVE)
