# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations

def plot_trajectory(position_csv_path, contact_threshold, fig_size, auto_save, plot_contact):
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
    
if __name__ == "__main__":
    INPUT_CSV = "20251101_01"
    INPUT_POSITION_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position.csv"
    CONTACT_THRESHOLD = 50.0
    FIG_SIZE = (10, 5)
    AUTO_SAVE = True
    PLOT_CONTACT = True # 接触点の表示

    plot_trajectory(INPUT_POSITION_CSV, CONTACT_THRESHOLD, FIG_SIZE, AUTO_SAVE, PLOT_CONTACT)