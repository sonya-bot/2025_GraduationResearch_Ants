# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import calculate_thresholds as calc
import plot_social_network

def plot_distance_over_time(input_filename, contact_threshold):
    """
    個体の位置データから、全ての個体ペア間の距離の時系列変化を棒グラフで描画する。
    接触しきい値を下回った時点を赤い点で示す。

    Args:
        input_filename (str): 位置情報(x, y)が含まれるCSVファイル名。
        contact_threshold (float): 2個体が「接触している」と判断する最大距離（ピクセル）。
    """
    # 1. データの読み込み
    try:
        df = pd.read_csv(input_filename)
        print(f"'{input_filename}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {input_filename}")
        return
    
    # 2. 個体IDの特定と個体数のチェック
    x_cols = [col for col in df.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]
    n_individuals = len(individual_ids)
    
    if n_individuals <= 1:
        print(f"エラー: 検出された個体数が {n_individuals} のため、距離グラフを作成できません。")
        return
    
    print(f"{n_individuals} 個体を対象に、全ペアの距離変化グラフを作成します。")
    
    # 3. 全ての個体のペアについてループ処理
    pair_combinations = list(combinations(individual_ids, 2))
    
    for id1, id2 in pair_combinations:
        # 4. ペア間のユークリッド距離を全フレームにわたって計算
        pos1 = df[[f'x{id1}', f'y{id1}']].to_numpy()
        pos2 = df[[f'x{id2}', f'y{id2}']].to_numpy()
        distances = np.sqrt(np.sum((pos1 - pos2)**2, axis=1))
        
        # 5. グラフの描画
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # 棒グラフを作成
        frames = df['position']
        ax.bar(frames, distances, color='lightblue', label=f'Distance between ID:{id1} and ID:{id2}')
        
        # 6. 接触点の特定とプロット
        contact_frames = frames[distances <= contact_threshold]
        contact_distances = distances[distances <= contact_threshold]
        
        if not contact_frames.empty:
            plot_y = np.zeros_like(contact_frames)
            ax.plot(contact_frames, plot_y, 'o', color='red', markersize=3, label=f'Contact (<= {contact_threshold} px)')
        
        # グラフの体裁を整える
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Distance (pixels)', fontsize=12)
        ax.set_title(f'Distance over Time between Individual {id1} and {id2}', fontsize=14)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # y軸の範囲を調整して見やすくする
        ax.set_ylim(0, distances.max() * 1.1)

        # Y軸の目盛りを細かくする(nbinsで調整可能)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=20))
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # ◆◆◆ 設定 ◆◆◆
    # 1. 分析対象の位置データファイル
    INPUT_CSV = "/Volumes/100.108.13.8/analysis_data/20251015_03/20251015_03-position.csv" # Macでの実行時
    CONTACT_THRESHOLD_PIXELS = plot_social_network.CONTACT_THRESHOLD_PIXELS
    # ◆◆◆ 実行 ◆◆◆
    plot_distance_over_time(INPUT_CSV, CONTACT_THRESHOLD_PIXELS)