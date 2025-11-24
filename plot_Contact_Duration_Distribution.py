# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import numpy.fft as fft 
import matplotlib.pyplot as plt
import platform
import os
import networkx as nx
import sys
from itertools import combinations 

def plot_contact_duration_distribution(position_csv_path, contact_threshold, fig_size, auto_save, use_loglog_plot):
    """
    位置データから、個体の「所属グループサイズ」ごとの持続時間分布を計算し、
    片対数グラフ（Y軸対数）で描画する。
    NetworkXを用いて連結成分分解を行い、排他的なグループサイズを判定する。
    """

# 1.データ入力
    try:
        #
        df_pos = pd.read_csv(position_csv_path)
        print(f"'{position_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return
    
# 2.個体IDの特定と個体ごとの反復処理
    x_cols = [col for col in df_pos.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]

    n_individuals = len(individual_ids)
    if n_individuals <= 1:
        print(f"エラー: 検出された個体数が {n_individuals} のため、グラフを作成できません。")
        print("プログラムを終了します")
        return
    
    print(f"{n_individuals} 個体を検出しました。グループサイズの持続時間を解析します...")

    # 持続時間の格納用辞書
    # Key: サイズ (1, 2, ..., N), Value: 持続時間(秒)のリスト
    duration_storage = {size: [] for size in range(1, n_individuals + 1)}

    # 個体の状態追跡用辞書
    # Key: 個体ID, Value: {'current_size': int, 'frame_count': int}
    tracker = {uid: {'current_size': 0, 'frame_count': 0} for uid in individual_ids}

    # 座標データをNumpy配列化して高速化 (shape: フレーム数 x 2)
    coords_dict = {}
    for uid in individual_ids:
        coords_dict[uid] = df_pos[[f'x{uid}', f'y{uid}']].to_numpy()

    total_frames = len(df_pos)

# 3.FPSの定義
    FPS = 2.0 # 1フレーム=1/2秒
    sample_spacing_second = 1.0 / FPS # 周波数計算用にサンプリング間隔(秒)を計算
    # 時間を時間(秒)から時間(分)に変更
    sample_spacing_minutes = sample_spacing_second / 60 
    print(f"  - サンプリング周波数: {FPS} Hz (サンプリング間隔: {sample_spacing_minutes} m)")

# 4.フレームごとのネットワーク解析と状態追跡
    print("フレームごとの解析を実行中...")
    # カウンターの初期化
    stats_trio_chain_frames = 0
    stats_trio_triangle_frames = 0
    
    for i in range(total_frames):
        # 4-1. ネットワーク構築 (距離判定)
        G = nx.Graph()
        G.add_nodes_from(individual_ids)
        
        # 全ペアの距離を計算し、閾値以下ならエッジを追加
        for j in range(n_individuals):
            for k in range(j + 1, n_individuals):
                id_a = individual_ids[j]
                id_b = individual_ids[k]
                
                pos_a = coords_dict[id_a][i]
                pos_b = coords_dict[id_b][i]
                
                dist = np.sqrt(np.sum((pos_a - pos_b)**2))
                
                if dist <= contact_threshold:
                    G.add_edge(id_a, id_b)

        if i == total_frames - 1:
            print(f" - ネットワーク(全 {total_frames} Frame)を構築しました。")

        # 4-2. 連結成分の特定と状態更新
        connected_components = list(nx.connected_components(G))
        
        # 個体ID -> 現在のグループサイズ のマッピングを作成
        current_frame_sizes = {}
        
        for component in connected_components:
            size = len(component)
            if size == 3:
                # この成分（グループ）の部分グラフを作成
                subgraph = G.subgraph(component)
                num_edges = subgraph.number_of_edges()
                
                if num_edges == 3:
                    # エッジが3本 = Triangle (全結合)
                    stats_trio_triangle_frames += 1
                    # print(f"Frame {i}: Triangle detected {component}") # デバッグ用
                elif num_edges == 2:
                    # エッジが2本 = Chain (鎖状)
                    stats_trio_chain_frames += 1
                    # print(f"Frame {i}: Chain detected {component}") # デバッグ用

                # グラフ描画用には、形状に関わらず「Size 3」として統合して記録
            for uid in component:
                current_frame_sizes[uid] = size

            if i == total_frames - 1:
                print(f" - 連結成分(全 {total_frames} Frame)を特定しました。")

        # 4-3. 各個体の状態更新と持続時間記録
        for uid in individual_ids:
            new_size = current_frame_sizes[uid]
            prev_size = tracker[uid]['current_size']
            count = tracker[uid]['frame_count']

            if new_size == prev_size:
                # 状態継続: カウントアップ
                tracker[uid]['frame_count'] += 1
            else:
                # 状態変化: 前の状態を記録 (初期状態0は除外)
                if prev_size > 0:
                    duration_sec = count / FPS
                    duration_storage[prev_size].append(duration_sec)
                
                # 新しい状態にリセット
                tracker[uid]['current_size'] = new_size
                tracker[uid]['frame_count'] = 1

                # ループ終了後の後処理 (最後の継続時間を記録)
    for uid in individual_ids:
        size = tracker[uid]['current_size']
        count = tracker[uid]['frame_count']
        if size > 0:
            duration_sec = count / FPS
            duration_storage[size].append(duration_sec)

    print(f" - 状態解析(全 {total_frames} Frame) が完了しました。")

# 5.グラフの描画

    for size in range(1, n_individuals + 1):
        durations = duration_storage[size]
        
        if len(durations) > 0:
            # 個別のFigureを作成
            plt.figure(figsize=fig_size)
            ax = plt.gca()

            label_text = f""
            if size == 1: label_text += "Isolated"
            elif size == 2: label_text += "Pair"
            elif size == 3: label_text += "Trio"
        
            
            # 対数スケールの設定
            if use_loglog_plot:
                ax.set_xlim(0.1,100)           
                ax.set_xscale('log')
                ax.set_xlabel('Duration (s) [Log Scale]', fontsize=12)
            else:
                ax.set_xlim(0,100)
                ax.set_xlabel('Duration (s)', fontsize=12)

            # ヒストグラム描画
            weights = np.ones_like(durations) / len(durations) * 100
            bins = np.logspace(np.log10(0.1), np.log10(100), 30) if use_loglog_plot else 30
            ax.hist(durations, bins=bins, weights=weights,
                    label=label_text,
                    alpha=0.7, edgecolor='black', histtype='bar', log=True)
    
            ax.set_ylim(0.1, 100)  # Y軸は対数スケールなので下限を0.1に設定
            ax.set_yscale('log')
            ax.set_ylabel('Frequency (%) [Log Scale]', fontsize=12)
            
            ax.set_title(f'Contact Duration Histogram: {label_text} (N={n_individuals})', fontsize=14)
            ax.grid(True, which="major", axis='y', linestyle='--', alpha=0.7)
            ax.grid(True, which="major", axis='x', linestyle=':', alpha=0.4)
            ax.legend(fontsize=10, loc='upper right')
            
            # コンソール出力
            mean_val = np.mean(durations)
            print(f" [Size {size}] Count: {len(durations)}, Mean: {mean_val:.2f}s")

            # 保存または表示
            if auto_save:
                output_filename = f"Contact_Duration_Distribution(N={n_individuals}).png"
                output_directory = os.path.dirname(position_csv_path)
                save_path = os.path.join(output_directory, output_filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f" - グラフを保存しました: {save_path}")
            else:
                print(" - グラフを表示します")
                plt.show()


if __name__ == "__main__":
    # 位置データの入力
    INPUT_CSV = "20251101_01"
    INPUT_POSITION_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position.csv"
    # 接触判定に使用するしきい値
    CONTACT_THRESHOLD = 50.0  # ピクセル単位の接触しきい値
    # グラフのサイズを指定
    FIG_SIZE = (10, 5)
    # グラフの自動保存設定
    AUTO_SAVE = False
    # 対数スケールの設定
    USE_LOGLOG_PLOT = True  # True: 両対数プロット, False: 半対数プロット

    # 関数を呼び出し
    plot_contact_duration_distribution(INPUT_POSITION_CSV, CONTACT_THRESHOLD, FIG_SIZE, AUTO_SAVE, USE_LOGLOG_PLOT)