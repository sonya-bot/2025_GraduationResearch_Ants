#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

# ◆◆◆ 設定箇所 ◆◆◆

# 分析対象のファイル名
FILENAME = '/Users/sonya/Library/CloudStorage/OneDrive-HiroshimaCityUniversity/2025/boids_trajectory.csv'

# グラフの種類をここで切り替えます
# True:  両対数グラフ (傾きで拡散の種類を分析するのに適しています)
# False: 片対数グラフ (縦軸のみ対数。移動距離の大きさの変化を見やすいです)
USE_LOGLOG_PLOT = True

# 【文字化け対策】日本語フォントを設定
try:
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Meiryo'
    elif platform.system() == 'Darwin': # macOS
        plt.rcParams['font.family'] = 'Hiragino Sans'
    else: # Linux
        plt.rcParams['font.family'] = 'IPAexGothic'
except Exception as e:
    print(f"日本語フォントの設定中にエラーが発生しました: {e}")
    print("グラフの日本語が文字化けする可能性があります。")


def calculate_boids_msd(filename, max_lag_ratio=0.5):
    """
    Boidsの軌跡データを読み込み、各boidのMSD（Mean Squared Displacement）を計算する。
    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {filename}")
        return None
    # 'id' 列が存在するか確認
    if 'id' not in df.columns:
        print(f"エラー: ファイル '{filename}' に 'id' 列が見つかりません。")
        return None


    boid_ids = df['id'].unique()
    msd_results = {}

    for boid_id in boid_ids:
        boid_df = df[df['id'] == boid_id].sort_values('frame')
        positions = boid_df[['position_x', 'position_y']].values
        n_steps = len(positions)
        
        max_lag = int(n_steps * max_lag_ratio)
        if max_lag <= 1:
            continue

        lags = np.arange(1, max_lag)
        msds = []

        for lag in lags:
            diff = positions[lag:] - positions[:-lag]
            squared_disp = np.sum(diff**2, axis=1)
            msds.append(np.mean(squared_disp))
        
        msd_results[boid_id] = (lags, np.array(msds))

    return msd_results

# --- メインの描画処理 ---

# boidごとのMSDデータを計算
all_boids_msd = calculate_boids_msd(FILENAME)

if all_boids_msd and len(all_boids_msd) > 0:
    # --- START MODIFICATION ---
    # グラフをグリッド表示するための設定
    n_boids = len(all_boids_msd)
    n_cols = 5  # 1行あたりのグラフの数
    # 必要な行数を計算 (例: 12個のboidなら 12 / 5 = 2.4 -> 3行)
    n_rows = (n_boids + n_cols - 1) // n_cols

    # グリッド状の描画領域を作成 (figsizeで全体のサイズを調整)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    
    # axesが常に2次元配列になるように調整 (boidが5個以下の場合に対応)
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # 描画領域を1次元化してループしやすくする
    axes_flat = axes.flatten()

    # 各boidのデータをそれぞれの描画領域にプロット
    for i, (boid_id, (lag_times, msd_values)) in enumerate(all_boids_msd.items()):
        ax = axes_flat[i]  # i番目の描画領域を選択

        if lag_times is not None and msd_values is not None and len(lag_times) > 0:
            # グラフの種類に応じてプロット
            if USE_LOGLOG_PLOT:
                ax.loglog(lag_times, msd_values, 'o-', markersize=3, alpha=0.8, linewidth=1.5)
            else:
                ax.semilogy(lag_times, msd_values, 'o-', markersize=3, alpha=0.8, linewidth=1.5)

            # 両対数グラフの場合のみ、参照線を描画
            if USE_LOGLOG_PLOT and len(lag_times) > 1 and len(msd_values) > 1:
                # 参照線1: 傾き1 (通常拡散)
                line1 = msd_values[1] / lag_times[1] * lag_times
                ax.plot(lag_times, line1, 'r--', alpha=0.7, label='傾き 1')
                # 参照線2: 傾き2 (バリスティックな動き)
                line2 = msd_values[1] / (lag_times[1]**2) * (lag_times**2)
                ax.plot(lag_times, line2, 'b--', alpha=0.7, label='傾き 2')
                ax.legend(fontsize=8)
        
        # 各グラフの体裁を設定
        ax.set_title(f'Boid ID: {boid_id}', fontsize=12)
        ax.set_xlabel('ラグタイム τ (frame)', fontsize=10)
        ax.set_ylabel('MSD(τ)', fontsize=10)
        ax.grid(True, which="both", ls="--")

    # boidの数に応じて余った描画領域を非表示にする
    for i in range(n_boids, len(axes_flat)):
        axes_flat[i].axis('off')

    # 全体のタイトルを設定
    if USE_LOGLOG_PLOT:
        fig.suptitle('Boidごとの平均二乗変位 (両対数グラフ)', fontsize=16)
    else:
        fig.suptitle('Boidごとの平均二乗変位 (片対数グラフ)', fontsize=16)

    # グラフが重ならないようにレイアウトを自動調整
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # suptitleとの重なりを避ける

    # グラフをファイルに保存
    # output_filename = 'boids_msd_plot_grid.png'
    # plt.savefig(output_filename, dpi=300)
    # print(f"グラフを '{output_filename}' として保存しました。")

    # グラフを表示
    plt.show()

else:
    print("描画するデータが見つかりませんでした。")

# --- END MODIFICATION ---