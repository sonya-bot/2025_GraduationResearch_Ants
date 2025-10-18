#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

# ◆◆◆ 設定箇所 ◆◆◆

# 分析対象のファイル名
FILENAME =  "d:/analysis_data/20251016_02/20251016_02-position.csv"

# グラフの種類
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

def convert_wide_to_long(df_wide):
    """
    ワイド形式のDataFrameをロング形式に変換する（メモリ上でのみ処理）。
    """
    df_temp = df_wide.copy() # 元のDataFrameを汚染しないようにコピー
    df_temp.rename(columns={'position': 'frame'}, inplace=True)
    
    column_mapping = {col: f"{col[0]}_{col[1:]}" for col in df_temp.columns if col not in ['frame']}
    df_temp.rename(columns=column_mapping, inplace=True)
    
    df_long = pd.wide_to_long(
        df_temp,
        stubnames=['x', 'y'],
        i='frame',
        j='id',
        sep='_',
        suffix='\\d+'
    ).reset_index()
    
    df_long.rename(columns={'x': 'position_x', 'y': 'position_y'}, inplace=True)
    return df_long


def calculate_msd_from_wide_format(filename, max_lag_ratio=0.5):
    """
    ワイド形式の軌跡データを読み込み、変換してからMSDを計算する。
    元のファイルは変更しない。
    """
    try:
        df_wide = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {filename}")
        return None

    # メモリ上でデータ形式の変換処理を呼び出す
    df = convert_wide_to_long(df_wide)
    
    results = {}
    boid_ids = df['id'].unique()
    n_frames = df['frame'].max()
    max_lag = int(n_frames * max_lag_ratio)
    if max_lag == 0: max_lag = 1

    for boid_id in boid_ids:
        boid_df = df[df['id'] == boid_id].sort_values(by='frame').set_index('frame')
        all_frames = pd.DataFrame(index=np.arange(boid_df.index.min(), boid_df.index.max() + 1))
        boid_df = boid_df.reindex(all_frames.index).interpolate(method='linear')
        coords = boid_df[['position_x', 'position_y']].to_numpy()
        
        msd = []
        lag_times = list(range(1, max_lag))

        for lag in lag_times:
            diff = coords[lag:] - coords[:-lag]
            squared_disp = np.sum(diff**2, axis=1)
            if len(squared_disp) > 0:
                msd.append(np.mean(squared_disp))
            else:
                msd.append(np.nan)
        
        valid_lags = [lag for i, lag in enumerate(lag_times) if not np.isnan(msd[i])]
        valid_msd = [val for val in msd if not np.isnan(val)]
        
        if valid_lags:
             results[boid_id] = {'lags': valid_lags, 'msd': valid_msd}

    return results


def plot_msd(msd_results, source_filename):
    """
    計算されたMSDをプロットする。
    """
    if not msd_results:
        print("プロットするデータがありません。")
        return

    boid_ids = list(msd_results.keys())
    n_boids = len(boid_ids)
    
    n_cols = 3
    n_rows = (n_boids + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True)
    axes_flat = axes.flatten()

    fig.suptitle(f'MSD Plot ({ "Log-Log" if USE_LOGLOG_PLOT else "Semi-Log" })', fontsize=16)

    for i, boid_id in enumerate(boid_ids):
        ax = axes_flat[i]
        result = msd_results.get(boid_id)
        if not result: continue

        lag_times = result['lags']
        msd_values = result['msd']

        if lag_times:
            if USE_LOGLOG_PLOT:
                ax.loglog(lag_times, msd_values, 'o-', markersize=3)
            else:
                ax.semilogy(lag_times, msd_values, 'o-', markersize=3)

            if USE_LOGLOG_PLOT and len(lag_times) > 1 and len(msd_values) > 1:
                line1 = msd_values[1] / lag_times[1] * np.array(lag_times)
                ax.plot(lag_times, line1, 'r--', alpha=0.7, label='Slope 1')
                line2 = msd_values[1] / (lag_times[1]**2) * (np.array(lag_times)**2)
                ax.plot(lag_times, line2, 'b--', alpha=0.7, label='Slope 2')
                ax.legend(fontsize=8)
        
        ax.set_title(f'Individual ID: {boid_id}', fontsize=12)
        ax.set_xlabel('Lag Time τ (frame)', fontsize=10)
        ax.set_ylabel('MSD(τ)', fontsize=10)
        ax.grid(True, which="both", ls="--")

    for i in range(n_boids, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.show()
        
    # output_filename = "msd_plot_from_c00001.png"
    # plt.savefig(output_filename, dpi=300)
    # print(f"グラフを '{output_filename}' として保存しました。")


if __name__ == '__main__':
    msd_data = calculate_msd_from_wide_format(FILENAME)
    if msd_data:
        plot_msd(msd_data, FILENAME)