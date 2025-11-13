#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import os

# 0.データの読み込み確認
def load_data(position_csv_path):
    try:
        df = pd.read_csv(position_csv_path)
        print(f"'{position_csv_path}'を正常に読み込みました。")
        return df
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return None
    
# 1. ワイド形式からロング形式への変換関数
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

# 2. MSD計算関数
def calculate_msd_from_wide_format(position_csv_path, max_lag_ratio=0.5):
    """
    ワイド形式の軌跡データを読み込み、変換してからMSDを計算する。
    元のファイルは変更しない。
    """
    try:
        df_wide = pd.read_csv(position_csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
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
            results[boid_id] = {'lags_frames': valid_lags, 'msd': valid_msd}

    return results

# 3. MSDプロット関数
def plot_msd(msd_results, position_csv_path, use_loglog_plot, fig_size, auto_save):
    """
    計算されたMSDをプロットする。
    """
    if not msd_results:
            print("プロットするデータがありません。")
            return
        
    boid_ids = list(msd_results.keys())
        
    for i, boid_id in enumerate(boid_ids):
        
        plt.figure(figsize=fig_size)
        ax = plt.gca() 
        
        result = msd_results.get(boid_id) 
        if not result: continue

        lag_times_frames = np.array(result['lags_frames'])
        msd_values = np.array(result['msd'])

        FPS = 2.0  # 1フレーム=1/2秒
        lag_time_sec = lag_times_frames / FPS
        lag_times_min = lag_time_sec / 60.0
        
        print(f"  ID {boid_id}: Lag time (min): {lag_times_min.min():.2f} - {lag_times_min.max():.2f}")

        if lag_times_min.size > 0:
            if use_loglog_plot:
                ax.loglog(lag_times_min, msd_values, 'o-', markersize=3)
            else:
                ax.semilogy(lag_times_min, msd_values, 'o-', markersize=3)

            if use_loglog_plot and len(lag_times_min) > 1 and len(msd_values) > 1:
                # 傾き2と傾き1の参考線を追加
                # 傾き2は二乗に比例する線
                slope_2 = msd_values[1] / (lag_times_min[1]**2) * (lag_times_min**2)
                ax.plot(lag_times_min, slope_2, 'b--', alpha=0.7, label='Slope 2 (Ballistic)')
                # 傾き1は一次に比例する線
                slope_1 = msd_values[1] / lag_times_min[1] * lag_times_min
                ax.plot(lag_times_min, slope_1, 'r--', alpha=0.7, label='Slope 1 (Diffusive)')
                ax.legend(fontsize=8, loc='upper left')
        
        ax.set_title(f'MSD({"Log-Log" if use_loglog_plot else "Semi-Log"})(ID:{boid_id})', fontsize=12)
        ax.set_xlabel('Lag Time τ (minutes)', fontsize=10)
        ax.set_ylabel('MSD(τ)', fontsize=10)
        ax.set_ylim(bottom=1e-2, top=1e10) #最大値は10の10乗まで対応
        ax.grid(True, which="both", ls="--")

        # グラフの自動保存設定
        if auto_save:
            output_filename = (f"MSD({'Log-Log' if use_loglog_plot else 'Semi-Log'})(ID_{boid_id}).png")
            output_directory = os.path.dirname(position_csv_path)
            save_path = os.path.join(output_directory, output_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" - MSDグラフを保存しました: {save_path}")
        else:
            print(f" - MSDグラフを表示します: ID {boid_id}")
            plt.show()

# メイン処理
if __name__ == '__main__':
    # 入力ファイル名の指定
    INPUT_CSV = "20251030_02" #日時の指定だけで良い
    INPUT_POSITION_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position.csv"

    # 対数表示を行うかどうか
    USE_LOGLOG_PLOT = True  # True: 両対数プロット, False: 半対数プロット

    # グラフのサイズを指定
    FIG_SIZE = (10, 5)

    # グラフの自動保存
    AUTO_SAVE = False  # True: 自動保存, False: 表示のみ

    # 関数を順に実行
    df = load_data(INPUT_POSITION_CSV)
    if df is not None:
        msd_results = calculate_msd_from_wide_format(INPUT_POSITION_CSV, max_lag_ratio=0.5)
        plot_msd(msd_results, INPUT_POSITION_CSV, USE_LOGLOG_PLOT, FIG_SIZE, AUTO_SAVE)