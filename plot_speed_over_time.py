# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator, LogLocator
import calculate_thresholds # 閾値計算用のモジュール

def data_input(velocity_csv_path):
    """
    入力データの読み込み、FPS定義、時間軸の計算を行う。
    """
# 1.入力データの読み込み
    try:
        df_vel = pd.read_csv(velocity_csv_path)
        print(f"'{velocity_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {velocity_csv_path}")
        return None, None, None, None

    # 出力先定義
    output_dir = os.path.dirname(velocity_csv_path)

# 2.FPSの定義
    FPS = 2.0 # 1フレーム=1/2秒
    total_frames = len(df_vel)
    frames = df_vel['position']
        
    # 時間計算
    time_minutes = frames / FPS / 60
    total_time_in_minutes = total_frames / FPS / 60
    
    print(f"時間軸 (秒) を計算しました (FPS={FPS}, 合計時間: {total_time_in_minutes:.2f} 分)。")

    # 個体IDの特定
    speed_cols = [col for col in df_vel.columns if col.startswith('speed_')]
    individual_ids = [col.replace('speed_', '') for col in speed_cols]

    return df_vel, output_dir, time_minutes, individual_ids


def calculate_speed(df_vel, individual_ids, remove_outliers):
    """
    速度データの数値変換と外れ値処理を行う。
    修正: 不要な閾値計算を削除し、引数を整理しました。
    """
# 3.速度データの列を特定
    speed_cols = [f'speed_{uid}' for uid in individual_ids]
    
# 4.速度データ列の数値をfloat変換
    for col in speed_cols:
        df_vel[col] = pd.to_numeric(df_vel[col], errors='coerce')
        df_vel[col].fillna(0, inplace=True)
    print(f"速度データ列を数値に変換しました。")

# 5.外れ値の処理
    if remove_outliers:
        predefined_threshold = 200.0 # 閾値設定
        for col in speed_cols:
            outlier_count = df_vel[df_vel[col] > predefined_threshold].shape[0]
            if outlier_count > 0:
                print(f"  列 '{col}': {predefined_threshold:.2f} を超える {outlier_count} 個の外れ値を処理しました。")
                df_vel.loc[df_vel[col] > predefined_threshold, col] = np.nan
        print("外れ値の処理が完了しました。") 
    
    return df_vel

def draw_graph(plot_data, x_data, fig_size, title_text, use_x_log, x_label, use_y_log, y_label, auto_save, save_path):
    """
    グラフの描画、設定、保存を一括で行う共通関数。
    """
# 6.グラフの描画
    fig, ax = plt.subplots(figsize=fig_size)
    
    # データをループしてプロット
    for label, y_values in plot_data.items():
        ax.plot(x_data, y_values, linewidth=0.8, label=label)

    # グラフの体裁
    ax.set_title(title_text, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 対数スケールの設定 (X軸)
    if use_x_log:
        ax.set_xscale('log')
        ax.set_xlabel(f"{x_label} (Log Scale)", fontsize=12)
    else:
        ax.set_xlabel(f"{x_label}", fontsize=12)
        # X軸の範囲設定
        ax.set_xlim(0, x_data.max())
        ax.set_xticks(np.arange(0, x_data.max() + 1, 20))

    # 対数スケールの設定 (Y軸)
    if use_y_log:
        ax.set_yscale('log')
        ax.set_ylabel(f"{y_label} (Log Scale)", fontsize=12)
        ax.yaxis.set_major_locator(LogLocator(numticks=10))
        ax.set_ylim(0.1, 200)

    else:
        ax.set_ylabel(f"{y_label}", fontsize=12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.set_ylim(0, 200)

    # 凡例
    if len(plot_data) <= 10:
        ax.legend(loc='upper right', fontsize='small')
        
    # 保存または表示
    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" - グラフを保存しました: {save_path}")
        plt.close()
    else:
        print(f" - グラフを表示します: {title_text}")
        plt.show()
        

def plot_speed_over_time(velocity_csv_path, remove_outliers, use_x_log, use_y_log, fig_size, auto_save):
    """
    速度の時系列データをプロットするメイン関数。
    """
# 1. データ読み込み & 前処理
    df_vel, output_dir, time_minutes, individual_ids = data_input(velocity_csv_path)
    if df_vel is None: return # 読み込み失敗時は終了
    df_vel = calculate_speed(df_vel, individual_ids, remove_outliers)
    

# 2. 全個体の速度データをプロット (Overlay)
    all_plot_data = {f'ID: {uid}': df_vel[f'speed_{uid}'] for uid in individual_ids}
    save_path_all = os.path.join(output_dir, "Speed_over_Time_(All).png")
    
    draw_graph(
        plot_data=all_plot_data,
        x_data=time_minutes,
        fig_size=fig_size,
        title_text='Speed over Time (All Individuals)',
        use_x_log=use_x_log, 
        x_label='Time (min)',
        use_y_log=use_y_log,
        y_label='Speed (pixels/frame)',
        auto_save=auto_save,
        save_path=save_path_all,
    )

# 3. 個別の速さグラフ (Individual)
    for i_id in individual_ids:
        ind_plot_data = {f'Speed of {i_id}': df_vel[f'speed_{i_id}']}
        save_path_ind = os.path.join(output_dir, f"Speed_over_Time_(ID_{i_id}).png")
        
        draw_graph(
            plot_data=ind_plot_data,
            x_data=time_minutes,
            fig_size=fig_size,
            title_text=f'Speed over Time Individual (ID:{i_id})',
            use_x_log=use_x_log,
            x_label='Time (min)',
            use_y_log=use_y_log,
            y_label='Speed',
            auto_save=auto_save,
            save_path=save_path_ind,
        )


if __name__ == '__main__':
# メイン処理
    INPUT_CSV = "20251030_01"
    INPUT_VELOCITY_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position_velocity.csv"
    
    REMOVE_OUTLIERS = True
    USE_Y_LOG = True
    USE_X_LOG = False
    FIG_SIZE = (6, 4)
    AUTO_SAVE = False

    plot_speed_over_time(INPUT_VELOCITY_CSV, REMOVE_OUTLIERS, USE_X_LOG, USE_Y_LOG, FIG_SIZE, AUTO_SAVE)