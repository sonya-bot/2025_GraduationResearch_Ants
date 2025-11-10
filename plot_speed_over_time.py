# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import calculate_thresholds as calc # 閾値計算用のモジュールをインポート
from matplotlib.ticker import MaxNLocator, LogLocator

def plot_speed_over_time(input_filename, remove_outliers, use_log_scale, fig_size, auto_save):
# 1.データの読み込み
    try:
        df = pd.read_csv(input_filename)
        print(f"'{input_filename}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {input_filename}")
        return

# 2.個体IDの特定と個体ごとの反復処理
    # 速度データの列を特定
    speed_cols = [col for col in df.columns if col.startswith('speed_')]
    individual_ids = [col.replace('speed_', '') for col in speed_cols]
    n_individuals = len(individual_ids)
    if not speed_cols:
        print("速度データの列が見つかりません。")
        return
    
    # 速度データ列の形(str)を数値(float)に変換
    for col in speed_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(0, inplace=True)
    print("速度データ列を数値に変換しました。")

    # 外れ値の処理
    if remove_outliers:
        print("外れ値の処理を実行します...")
        # # 各速度列に対して外れ値を検出し、0に置換
        # for col in speed_cols:
        #     non_zero_speeds = df[df[col] > 0][col]
        #     if not non_zero_speeds.empty:
        #         outlier_threshold = non_zero_speeds.quantile(0.999) 
        #         outlier_count = df[df[col] > outlier_threshold].shape[0]
        #         if outlier_count > 0:
        #             print(f"  列 '{col}': {outlier_threshold:.2f} を超える {outlier_count} 個の外れ値を0に置換しました。")
        #             df.loc[df[col] > outlier_threshold, col] = 0
        #     else:
        #         print(f"  列 '{col}': 速度が0以外のデータがないため、外れ値処理をスキップしました。")
        # print("外れ値の処理が完了しました。")

        # 外れ値のしきい値を任意の速度に設定する場合、こちらを使用
        predefined_threshold = 200.0 # 閾値を設定 例: 50.0
        for col in speed_cols:
            outlier_count = df[df[col] > predefined_threshold].shape[0]
            if outlier_count > 0:
                print(f" -  列 '{col}': {predefined_threshold:.2f} を超える {outlier_count} 個の外れ値を0に置換しました。")
                df.loc[df[col] > predefined_threshold, col] = np.nan # 外れ値を NaN に置換
        print("外れ値の処理が完了しました。") 

    
# 3.グラフの横軸となる時間(秒)への変換
    FPS = 2.0 # 1フレーム=1/2秒
    total_frames = len(df)
    frames = df['position']
        
    # position (フレーム番号) を 時間 (秒) に変換
    # (変数 'frames' は 'df_pos['position']' と同義)
    time_seconds = frames / FPS
    # 時間を時間(秒)から時間(分)に変更
    time_minutes = time_seconds / 60
    # ▼ 修正：合計時間（秒）を計算して表示
    total_time_in_seconds = total_frames / FPS
    total_time_in_minutes = total_time_in_seconds / 60
    
    print(f"時間軸 (秒) を計算しました (FPS={FPS}, 合計時間: {total_time_in_minutes:.2f} 分)。")


# 4.全個体の速度データをプロット

    # グラフの標準サイズを定義
    plt.figure(figsize=fig_size)
    ax_overlay = plt.gca() # 現在のAxesを取得

    # 対数表示の場合のY軸上限を計算
    # 表示したい上限 (100) とデータの最大値のうち、大きい方を採用
    max_speed_in_data = df[speed_cols].max().max() * 1.1
    y_axis_top_limit = max(max_speed_in_data, 100)

    # 対数表示の設定
    if use_log_scale:
            print("縦軸を対数表示に設定します。")
            ax_overlay.set_yscale('log')
            ax_overlay.set_ylim(bottom=0.1, top=y_axis_top_limit)
            ax_overlay.yaxis.set_major_locator(LogLocator(numticks=10))
    else:
            print("縦軸を通常表示に設定します。")
            ax_overlay.set_ylim(bottom=0, top=y_axis_top_limit)
            ax_overlay.yaxis.set_major_locator(MaxNLocator(nbins=10))
    
    ax_overlay.set_title(f'Speed over Time (All Individuals)')
    # X軸のラベルを 'Frame' から 'Time (min)' に変更
    ax_overlay.set_xlabel(f'Time ({total_time_in_minutes:.2f} min)') 
    ax_overlay.set_ylabel('Speed (pixels/frame)')
    ax_overlay.grid(True, linestyle='--', alpha=0.6)
    print("全個体の速度データをプロットします...")

    # ループして、同じaxに色分けしてプロット
    for i_id in individual_ids:
        speed_col_name = f'speed_{i_id}'
        # X軸を 'df['position']' から 'time_minutes' に変更
        ax_overlay.plot(time_minutes, df[speed_col_name], linewidth=0.8, label=f'ID: {i_id}')

    # 凡例
    if len(individual_ids) <= 10:
        ax_overlay.legend(fontsize='small')

    # グラフの自動保存設定
    if auto_save:
        save_path = "Speed_over_Time_(All).png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" - 全体グラフを保存しました: {save_path}")
    else:
        print(f" - 全体グラフを表示します。")
        plt.show()

    # 個別の速さグラフ (個体ごとに新しいウィンドウ)
    for i_id in individual_ids:
        plt.figure(figsize=fig_size)
        ax = plt.gca() # 新しいFigureのAxesを取得
        
        speed_col_name = f'speed_{i_id}'
        
        # X軸を 'df['position']' から 'time_minutes' に変更
        ax.plot(time_minutes, df[speed_col_name], label=f'Speed of {i_id}')

        ax.set_title(f'Speed over Time Individual (ID:{i_id})')
        # X軸のラベルを 'Frame' から 'Time (min)' に変更
        ax.set_xlabel(f'Time')
        ax.set_ylabel('Speed')
        ax.grid(True, linestyle='--', alpha=0.6)

        # 対数表示の設定
        if use_log_scale:
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.1, top=y_axis_top_limit)
            ax.yaxis.set_major_locator(LogLocator(numticks=10))
        else:
            ax.set_ylim(bottom=0, top=y_axis_top_limit)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

        # グラフの自動保存設定
        if auto_save:
            save_path = f"Speed_over_Time_(ID_{i_id}).png"
            plt.savefig(save_path)
            print(f" - 速度変化グラフを保存しました: {save_path}")
        else:
            print(f" - 速度変化グラフを表示します: ID {i_id}")
        plt.show()

# メイン処理
if __name__ == '__main__':
    INPUT_CSV = "20251016_02"
    INPUT_VELOCITY_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position_velocity.csv"
    # 使用する閾値 KEY を選択
    # 'q1', 'median_q2', 'q3', 'avg_half' などから閾値のキーを選択(calculate_thresholdsで計算されるもの)
    # しきい値を使用しない場合は None に設定
    # 外れ値を除去するかどうか (True: 除去する, False: 除去しない)
    REMOVE_OUTLIERS = True
    # 縦軸を対数表示するかどうか (True: 対数表示, False: 通常表示)
    USE_LOG_SCALE = True
    # グラフのサイズを指定
    FIG_SIZE = (10, 5) # 横長のグラフ
    # グラフの自動保存
    AUTO_SAVE = False

    plot_speed_over_time(INPUT_VELOCITY_CSV, remove_outliers=REMOVE_OUTLIERS, use_log_scale=USE_LOG_SCALE, fig_size=FIG_SIZE, auto_save=AUTO_SAVE)
