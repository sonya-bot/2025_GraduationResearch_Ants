# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import os
import calculate_thresholds as calc # 閾値計算用のモジュールをインポート
from matplotlib.ticker import MaxNLocator, LogLocator


def plot_histogram_dashboard(velocity_csv_path, remove_outliers, remove_threshold, velocity_threshold, use_log_scale, fig_size, auto_save):
    """
    1. 全個体の速さの分布（積み上げヒストグラム）
    2. 個体ごとの速さの分布（ヒストグラム）
    これらを1枚の画像に出力する。
    """
# 1.データの読み込み
    try:
        df = pd.read_csv(velocity_csv_path)
        print(f"'{velocity_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {velocity_csv_path}")
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
        predefined_threshold = remove_threshold # 閾値を設定 例: 50.0
        for col in speed_cols:
            outlier_count = df[df[col] > predefined_threshold].shape[0]
            if outlier_count > 0:
                print(f" -  列 '{col}': {predefined_threshold:.2f} を超える {outlier_count} 個の外れ値を処理しました。")
                df.loc[df[col] > predefined_threshold, col] = np.nan # 外れ値を NaN に置換
        print("外れ値の処理が完了しました。") 

# 3.閾値の計算
    threshold_values, _ = calc.get_threshold_values(velocity_csv_path)
    selected_threshold = None # 閾値変数を初期化
    if threshold_values is None:
        print("閾値の計算に失敗したため、処理を続行します（閾値線なし）。") 
    else:
        if velocity_threshold is not None:
            selected_threshold = threshold_values.get(velocity_threshold)
            print(f"使用する閾値のキー: {velocity_threshold}, 値: {selected_threshold}")
        else:
            print("閾値を使用しません（閾値線なし）。")

# 4.全個体の速度分布データをプロット

    # グラフの描画サイズを定義
    plt.figure(figsize=fig_size)
    ax_summary = plt.gca()
    
    speed_data_list = []
    labels_list = []
    
    # ヒストグラムのビンの設定 (0から最大値まで)
    max_speed_for_hist = 50 # X軸の表示上限
    bins = np.linspace(0, max_speed_for_hist, 100) # 0もヒストグラムに含める

    for col in speed_cols:
        speeds = df[col].dropna() 
        speed_data_list.append(speeds)
        labels_list.append(f"ID:{col.replace('speed_', '')}")

    # Y軸を対数表示(use_log_scale)にするか
    ax_summary.hist(speed_data_list, bins=bins, stacked=True, label=labels_list, log=use_log_scale)

    ax_summary.set_title(f'Speed Distribution (All Individuals)')
    ax_summary.set_xlabel('Speed (pixels/frame)')
    ax_summary.set_ylabel(f'Frequency{" (Log Scale)" if use_log_scale else ""}')
    ax_summary.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax_summary.set_xlim(0, 60) 
    ax_summary.set_xticks(np.arange(0, 61, 20))
    if n_individuals <= 10:
        ax_summary.legend(fontsize='small')
    
    # 閾値を凡例として表示
    if selected_threshold is not None:
        ax_summary.axvline(x=selected_threshold, color='red', linestyle='--', linewidth=1.5, 
                           label=f'Threshold ({velocity_threshold}): {selected_threshold:.2f}')
    
    # 凡例をまとめて右上に表示 (個体数が多いと凡例が大きくなるため fontsize を 'small' に)
    ax_summary.legend(loc='upper right', fontsize='small')
    
    if auto_save:
        output_filename = "Speed_Distribution(All).png"
        output_directory = os.path.dirname(velocity_csv_path)
        save_path = os.path.join(output_directory, output_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" - 全体ヒストグラムを保存しました: {save_path}")
    else:
        print(f" - 全体ヒストグラムを表示します")
        plt.show()

    # 個体ごとの速さ分布 (個体ごとに個別のウィンドウ) 
    for i_id in individual_ids:
        plt.figure(figsize=fig_size)
        ax_hist = plt.gca()
        
        individual_speeds = df[f'speed_{i_id}'].dropna()
        
        if not individual_speeds.empty:
            ax_hist.hist(individual_speeds, bins=bins, alpha=0.75, edgecolor='black', log=use_log_scale)

        ax_hist.set_title(f'Speed Distribution (Individual ID:{i_id})')
        ax_hist.set_xlabel('Speed (pixels/frame)')
        ax_hist.set_ylabel(f'Frequency{" (Log Scale)" if use_log_scale else ""}')
        ax_hist.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax_hist.set_xlim(0, 60)
        ax_hist.set_xticks(np.arange(0, 61, 20))

        # 閾値を凡例として表示
        if selected_threshold is not None:
            ax_hist.axvline(x=selected_threshold, color='red', linestyle='--', linewidth=1.5, 
                            label=f'Threshold ({velocity_threshold}): {selected_threshold:.2f}')
        
            # 凡例をまとめて右上に表示 (個体数が多いと凡例が大きくなるため fontsize を 'small' に)
            ax_hist.legend(loc='upper right', fontsize='small')
        

        if auto_save:
            output_filename = f"Speed_Distribution(ID_{i_id}).png"
            output_directory = os.path.dirname(velocity_csv_path)
            save_path = os.path.join(output_directory, output_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" - 個別ヒストグラムを保存しました: {save_path}")
        else:
            print(f" - 個別ヒストグラムを表示します: ID {i_id}")
            plt.show()

# メイン処理
if __name__ == '__main__':
    # データの入力ファイル
    INPUT_CSV = "20251030_02"
    INPUT_VELOCITY_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position_velocity.csv"
    # 外れ値を除去するかどうか (True: 除去する, False: 除去しない)
    REMOVE_OUTLIERS = True
    REMOVE_THRESHOLD = 200.0  # 外れ値とみなす速度の閾値 (ピクセル/フレーム)
    # 使用する閾値 KEY を選択
    # 'q1', 'median_q2', 'q3', 'avg_half' などから閾値のキーを選択(calculate_thresholdsで計算されるもの)
    # 数値を指定して直接閾値を設定することも可能
    # しきい値を使用しない場合は None に設定
    VELOCITY_THRESHOLD = "avg_half" 
    # 縦軸を対数表示するかどうか (True: 対数表示, False: 通常表示)
    USE_LOG_SCALE = True
    # グラフのサイズを指定
    FIG_SIZE = (10, 5) # 横長のグラフ
    # グラフの自動保存
    AUTO_SAVE = False

    plot_histogram_dashboard(INPUT_VELOCITY_CSV, REMOVE_OUTLIERS, REMOVE_THRESHOLD, VELOCITY_THRESHOLD, USE_LOG_SCALE, FIG_SIZE, AUTO_SAVE)