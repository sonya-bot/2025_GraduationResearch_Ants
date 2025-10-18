# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import calculate_thresholds as calc # 閾値計算用のモジュールをインポート

try:
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Meiryo'
    elif platform.system() == 'Darwin': # macOS
        plt.rcParams['font.family'] = 'Hiragino Sans'
    else: # Linux
        plt.rcParams['font.family'] = 'IPAexGothic'
except Exception as e:
    print(f"日本語フォントの設定中にエラーが発生しました: {e}")

def plot_histogram_dashboard(input_filename, key_for_threshold):
    """
    1. 全個体の速さの分布（積み上げヒストグラム）
    2. 個体ごとの速さの分布（ヒストグラム）
    これらを1枚の画像に出力する。
    """
    try:
        df = pd.read_csv(input_filename)
        print(f"'{input_filename}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {input_filename}")
        return
    
    # 速度データの列を特定
    speed_cols = [col for col in df.columns if col.startswith('speed_')]
    if not speed_cols:
        print("速度データの列が見つかりません。")
        return
    
        # 速度データ列の形(str)を数値(float)に変換
    for col in speed_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(0, inplace=True)
    print("速度データ列を数値に変換しました。")

# 閾値データの修得(calculate_thresholds.pyからインポート)
# 引数で受け取った input_filename を使うように修正
    threshold_values, _ = calc.get_threshold_values(input_filename)
    if threshold_values is None:
        print("閾値の計算に失敗したため、プログラムを終了します。")
        return
    else:
        if key_for_threshold is not None:
            selected_threshold = threshold_values.get(key_for_threshold)
            print(f"使用する閾値のキー: {key_for_threshold}, 値: {selected_threshold}")
            for col in speed_cols:
                df.loc[df[col] <= selected_threshold, col] = 0 # 閾値以下を0に置換
        else:
            print("閾値を使用しません 元のデータで描画します")

    # 'speed_'で始まる列名から個体IDを特定
    speed_cols = [col for col in df.columns if col.startswith('speed_')]
    individual_ids = [col.replace('speed_', '') for col in speed_cols]
    n_individuals = len(individual_ids)

    if not speed_cols:
        print("速さのデータの列が見つかりません。")
        return

    # グラフの総数は「合計グラフ(1) + 個体数」
    n_plots = n_individuals + 1

    # サブプロットのレイアウトを自動計算
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = (n_plots + n_cols - 1) // n_cols

    # 描画領域を作成
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True)
    axes_flat = axes.flatten() if n_plots > 1 else [axes]

    fig.suptitle(f'Speed_Distribution (threshold: {key_for_threshold}, {selected_threshold})', fontsize=16)

    # 1つ目のグラフを「積み上げヒストグラム」に変更
    ax_summary = axes_flat[0]
    
    # 全個体の速さデータをリストに格納（修正箇所）
    speed_data_list = []
    labels_list = []
    for col in speed_cols:
        if selected_threshold is not None:
            speeds = df[col][df[col] >= selected_threshold]
        else:
            speeds = df[col][df[col] > 0]
        
        speed_data_list.append(speeds)
        labels_list.append(f"ID {col.replace('speed_', '')}")

    # 積み上げヒストグラムを作成
    ax_summary.hist(speed_data_list, bins='auto', stacked=True, label=labels_list,log=True)
    # ax_summary.set_yscale('log')

    ax_summary.set_title('All_Individuals', fontsize=14)
    ax_summary.set_xlabel('Speed')
    ax_summary.set_ylabel('Frequency')
    ax_summary.grid(True, axis='y', linestyle='--', alpha=0.5)
    if n_individuals <= 10:
        ax_summary.legend(fontsize='small')
    # 

    # --- 2つ目以降のグラフ: 個体ごとの速さ分布 ---
    for i, i_id in enumerate(individual_ids):
        ax_hist = axes_flat[i + 1]
        
        if selected_threshold is not None:
            individual_speeds = df[f'speed_{i_id}'][df[f'speed_{i_id}'] >= selected_threshold]
        else:
            individual_speeds = df[f'speed_{i_id}'][df[f'speed_{i_id}'] > 0]
        
        if not individual_speeds.empty:
            ax_hist.hist(individual_speeds, bins=np.linspace(0, 40, 100), alpha=0.75, edgecolor='black', log=True)


        ax_hist.set_title(f'Individual ID: {i_id}', fontsize=12)
        ax_hist.set_xlabel('Speed')
        ax_hist.set_ylabel('Frequency')
        ax_hist.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 余った描画領域を非表示にする
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.show()

if __name__ == '__main__':
    # ◆◆◆ 設定 ◆◆◆
    INPUT_CSV = "d:/analysis_data/20251016_02/20251016_02-position_velocity.csv"
    KEY = "median_q2"

    plot_histogram_dashboard(INPUT_CSV, KEY)