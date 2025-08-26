# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import platform
import calculate_thresholds as calc # 閾値計算用のモジュールをインポート

def plot_speed_over_time(input_filename, key_for_threshold):
    # データの読み込み
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
    threshold_values, _ = calc.get_threshold_values(INPUT_CSV)
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

    # 'speed'を含む列を探して個体IDを特定
    speed_cols = [col for col in df.columns if col.startswith('speed_')]
    individual_ids = [col.replace('speed', '') for col in speed_cols]
    n_individuals = len(individual_ids)

    if n_individuals == 0:
        print("グラフ化するspeedデータが見つかりません。")
        return

    # 個体数に応じて、できるだけ正方形に近いレイアウトにする
    n_cols = int(np.ceil(np.sqrt(n_individuals + 1))) # +1は全個体グラフ用
    n_rows = (n_individuals + n_cols - 1) // n_cols
    
    # 図全体のサイズを定義
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True)
    # グラフが1つの場合でも対応できるように、axesを1次元配列に変換
    axes_flat = axes.flatten() if n_individuals > 1 else [axes]

    # 図全体のタイトル
    fig.suptitle(f'Speed over Time (threshold: {key_for_threshold}, {selected_threshold})', fontsize=16)

    # 最初のグラフ(axes_flat[0])に、全個体の速さを重ねてプロット
    ax_overlay = axes_flat[0]
    ax_overlay.set_title('All Individuals')
    ax_overlay.set_xlabel('Frame')
    ax_overlay.set_ylabel('Speed')
    ax_overlay.set_ylim(0, df[speed_cols].max().max())   # 縦軸の統一
    ax_overlay.grid(True, linestyle='--', alpha=0.6)

    # ループして、同じaxに色分けしてプロット
    for i_id in individual_ids:
        speed_col_name = f'speed{i_id}'
        ax_overlay.plot(df['position'], df[speed_col_name], linewidth=0.8, label=f'ID: {i_id}')

    # 個体数が多すぎない場合のみ凡例を表示
    if len(individual_ids) <= 10:
        ax_overlay.legend(fontsize='small')

    # 2番目以降のグラフに個別の速さをプロット
    for i, i_id in enumerate(individual_ids):
        # プロットする位置をi+1にずらす
        ax = axes_flat[i + 1] 
        
        speed_col_name = f'speed{i_id}'
        
        ax.plot(df['position'], df[speed_col_name], label=f'Speed of {i_id}')

        ax.set_title(f'Individual ID: {i_id}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Speed')
        ax.set_ylim(0, df[speed_cols].max().max())   # 縦軸の統一
        ax.grid(True, linestyle='--', alpha=0.6)

    # 余った描画領域を非表示にする
    # ループの開始を n_individuals + 1に修正
    for i in range(n_individuals + 1, len(axes_flat)):
        axes_flat[i].axis('off')

    # グラフの表示
    plt.show()

if __name__ == '__main__':
    # ◆◆◆ 設定 ◆◆◆
    INPUT_CSV = '/Users/sonya/Library/CloudStorage/OneDrive-HiroshimaCityUniversity/2025/UMATracker/datas/c00001(edit_2)-position-velocity.csv'
    # 使用する閾値 KEY を選択
    # 'q1', 'median_q2', 'q3', 'avg_half' などから閾値のキーを選択(calculate_thresholdsで計算されるもの)
    # しきい値を使用しない場合は None に設定
    KEY = "median_q2"
    plot_speed_over_time(INPUT_CSV, KEY)

