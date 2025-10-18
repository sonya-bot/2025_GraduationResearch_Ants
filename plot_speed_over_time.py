# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import calculate_thresholds as calc # 閾値計算用のモジュールをインポート

def plot_speed_over_time(input_filename, key_for_threshold , remove_outliers):
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

        # 外れ値の処理
    if remove_outliers:
        # print("外れ値の処理を実行します...")
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
        predefined_threshold = 100.0 # 閾値を設定 例: 50.0
        for col in speed_cols:
            outlier_count = df[df[col] > predefined_threshold].shape[0]
            if outlier_count > 0:
                print(f"  列 '{col}': {predefined_threshold:.2f} を超える {outlier_count} 個の外れ値を0に置換しました。")
                df.loc[df[col] > predefined_threshold, col] = 0 # 外れ値を0に置換
        print("外れ値の処理が完了しました。") 



    # 閾値データの修得(calculate_thresholds.pyからインポート)
    # main_runner.py から渡されたファイル名を使うように修正
    threshold_values, _ = calc.get_threshold_values(input_filename)
    if threshold_values is None:
        print("閾値の計算に失敗したため、プログラムを終了します。") 
        return
    else:
        if key_for_threshold is not None:
            selected_threshold = threshold_values.get(key_for_threshold)
            print(f"使用する閾値のキー: {key_for_threshold}, 値: {selected_threshold}")
            # for col in speed_cols:
            #     df.loc[df[col] <= selected_threshold, col] = 0 # 閾値以下を0に置換
            # 閾値を超えたフレーム番号を表示
            for col in speed_cols:
                threshold_frames = df[df[col] > selected_threshold]['position']
                if not threshold_frames.empty:
                    print(f"  列 '{col}': 閾値 {selected_threshold} を超えたフレーム番号: {threshold_frames.values}")
        else:
            print("閾値を使用しません 元のデータで描画します")

    # 'speed'を含む列を探して個体IDを特定
    speed_cols = [col for col in df.columns if col.startswith('speed_')]
    individual_ids = [col.replace('speed', '') for col in speed_cols]
    n_individuals = len(individual_ids)

    if n_individuals == 0:
        print("グラフ化するspeedデータが見つかりません。")
        return

    # グラフの総数は「合計グラフ(1) + 個体数」
    n_plots = n_individuals + 1

    # サブプロットのレイアウトを自動計算
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # 図全体のサイズを定義
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True)
    # グラフが1つの場合でも対応できるように、axesを1次元配列に変換
    axes_flat = np.atleast_1d(axes).flatten()

    # 図全体のタイトル
    fig.suptitle(f'Speed over Time (threshold: {key_for_threshold}, {selected_threshold})', fontsize=16)

    # 最初のグラフ(axes_flat[0])に、全個体の速さを重ねてプロット
    ax_overlay = axes_flat[0]
    ax_overlay.set_title('All Individuals')
    ax_overlay.set_xlabel('Frame')
    ax_overlay.set_ylabel('Speed (pixels/frame)') # 単位を追記
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
    # # ループの開始を n_individuals + 1に修正
    # for i in range(n_individuals + 1, len(axes_flat)):
    #     axes_flat[i].axis('off')

        # 閾値線はそのまま描画
    if selected_threshold is not None:
        for i in range(n_plots):
            axes_flat[i].axhline(y=selected_threshold, color='red', linestyle='--', linewidth=1.5)

    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].axis('off')

    # グラフの表示
    plt.show()

if __name__ == '__main__':
    # ◆◆◆ 設定 ◆◆◆
    INPUT_CSV = "d:/analysis_data/20251016_02/20251016_02-position_velocity.csv"
    # 使用する閾値 KEY を選択
    # 'q1', 'median_q2', 'q3', 'avg_half' などから閾値のキーを選択(calculate_thresholdsで計算されるもの)
    # しきい値を使用しない場合は None に設定
    # 外れ値を除去するかどうか (True: 除去する, False: 除去しない)
    REMOVE_OUTLIERS = True
    
    # 使用する閾値 KEY を選択
    KEY = "median_q2"
    
    plot_speed_over_time(INPUT_CSV, KEY, remove_outliers=REMOVE_OUTLIERS)
