# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import os
from itertools import combinations
import calculate_thresholds as calc
import plot_social_network

def plot_distance_over_time(position_csv_path, contact_threshold, use_log_scale, fig_size, auto_save):
    """
    個体の位置データから、全ての個体ペア間の距離の時系列変化を棒グラフで描画する。
    接触しきい値を下回った時点を赤い点で示す。

    Args:
        input_filename (str): 位置情報(x, y)が含まれるCSVファイル名。
        contact_threshold (float): 2個体が「接触している」と判断する最大距離（ピクセル）。
    """
# 1.データの読み込み
    try:
        df = pd.read_csv(position_csv_path)
        print(f"'{position_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return
    
# 2. 個体IDの特定と個体数のチェック
    x_cols = [col for col in df.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]
    n_individuals = len(individual_ids)
    
    if n_individuals <= 1:
        print(f"エラー: 検出された個体数が {n_individuals} のため、距離グラフを作成できません。")
        return
    
    print(f"{n_individuals} 個体を対象に、全ペアの距離変化グラフを作成します。")

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
    
# 4. 全ての個体のペアについてループ処理
    pair_combinations = list(combinations(individual_ids, 2))
    
    for id1, id2 in pair_combinations:
        # 4. ペア間のユークリッド距離を全フレームにわたって計算
        pos1 = df[[f'x{id1}', f'y{id1}']].to_numpy()
        pos2 = df[[f'x{id2}', f'y{id2}']].to_numpy()
        distances = np.sqrt(np.sum((pos1 - pos2)**2, axis=1))
        
        # 5. グラフの描画
        plt.figure(figsize=fig_size)
        ax = plt.gca() # 現在のAxesを取得

        # ▼ 修正: Y軸の対数表示ロジックを修正
        # 最初にY軸の上限を計算
        y_axis_top_limit = max(distances.max() * 1.1, 100) # 1.1倍のマージン
        
        if use_log_scale:
                print("縦軸を対数表示に設定します。")
                ax.set_yscale('log')
                ax.set_ylim(bottom=0.1, top=y_axis_top_limit)
        else:
                print("縦軸を通常表示に設定します。")
                ax.set_ylim(bottom=0, top=y_axis_top_limit)
        # ▲ 修正 ▲
        
        # ▼ 修正: X軸に frames ではなく time_minutes を使用
        # 棒グラフを作成
        # 注意: 棒グラフ(bar)はデータ点が多い(>1000)と非常に重くなるため、
        # 本来は ax.plot(time_minutes, distances, ...) の折れ線グラフを推奨
        ax.bar(time_minutes, distances, color='lightblue', width=(1.0/FPS/60.0), # 棒の幅を1フレーム相当に
               label=f'Distance between (ID:{id1} , ID:{id2})')
        
        # 6. 接触点の特定とプロット
        contact_indices = (distances <= contact_threshold)
        contact_times = time_minutes[contact_indices] # フレーム番号ではなく、対応する「時間(分)」を取得
        
        if not contact_times.empty:
            plot_y = np.zeros_like(contact_times)
            # ▼ 修正: X軸に contact_times を使用
            ax.plot(contact_times, plot_y, 'o', color='red', markersize=3, 
                    label=f'Contact (<= {contact_threshold} px)')
        
        # グラフの体裁を整える
        # ▼X軸ラベルと範囲設定 (Speed_over_time と統一)
        ax.set_xlabel(f'Time ({total_time_in_minutes:.2f} min)', fontsize=12)
        ax.set_ylabel('Distance (pixels)', fontsize=12)
        ax.set_title(f'Distance over Time (ID:{id1} , {id2})', fontsize=14)
        
        # X軸の範囲と目盛りを設定 (0-180分, 20分間隔)
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 20))

        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # ax.set_ylim(0, distances.max() * 1.1) # <- 削除 (Y軸バグの原因)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
        
        plt.tight_layout()
        
        # ▼ 修正: auto_save ロジックと plt.show() の分離
        if auto_save:
            output_filename = f'Distance_over_Time (ID:{id1} , {id2}).png'
            output_directory = os.path.dirname(position_csv_path)
            save_path = os.path.join(output_directory, output_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" - 距離変化グラフを保存しました: {save_path}")
        else:
            print(f" - 距離変化グラフを表示します: ID {id1} , {id2}")
            plt.show()


# メイン処理
if __name__ == '__main__':
    INPUT_CSV = "20251030_02"
    INPUT_POSITION_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position.csv"
    CONTACT_THRESHOLD = 50.0
    # 縦軸を対数表記するかどうか
    USE_LOG_SCALE = True
    # グラフのサイズを指定
    FIG_SIZE = (10, 5)
    # グラフの自動保存
    AUTO_SAVE = False

    plot_distance_over_time(INPUT_POSITION_CSV, CONTACT_THRESHOLD, USE_LOG_SCALE, FIG_SIZE, AUTO_SAVE)