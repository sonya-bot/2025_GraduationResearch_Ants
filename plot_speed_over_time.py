# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from matplotlib.ticker import MaxNLocator
import calculate_thresholds  # アップロードされた閾値計算モジュール

# 1.計算・前処理関数
def data_input(velocity_csv_path, remove_outliers):
    """
    入力データの読み込み、FPS定義、時間軸の計算を行う。
    """
# 1.入力データの読み込み
    try:
        df_vel = pd.read_csv(velocity_csv_path)
        print(f"'{velocity_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {velocity_csv_path}")
        return None, None, None

# 2.個体IDの特定
    speed_cols = [col for col in df_vel.columns if col.startswith('speed_')]
    individual_ids = [col.replace('speed_', '') for col in speed_cols]

#3.数値への変換
    for col in speed_cols:
        df_vel[col] = pd.to_numeric(df_vel[col], errors='coerce')
        df_vel[col].fillna(0, inplace=True)
    print(f" - 速度データ列を数値に変換しました。")

# 4.平滑化処理(Hayashi,2012)に従う
    # 60秒間 (FPS x 60) の移動平均をとる
    fps = 2.0 # 1フレーム=1/2秒
    window_sec = 60 # 平滑化ウィンドウ (秒)
    window_frames = int(window_sec * fps)
    if window_frames > 1:
        # print(f"  - {window_sec}秒間 ({window_frames} frames) の移動平均を適用")
        for col in speed_cols:
            df_vel[col] = df_vel[col].rolling(window=window_frames, center=True, min_periods=1).mean().fillna(0)

# 5.外れ値の処理
    if remove_outliers:
        predefined_threshold = 200.0 # 閾値設定
        for col in speed_cols:
            outlier_count = df_vel[df_vel[col] > predefined_threshold].shape[0]
            if outlier_count > 0:
                print(f"  列 '{col}': {predefined_threshold:.2f} を超える {outlier_count} 個の外れ値を処理しました。")
                df_vel.loc[df_vel[col] > predefined_threshold, col] = np.nan
        print("外れ値の処理が完了しました。") 


# 5.時間軸の計算
    time_minutes = df_vel.index / fps / 60.0  # 分単位の時間軸
    total_time_in_minutes = len(df_vel) / fps / 60.0

    print(f" - 時間軸 (秒) を計算しました (FPS={fps}, 合計時間:{total_time_in_minutes:.2f}分)。")


    return df_vel, individual_ids, time_minutes, fps

def get_individual_thresholds(velocity_csv_path, velocity_threshold):
    """
    指定された速度閾値キーに基づいて、個体ごとの閾値を取得する関数。
    """
    _, individual_results = calculate_thresholds.get_threshold_values(velocity_csv_path)

    threshold_map = {}
    if individual_results:
        for res in individual_results:
            threshold_map[res['id']] = res.get(velocity_threshold, 0)
    
    print(f" - 個体ごとの速度閾値を取得しました (Key:{velocity_threshold})。")

    return threshold_map

# 2.グラフ描画関数
def get_layout_params(n_plots, single_fig_size):
    """
    プロット数に応じて最適な行数・列数・figsizeを計算するヘルパー関数
    """
    if n_plots == 0:
        return 1, 1, single_fig_size
    
    # プロット数が1つの場合は、1x1で単一サイズを返す
    if n_plots == 1:
        return 1, 1, single_fig_size

    n_cols = min(4, n_plots) # 最大4列
    n_rows = math.ceil(n_plots / n_cols)
    
    total_width = single_fig_size[0] * n_cols
    total_height = single_fig_size[1] * n_rows
    
    return n_rows, n_cols, (total_width, total_height)

def plot_speed_over_time(target_dict, mode, base_path, use_y_log, velocity_threshold_key, single_fig_size, save_path, auto_save):
    """
    グラフ1.各個体の速度の時系列グラフを描画する関数。
    """
    n_plots = len(target_dict)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, single_fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    # 各コロニーのループ処理
    print(f"\n - 速度時系列一覧グラフ作成 ({mode})")
    for i, (name, folder_id) in enumerate(target_dict.items()):
        # プロットエリアを超えたら終了
        if i >= len(axes):
            break
        
        ax = axes[i]
        print(f"Processing: {name}...")

        # パスの構築
        # (CSVファイル名規則は '{folder_id}-position_velocity.csv' と仮定)
        current_csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position_velocity.csv")

        # データ読み込み (data_input関数を利用)
        # ※ remove_outliers はここで True/False を指定するか、関数の引数に追加して渡す必要があります
        # 今回は暫定的に True としています
        df_vel, ids, time_minutes, fps = data_input(current_csv_path, remove_outliers=True)

        # データ読み込み失敗時
        if df_vel is None:
            ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center')
            ax.set_title(name)
            continue

        threshold_map = get_individual_thresholds(current_csv_path, velocity_threshold_key)

        # frameから分への変換
        time_in_minutes = fps * 60

        # 4. 個体ごとのプロット (重ね合わせ)
        for uid in ids:
            speed_col = f'speed_{uid}'
            # 閾値線を追加
            speed_per_min = df_vel[speed_col] * time_in_minutes  # ピクセル/分に変換
            ax.plot(time_minutes, speed_per_min, linewidth=0.8, alpha=0.6)
            # ★修正点: 自動割り当てされた色を取得
            line = ax.get_lines()[-1]
            plot_color = line.get_color()
            th_original = threshold_map.get(str(uid))
            if th_original is not None:
                th_converted = th_original * time_in_minutes
                ax.axhline(y=th_converted, color=plot_color, linestyle='--', linewidth=1.0, alpha=0.4
                           , label=f'ID:{uid} - {th_converted:.1f}')

        # 5. グラフの体裁設定
        ax.set_title(name, fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if use_y_log:
            ax.set_yscale('log')
            ax.set_ylim(1e-1,1e4)
        else:
            ax.set_ylim(0,8000)
        
        # 軸ラベルは最下段と左端のみに表示してスッキリさせる
        if i >= (n_rows - 1) * n_cols: # 最下段
            ax.set_xlabel('Time [min]', fontsize=20)
        if i % n_cols == 0: # 左端
            ax.set_ylabel('Speed [pixel/min]', fontsize=20)

        # 凡例 (個体数が多いため、文字を小さくするか、あるいは非表示にする)
        if len(ids) <= 10:
            ax.legend(loc='lower left', fontsize=15, title="ID - threshold[pixel/min]", title_fontsize=15)

    # 6. 余ったサブプロットを非表示にする
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.suptitle(f"Speed Over Time ({mode.capitalize()})", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0.01,0,1,1])

    # 7. 保存または表示
    if auto_save:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")
        plt.close()
    else:
        print(f"グラフを表示します: {save_path}")
        plt.show()

def plot_speed_distribution(target_dict, mode, base_path, use_x_log, velocity_threshold_key, single_fig_size, save_path, auto_save):
    """
    グラフ2. 各個体の速度分布を描画する関数。
    """
    n_plots = len(target_dict)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, single_fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    # 各コロニーのループ処理
    print(f"\n - 速度分布一覧グラフ作成 ({mode})")
    for i, (name, folder_id) in enumerate(target_dict.items()):
        ax = axes[i]
        print(f" - Processing {name}...")
        
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position_velocity.csv")
        df_vel, ids, _, fps = data_input(csv_path, remove_outliers=True)

        if df_vel is None:
            ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center')
            continue

        threshold_map = get_individual_thresholds(csv_path, velocity_threshold_key)

        # frameから分への変換
        time_in_minutes = fps * 60

        # ビン設定 (px/frame単位)
        local_max = 0
        for uid in ids:
            # maxをとってから変換
            local_max = max(local_max, df_vel[f'speed_{uid}'].max())
        
        # 上限キャップも変換 (例: 150 px/frame -> 18000 px/min)
        limit_px_fr = 150
        limit_px_min = limit_px_fr * time_in_minutes
        
        plot_max_original = min(local_max, limit_px_fr)
        plot_max_converted = plot_max_original * time_in_minutes
        
        if use_x_log:
        # 対数軸の場合: 0を含められないため、最小値を1.0 (px/min) 程度に設定
            plot_min_converted = 1.0 
            # log10スケールで等間隔なビンを作成
            if plot_max_converted <= plot_min_converted: plot_max_converted = 100 # 安全策
            bins = np.logspace(np.log10(plot_min_converted), np.log10(plot_max_converted), 50)
            ax.set_xscale('log')
        else:
            # 線形軸の場合
            plot_min_converted = 0
            bins = np.linspace(0, plot_max_converted, 50)

        for uid in ids:
            # データを変換
            data_original = df_vel[f'speed_{uid}'].dropna()
            data_converted = data_original * time_in_minutes
            
            if len(data_converted) == 0: continue

            weights = np.ones_like(data_converted) / len(data_converted)
            
            # 確率正規化 (weights)
            n, bins_out, patches = ax.hist(data_converted, bins=bins, weights=weights, histtype='step', 
                                                    linewidth=1.2, alpha=0.7, label=None)
            
            # ★修正点: 自動割り当てされた色を取得
            hist_color = patches[0].get_edgecolor()
            
            # 閾値線 (色はヒストグラムと同じ hist_color を使用)
            th_original = threshold_map.get(str(uid))
            if th_original is not None:
                th_converted = th_original * time_in_minutes
                ax.axvline(x=th_converted, color=hist_color, linestyle='--', linewidth=2.0, alpha=0.8
                           , label=f'ID:{uid} - {th_converted:.1f}')

        # グラフの体裁設定
        ax.set_title(name, fontsize=20)
        ax.set_ylim(0,0.20)
        ax.set_xlim(0, plot_max_converted)
        ax.grid(True, linestyle='--', alpha=0.4)

        if i % n_cols == 0:
            ax.set_ylabel('Probability', fontsize=20)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Speed [pixel/min]', fontsize=20) 

        # # 凡例の表示
        # if len(ids) <= 10:
        #     ax.legend(loc='upper left', fontsize=15, title="ID - threshold[pixel/min]", title_fontsize=15)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.suptitle(f"Speed Distribution ({mode.capitalize()})", fontsize=20, y=0.98)
    plt.tight_layout()


    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")
        plt.close()
    else:
        print(f"グラフを表示します: {save_path}")
        plt.show()

def plot_activity_state_over_time(target_dict, mode, base_path, velocity_threshold_key, single_fig_size, save_path, auto_save):
    """
    グラフ3. 各個体の活動状態の時系列グラフを描画する関数。
    """
    n_plots = len(target_dict)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, single_fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    # 各コロニーのループ処理
    print(f"\n - 活動状態一覧グラフ作成 ({mode})")
    for i, (name, folder_id) in enumerate(target_dict.items()):
        # プロットエリアを超えたら終了
        if i >= len(axes):
            break

        ax = axes[i]
        print(f"Processing: {name}...")
        
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position_velocity.csv")
        df_vel, ids, time_minutes, _ = data_input(csv_path, remove_outliers=True)

        if df_vel is None:
            ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center')
            continue

        threshold_map = get_individual_thresholds(csv_path, velocity_threshold_key)

        for uid in ids:
            col = f'speed_{uid}'
            th = threshold_map.get(str(uid), 0)
            
            # 判定 (px/frame単位同士で比較でOK、0/1の結果は変わらないため)
            activity = (df_vel[col] > th).astype(int)
            
            ax.plot(time_minutes, activity, linewidth=1.0, alpha=0.4, label=f'ID:{uid}')

        ax.set_title(name, fontsize=20)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Inactive\n(0)', 'Active\n(1)'], fontsize=20)
        ax.set_ylim(-0.1, 1.2)
        # ax.set_xlim(85,185)
        ax.grid(True, linestyle='--', alpha=0.4)

        if i >= (n_rows - 1) * n_cols: # 最下段
            ax.set_xlabel('Time [min]', fontsize=20)
        if i % n_cols == 0: # 左端
            ax.set_ylabel('Activity State', fontsize=20)

        # 凡例
        if len(ids) <= 10:
            ax.legend(loc='upper right', fontsize=15)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.suptitle(f"Activity State Over Time (latter half) ({mode.capitalize()})", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0,0.05,1,1])

    # 7. 保存または表示
    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")
        plt.close()
    else:
        print(f"グラフを表示します: {save_path}")
        plt.show()

# 3.統計量計算関数
def collect_statistics(target_dict, mode, base_path, velocity_threshold_key, stats_list):
    """
    指定されたコロニー群の統計データ(平均速度、活動率など)を計算し、リストに追加する関数。
    """
    print(f" - 統計データ収集中 ({mode})...")
    
    for name, folder_id in target_dict.items():
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position_velocity.csv")
        df_vel, ids, _, fps = data_input(csv_path, remove_outliers=True)

        if df_vel is None:
            continue

        threshold_map = get_individual_thresholds(csv_path, velocity_threshold_key)
        
        # 単位変換係数
        px_per_frame_to_px_per_min = fps * 60.0
        
        for uid in ids:
            col = f'speed_{uid}'
            data = df_vel[col] # px/frame
            
            # 閾値 (px/frame)
            th_px_frame = threshold_map.get(str(uid), 0)
            
            # --- 統計量の計算 ---
            
            # 1. 速度関連 (px/min に変換)
            mean_speed = data.mean() * px_per_frame_to_px_per_min
            max_speed = data.max() * px_per_frame_to_px_per_min
            
            # 2. 総移動距離 (pixel)
            # 速度(px/frame)の総和 = 総移動距離
            total_distance = data.sum()
            
            # 3. 活動量関連
            # 活動フレーム数
            active_frames = (data > th_px_frame).sum()
            total_frames = len(data)
            
            # 活動率 (0.0 - 1.0)
            active_ratio = active_frames / total_frames if total_frames > 0 else 0
            
            # 活動時間 (分) = 活動フレーム数 / FPS / 60
            active_duration_min = active_frames / fps / 60.0
            
            # 閾値表示用 (px/min)
            th_px_min = th_px_frame * px_per_frame_to_px_per_min

            # 結果を辞書としてリストに追加
            stats_list.append({
                "Group_Type": mode,
                "Colony_Name": name,
                "ID": uid,
                "Threshold_px_min": round(th_px_min, 2),
                "Mean_Speed": round(mean_speed, 2),
                "Max_Speed": round(max_speed, 2),
                "Total_Distance": round(total_distance, 2),
                "Active_Ratio": round(active_ratio, 4),
                "Active_Duration_Min": round(active_duration_min, 2)
            })


# メイン処理
if __name__ == "__main__":
    # 入力ファイル
    ISO_DICT = {
        "Colony A": "20251030_01", "Colony B": "20251104_02", "Colony C": "20251106_01",
        "Colony D": "20251113_01", "Colony E": "20251117_02", "Colony G": "20251119_01",
        "Colony H": "20251120_02", "Colony I": "20251127_01",
        }
    PAIR_DICT = {
        "Colony A": "20251030_02", "Colony B": "20251105_01", "Colony C": "20251107_01",
        "Colony D": "20251113_03", "Colony E": "20251118_01", "Colony G": "20251119_02",
        "Colony H": "20251121_01", "Colony I": "20251127_02",
    }
    TRIO_DICT = {
        "Colony A": "20251101_01", "Colony B": "20251105_02", "Colony C": "20251110_01",
        "Colony D": "20251117_01", "Colony E": "20251118_02", "Colony G": "20251120_01",
        "Colony H": "20251126_01", "Colony I": "20251128_01",
    }   

    # 個別コロニーを指定(空リストの場合は全コロニーで実行)
    TARGET_SPECIFIC = ["Colony G"]

    # ベースパスの指定 (環境に合わせて変更)
    BASE_PATH = "/Users/sonya/卒論データ/analysis_data"

    VELOCITY_THRESHOLD = "avg_half"  # 'q1', 'median_q2', 'q3', 'avg_half', または None
    REMOVE_OUTLIERS = True           # 外れ値を除去するか (True / False)
    USE_X_LOG = True
    USE_Y_LOG = True
    FIG_SIZE = (6, 4)  # 個別グラフのサイズ
    AUTO_SAVE = True

    # 出力先パス
    OUTPUT_DIR = "/Users/sonya/卒論データ/analysis_data/graph_output" # または任意のフォルダ

    # 実行
    if AUTO_SAVE and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 統計データを蓄積するリスト
    all_statistics = []

    DATA_SETS = [
        (ISO_DICT, "Isolate", "Isolate"),
        (PAIR_DICT, "Pair", "Pair"),
        (TRIO_DICT, "Trio", "Trio")
        ]

    for source_dict, mode_label, file_suffix in DATA_SETS:
            
        # 処理対象の辞書を作成
        if TARGET_SPECIFIC:
            # 指定されたコロニーのみを抽出
            process_dict = {k: v for k, v in source_dict.items() if k in TARGET_SPECIFIC}
            # 指定コロニーがないモードはスキップ
            if not process_dict:
                continue
            is_specific_mode = True
        else:
            # 全コロニー
            process_dict = source_dict
            is_specific_mode = False

        if not process_dict: continue

        # --- A. 統計データの収集 ---
        # グラフ描画前に計算してリストに追加
        collect_statistics(process_dict, mode_label, BASE_PATH, VELOCITY_THRESHOLD, all_statistics)
        
        # 特定コロニー実行時
        if TARGET_SPECIFIC:
            for target_colony in TARGET_SPECIFIC:
                # この辞書にそのコロニーが含まれているか確認
                if target_colony in source_dict:
                    # 1つだけの辞書を作成
                    single_target_dict = {target_colony: source_dict[target_colony]}
                    
                    print(f"\nProcessing Specfic Target: {target_colony} ({mode_label}) ===")
                    
                    # ファイル名にコロニー名を含める
                    # 例: Speed_Over_Time_Colony A_Isolated.png
                    
                    # (1) 速度時系列
                    save_name = f"Speed_Over_Time_{target_colony}_{file_suffix}.png"
                    plot_speed_over_time(single_target_dict, f"{target_colony} - {mode_label}", 
                                         BASE_PATH, USE_Y_LOG, VELOCITY_THRESHOLD, FIG_SIZE, 
                                         os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
                    
                    # (2) 速度分布
                    save_name = f"Speed_Distribution_{target_colony}_{file_suffix}.png"
                    plot_speed_distribution(single_target_dict, f"{target_colony} - {mode_label}", 
                                            BASE_PATH, USE_X_LOG, VELOCITY_THRESHOLD, FIG_SIZE, 
                                            os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
                    
                    # (3) 活動状態時系列
                    save_name = f"Activity_State_{target_colony}_{file_suffix}.png"
                    plot_activity_state_over_time(single_target_dict, f"{target_colony} - {mode_label}", 
                                                  BASE_PATH, VELOCITY_THRESHOLD, FIG_SIZE, 
                                                  os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)

        # 全コロニー実行
        else:
            print(f"\nProcessing All Targets ({mode_label})")
            # # (1) 速度時系列
            save_name = f"Speed_Over_Time_All_{file_suffix}.png"
            plot_speed_over_time(source_dict, mode_label, BASE_PATH, USE_Y_LOG, VELOCITY_THRESHOLD, FIG_SIZE, 
                                 os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
            
            # # (2) 速度分布
            save_name = f"Speed_Distribution_All_{file_suffix}.png"
            plot_speed_distribution(source_dict, mode_label, BASE_PATH, USE_X_LOG, VELOCITY_THRESHOLD, FIG_SIZE, 
                                    os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
            
            # # (3) 活動状態時系列
            save_name = f"Activity_State_All_{file_suffix}.png"
            plot_activity_state_over_time(source_dict, mode_label, BASE_PATH, VELOCITY_THRESHOLD, FIG_SIZE, 
                                          os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
            
            # (4) 活動状態時系列(後半のみ)
            # save_name = f"Activity_State_All_{file_suffix},(latter_half).png"
            # plot_activity_state_over_time(source_dict, mode_label, BASE_PATH, VELOCITY_THRESHOLD, FIG_SIZE, 
            #                               os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
            

    if all_statistics and AUTO_SAVE:
        df_stats = pd.DataFrame(all_statistics)
        
        # 列の並び順を整える
        columns_order = [
            "Group_Type", "Colony_Name", "ID", "Threshold_px_min", 
            "Mean_Speed", "Max_Speed", "Total_Distance", 
            "Active_Ratio", "Active_Duration_Min"
        ]
        # 定義した列のみ、あるいは存在する列のみ抽出
        final_cols = [c for c in columns_order if c in df_stats.columns]
        df_stats = df_stats[final_cols]
        
        csv_save_path = os.path.join(OUTPUT_DIR, "Speed_summary.csv")
        df_stats.to_csv(csv_save_path, index=False)
        print(f"\統計データを保存しました: {csv_save_path}")



