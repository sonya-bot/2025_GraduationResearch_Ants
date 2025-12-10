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

    return df_vel, speed_cols, output_dir, time_minutes, individual_ids, FPS



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

def calculate_states(velocity_csv_path, remove_outliers, velocity_threshold_key):
    """
    データ読み込み、速度計算、閾値判定を行い、
    全個体の「活動状態(0/1)の時系列データ」を辞書で返す共通関数。
    """
    # 1. データ読み込み
    df_vel, speed_cols, output_dir, time_minutes, individual_ids, FPS = data_input(velocity_csv_path)
    if df_vel is None: return None, None, None, None, None

    # 2. 速度計算
    df_vel = calculate_speed(df_vel, individual_ids, remove_outliers)

    # 3. 閾値取得
    _, individual_results_list = calculate_thresholds.get_threshold_values(velocity_csv_path)
    thresh_map = {res['id']: res for res in individual_results_list}

    # 4. 状態判定 (0/1化)
    activity_states = {}
    for uid in individual_ids:
        ind_res = thresh_map.get(uid)
        if ind_res:
            th = ind_res.get(velocity_threshold_key)
            speed_col = f'speed_{uid}'
            # 0/1 の配列を格納
            activity_states[uid] = (df_vel[speed_col] > th).astype(int)
            
    return activity_states, output_dir, time_minutes, individual_ids, FPS

def draw_graph(plot_data, x_data, fig_size, title_text,
               use_x_log, x_label, use_y_log, y_label, y_fixed_range=None, y_tick_labels=None,
               plot_velocity_threshold=None, auto_save=False, save_path ="graph.png"):
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
        ax.set_ylim(0,200)
        if y_fixed_range is not None:
            ax.set_ylim(y_fixed_range)
            if y_tick_labels:
                ticks, labels = y_tick_labels
                ax.set_yticks(ticks)
                ax.set_yticklabels(labels)

    # 速度閾値のプロット
    if plot_velocity_threshold is not None:
        threshold = plot_velocity_threshold
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.0, label=f'Threshold: {threshold:.1f}')


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
        

def plot_speed_over_time(velocity_csv_path, remove_outliers , velocity_threshold, plot_velocity_threshold, use_x_log, use_y_log, fig_size, auto_save):
    """
    速度の時系列データをプロットするメイン関数。
    """
# 1. データ読み込み & 前処理
    df_vel, speed_cols, output_dir, time_minutes, individual_ids, FPS= data_input(velocity_csv_path)
    if df_vel is None: return # 読み込み失敗時は終了
    df_vel = calculate_speed(df_vel, individual_ids, remove_outliers)

# 閾値のプロットを行う場合、個体ごとの閾値を取得    
    ind_thresh_map = {}
    
    if plot_velocity_threshold:
        _, individual_res_list = calculate_thresholds.get_threshold_values(velocity_csv_path)
        if individual_res_list:
            ind_thresh_map = {res['id']: res.get(velocity_threshold) for res in individual_res_list}
            print(f"閾値線を描画します (Key: {velocity_threshold})")
    

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
        plot_velocity_threshold=None, # 全体グラフでは閾値は表示しない
        auto_save=auto_save,
        save_path=save_path_all,
    )

# 3. 個別の速さグラフ (Individual)
    for i_id in individual_ids:
        ind_plot_data = {f'Speed of {i_id}': df_vel[f'speed_{i_id}']}
        save_path_ind = os.path.join(output_dir, f"Speed_over_Time_(ID_{i_id}).png")

        ind_threshold = ind_thresh_map.get(i_id) if plot_velocity_threshold else None
        
        draw_graph(
            plot_data=ind_plot_data,
            x_data=time_minutes,
            fig_size=fig_size,
            title_text=f'Speed over Time Individual (ID:{i_id})',
            use_x_log=use_x_log,
            x_label='Time (min)',
            use_y_log=use_y_log,
            y_label='Speed',
            plot_velocity_threshold=ind_threshold,
            auto_save=auto_save,
            save_path=save_path_ind,
        )

def plot_activity_state(velocity_csv_path, remove_outliers, velocity_threshold, use_x_log, use_y_log, fig_size, auto_save):
    """
    各個体の速度データをもとに、活動状態(0/1)を計算・描画する。
    """
# 1. データ読み込み
    activity_states, output_dir, time_minutes, individual_ids, FPS = calculate_states(velocity_csv_path, remove_outliers, velocity_threshold)
    if activity_states is None: return

    print(f"{len(individual_ids)} 個体の活動状態グラフを作成します...")

    # 3. 個体ごとのループ処理
    for individual_id in individual_ids:
        activity_state = activity_states[individual_id]

        # 4.グラフの描画
        plot_data = {f'Activity State (ID:{individual_id})': activity_state}
        save_path = os.path.join(output_dir, f"Activity_State_over_Time_(ID_{individual_id}).png")
        
        yticks_settings = ([0, 1], ['Inactive \n (0)', 'Active \n (1)'])

        draw_graph(
            plot_data=plot_data,
            x_data=time_minutes,
            fig_size=fig_size,
            title_text=f'Activity State over Time (ID:{individual_id})',
            use_x_log=use_x_log, 
            x_label='Time (min)',
            use_y_log=False, # 活動状態グラフは2値
            y_label='State of Activity',
            y_fixed_range=(-0.2, 1.2),
            y_tick_labels=yticks_settings,
            auto_save=auto_save,
            save_path=save_path
        )


def plot_activity_duration_cumlative_sum(velocity_csv_path, remove_outliers, velocity_threshold, use_x_log, use_y_log, fig_size, auto_save):
    """
    各個体の活動持続時間の累積和グラフを描画する。
    持続時間が短い順にソートして表示する。
    """
    # 1. 共通関数を使って状態データを取得
    activity_states, output_dir, _, individual_ids, FPS = calculate_states(velocity_csv_path, remove_outliers, velocity_threshold)
    if activity_states is None: return

    print(f"{len(individual_ids)} 個体の活動持続時間(累積和)グラフを作成します...")

    for uid in individual_ids:
        # 0/1配列を取得
        states = activity_states[uid].values

        # 状態の変化点を検出
        diffs = np.diff(states)
        change_indices = np.where(diffs != 0)[0] + 1
        
        # 区間の開始と終了インデックス
        indices = np.concatenate(([0], change_indices, [len(states)]))
        
        # 各区間の長さ（フレーム数）と、その時の状態（0か1か）
        durations = np.diff(indices)
        values = states[indices[:-1]]
        
        # Active(1) と Inactive(0) に振り分け
        active_durs = durations[values == 1] / FPS   # 秒に変換
        inactive_durs = durations[values == 0] / FPS # 秒に変換
        # 昇順ソート
        active_durs.sort()
        inactive_durs.sort()
        
        # 累積和 (Cumulative Duration)
        active_cumsum = np.cumsum(active_durs)
        inactive_cumsum = np.cumsum(inactive_durs)
        
        # ランク (1, 2, ...)
        active_ranks = np.arange(1, len(active_durs) + 1)
        inactive_ranks = np.arange(1, len(inactive_durs) + 1)
        
        # --- グラフ描画 (専用処理) ---
        plt.figure(figsize=fig_size)
        ax = plt.gca()
        
        # Active (赤)
        if len(active_durs) > 0:
            ax.plot(active_cumsum, active_ranks, label='Active', color='red', linewidth=2.0, alpha=0.5)
        
        # Inactive (青/グレー)
        if len(inactive_durs) > 0:
            ax.plot(inactive_cumsum, inactive_ranks, label='Inactive', color='blue', linewidth=2.0, alpha=0.5)
            
        ax.set_title(f'Activity Duration Cumulative Sum (ID:{uid})', fontsize=14)
        ax.set_xlabel('Cumulative Duration (sec)', fontsize=12)
        ax.set_ylabel('Rank (ascending order)', fontsize=12)
        
        # if use_y_log:
        #     ax.set_yscale('log')
        #     ax.set_ylabel('Rank (ascending order) [Log Scale]', fontsize=12)
        
        if use_x_log:
            ax.set_xscale('log')
            ax.set_xlabel('Cumulative Duration (sec) [Log Scale]', fontsize=12)

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        # コンソール出力 (デバッグ用)
        print(f" [ID:{uid}] Active Count: {len(active_durs)}, Max Dur: {np.max(active_durs) if len(active_durs)>0 else 0:.1f}s")
        print(f" [ID:{uid}] Inactive Count: {len(inactive_durs)}, Max Dur: {np.max(inactive_durs) if len(inactive_durs)>0 else 0:.1f}s")

        # 保存または表示
        if auto_save:
            output_filename = f"Activity_Duration_Cumulative_Sum(ID_{uid}).png"
            save_path = os.path.join(output_dir, output_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" - グラフを保存しました: {save_path}")
            plt.close()
        else:
            print(f" - グラフを表示します: ID {uid}")
            plt.show()



    print("全ての個体の処理が完了しました")

# 実行プログラム設定
RUN_SPEED_OVER_TIME = False
RUN_ACTIVITY_STATE = False
RUN_ACTIVITY_DURATION_CUMULATIVE_SUM = True
    
if __name__ == '__main__':
# メイン処理
    INPUT_CSV = "20251101_01"
    INPUT_VELOCITY_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position_velocity.csv"
    
    REMOVE_OUTLIERS = True
    VELOCITY_THRESHOLD = "avg_half"
    PLOT_VELOCITY_THRESHOLD = True # 速度閾値をプロットするかどうか
    USE_Y_LOG = True
    USE_X_LOG = False
    FIG_SIZE = (6, 4)
    AUTO_SAVE = False

# 選択した処理を実行
    if RUN_SPEED_OVER_TIME:
        plot_speed_over_time(INPUT_VELOCITY_CSV, REMOVE_OUTLIERS, VELOCITY_THRESHOLD ,PLOT_VELOCITY_THRESHOLD, USE_X_LOG, USE_Y_LOG, FIG_SIZE, AUTO_SAVE)
    if RUN_ACTIVITY_STATE:
        plot_activity_state(INPUT_VELOCITY_CSV, REMOVE_OUTLIERS,VELOCITY_THRESHOLD , USE_X_LOG, USE_Y_LOG, FIG_SIZE, AUTO_SAVE)
    if RUN_ACTIVITY_DURATION_CUMULATIVE_SUM:
        plot_activity_duration_cumlative_sum(INPUT_VELOCITY_CSV, REMOVE_OUTLIERS, VELOCITY_THRESHOLD , USE_X_LOG, USE_Y_LOG, FIG_SIZE, AUTO_SAVE)