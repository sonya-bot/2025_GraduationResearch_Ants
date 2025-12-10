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

    return df_vel, speed_cols, output_dir, time_minutes, individual_ids


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
    df_vel, speed_cols, output_dir, time_minutes, individual_ids = data_input(velocity_csv_path)
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
    df_vel, speed_cols, output_dir, time_minutes, individual_ids = data_input(velocity_csv_path)
    if df_vel is None: return 

    # 速度データの数値化と外れ値処理 (修正済みの calculate_speed を使用)
    df_vel = calculate_speed(df_vel, individual_ids, remove_outliers)

# 2. 閾値情報の取得
    _, individual_results_list = calculate_thresholds.get_threshold_values(velocity_csv_path)
    
    if individual_results_list is None:
        print("閾値の計算に失敗したため、プログラムを終了します。")
        return

    # 個体IDをキーにした辞書に変換
    thresh_map = {res['id']: res for res in individual_results_list}

    print(f"{len(individual_ids)} 個体の活動状態グラフを作成します...")

# 3. 個体ごとのループ処理 (修正済み)
    for individual_id in individual_ids:
        # この個体の閾値データを取得
        ind_res = thresh_map.get(individual_id)
        if ind_res is None:
            print(f"ID: {individual_id} の閾値データが見つかりません。スキップします。")
            continue

        if velocity_threshold is not None:
            # 指定されたキー(例: 'avg_half')の値を取得
            selected_threshold = ind_res.get(velocity_threshold)
        else:
            print("閾値キーが None です。スキップします。")
            continue

        # 活動状態の計算 (1:活動, 0:非活動)
        speed_col_name = f'speed_{individual_id}'
        activity_state = (df_vel[speed_col_name] > selected_threshold).astype(int)

        # 4.グラフの描画
        # プロット用データ作成
        plot_data = {f'Activity State (ID:{individual_id})': activity_state}
        save_path = os.path.join(output_dir, f"Activity_State_over_Time_(ID_{individual_id}).png")
        
        # 活動状態用の設定
        yticks_settings = ([0, 1], ['Inactive \n (0)', 'Active \n (1)'])

        draw_graph(
            plot_data=plot_data,
            x_data=time_minutes,
            fig_size=fig_size,
            title_text=f'Activity State over Time (ID:{individual_id})',
            use_x_log=use_x_log, 
            x_label='Time (min)',
            use_y_log=False, # 活動状態グラフは2値なので対数にする必要はない
            y_label='State of Activity',
            y_fixed_range=(-0.2, 1.2),       # Y軸範囲を固定
            y_tick_labels=yticks_settings,    # カスタム目盛り
            auto_save=auto_save,
            save_path=save_path
        )
    
    return selected_threshold # 閾値プロット用に返す 

    print("全ての個体の処理が完了しました")
    
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

    plot_speed_over_time(INPUT_VELOCITY_CSV, REMOVE_OUTLIERS, VELOCITY_THRESHOLD ,PLOT_VELOCITY_THRESHOLD, USE_X_LOG, USE_Y_LOG, FIG_SIZE, AUTO_SAVE)
    plot_activity_state(INPUT_VELOCITY_CSV, REMOVE_OUTLIERS,VELOCITY_THRESHOLD , USE_X_LOG, USE_Y_LOG, FIG_SIZE, AUTO_SAVE)