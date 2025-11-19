# -*- coding utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import calculate_thresholds # 閾値計算用のモジュールをインポート
import plot_social_network
import os
from matplotlib.ticker import MaxNLocator, LogLocator
from itertools import combinations

# 0.実行設定 (初期状態はすべてFalse,main文の中で適宜変更して使用)
    # 1匹の場合、1匹の速度データをもとに閾値以下を0、以上を1とする信号を作成し、COS計算・描画する
RUN_PLOT_SINGLE_COS = False
    # 2匹以上の場合、ペアごとにCOSを計算・描画する
RUN_PLOT_PAIR_COS = False
    # 3匹以上の場合は後で追加予定
RUN_PLOT_TRIPLE_COS = False

def plot_solo_cos(position_csv_path, velocity_csv_path, velocity_threshold, contact_threshold, remove_outliers
                       , fig_size, auto_save):
    """
    1匹の個体の速度データをもとに、COSを計算・描画する。
    閾値以下を0、以上を1とする信号を作成し、時間変化グラフを描画する。
    """
# 1.データの読み込み
    # 位置データの読み込み
    try:
        df_pos = pd.read_csv(position_csv_path)
        print(f"'{position_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return
    # 速度データの読み込み
    try:
        # COS計算（speed）用
        df_vel = pd.read_csv(velocity_csv_path)
        print(f"'{velocity_csv_path}' を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {velocity_csv_path}")
        return
    
    # 速度データの列を特定
    speed_cols = [col for col in df_vel.columns if col.startswith('speed_')]
    if not speed_cols:
        print("速度データの列が見つかりません。")
        return
    
    # 速度データ列の形(str)を数値(float)に変換
    for col in speed_cols:
        df_vel[col] = pd.to_numeric(df_vel[col], errors='coerce')
        df_vel[col].fillna(0, inplace=True)
    print("速度データ列を数値に変換しました。")

    # 外れ値の処理
    if remove_outliers:
        predefined_threshold = 200.0 # 閾値を設定 例: 50.0
        for col in speed_cols:
            outlier_count = df_vel[df_vel[col] > predefined_threshold].shape[0]
            if outlier_count > 0:
                print(f"  列 '{col}': {predefined_threshold:.2f} を超える {outlier_count} 個の外れ値を0に置換しました。")
                df_vel.loc[df_vel[col] > predefined_threshold, col] = 0 # 外れ値を0に置換
        print("外れ値の処理が完了しました。") 

# 3.活動状態の判定
    # しきい値の取得
    threshold_values, _ = calculate_thresholds.get_threshold_values(velocity_csv_path)
    if threshold_values is None:
        print("閾値の計算に失敗したため、プログラムを終了します。") 
        return
    else:
        if velocity_threshold is not None:
            selected_threshold = threshold_values.get(velocity_threshold)
            print(f"使用する閾値のキー: {velocity_threshold}, 値: {selected_threshold}")
        else:
            print("閾値キーが None に設定されています。活動判定をスキップします。")
            return
    
#  4.活動状態(Activity State)の計算
    individual_id = speed_cols[0].replace('speed_', '')
    speed_col_name = speed_cols[0]
    print(f"個体 ID: {individual_id} の活動状態を判定します。")
    
    # 閾値より大きいフレームを 1 (活動)、それ以外を 0 (非活動) とする
    activity_state = (df_vel[speed_col_name] > selected_threshold).astype(int)
    print(f"  - 活動状態を判定しました。")

# 7.グラフの横軸となる時間(秒)への変換
    FPS = 2.0 # 1フレーム=1/2秒
    total_frames = len(df_pos)
    frames = df_pos['position']
        
    time_seconds = frames / FPS
    time_minutes = time_seconds / 60
    total_time_in_minutes = total_frames / FPS / 60
    
    print(f"  - 時間軸 (分) を計算しました (FPS={FPS}, 合計時間: {total_time_in_minutes:.2f} 分)。")

# 8.グラフの描画
    plt.figure(figsize=fig_size)
    ax = plt.gca()

    # 活動状態の時系列をプロット (Hayashi, 2012, Fig. 5(c) スタイル)
    ax.step(time_minutes, activity_state, where='mid', label=f'Activity State (ID:{individual_id})', linewidth=1.0)

    # タイトル (動的)
    ax.set_title(f'Activity State over Time (ID:{individual_id})', fontsize=14)
    
    # X軸 (分)
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_xlim(0, total_time_in_minutes)
    
    # Y軸 (0/1の活動状態)
    ax.set_ylabel('State of Activity', fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.5, 1.5) # 上下にも少し余白
    
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 5. グラフの表示 (ペアごとに1枚ずつ)
    if auto_save:
        # ファイル名 (動的)
        output_filename = f"Activity_State_over_Time_(ID_{individual_id}).png"
        output_directory = os.path.dirname(velocity_csv_path)
        save_path = os.path.join(output_directory, output_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" - 活動状態グラフを保存しました: {save_path}")
    else:
        print(f" - 活動状態グラフを表示します: ID {individual_id}")
        plt.show()

    print(f"個体 {individual_id} の処理が完了しました")


def plot_pair_cos(position_csv_path, velocity_csv_path, velocity_threshold, contact_threshold,remove_outliers
                       , fig_size, auto_save):
    """
    2匹の個体のペアごとにCOSを計算・描画する。
    COS = A - B + 2AB で定義
    A,Bはそれぞれ個体A、Bの活動状態(0:非活動,1:活動)
    """

# 1.データの読み込み
    # 位置データの読み込み
    try:
        df_pos = pd.read_csv(position_csv_path)
        print(f"'{position_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return
    # 速度データの読み込み
    try:
        # COS計算（speed）用
        df_vel = pd.read_csv(velocity_csv_path)
        print(f"'{velocity_csv_path}' を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {velocity_csv_path}")
        return
    
# 2.個体IDの特定と個体ごとの反復処理
    x_cols = [col for col in df_pos.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]

    n_individuals = len(individual_ids)
    if n_individuals <= 1:
        print(f"エラー: 検出された個体数が {n_individuals} のため、グラフを作成できません。")
        print("プログラムを終了します")
        return
    print(f"{n_individuals} 個体を対象に、全ペアの距離変化グラフを作成します。")
    
    # 全ての個体のペアについてループ処理
    pair_combinations = list(combinations(individual_ids, 2))

    # 速度データの列を特定
    speed_cols = [col for col in df_vel.columns if col.startswith('speed_')]
    if not speed_cols:
        print("速度データの列が見つかりません。")
        return
    
    # 速度データ列の形(str)を数値(float)に変換
    for col in speed_cols:
        df_vel[col] = pd.to_numeric(df_vel[col], errors='coerce')
        df_vel[col].fillna(0, inplace=True)
    print("速度データ列を数値に変換しました。")

    # 外れ値の処理
    if remove_outliers:
        predefined_threshold = 200.0 # 閾値を設定 例: 50.0
        for col in speed_cols:
            outlier_count = df_vel[df_vel[col] > predefined_threshold].shape[0]
            if outlier_count > 0:
                print(f"  列 '{col}': {predefined_threshold:.2f} を超える {outlier_count} 個の外れ値を0に置換しました。")
                df_vel.loc[df_vel[col] > predefined_threshold, col] = 0 # 外れ値を0に置換
        print("外れ値の処理が完了しました。") 

# 3.活動状態の判定
    # しきい値の取得
    threshold_values, _ = calculate_thresholds.get_threshold_values(velocity_csv_path)
    if threshold_values is None:
        print("閾値の計算に失敗したため、プログラムを終了します。") 
        return
    else:
        if velocity_threshold is not None:
            selected_threshold = threshold_values.get(velocity_threshold)
            print(f"使用する閾値のキー: {velocity_threshold}, 値: {selected_threshold}")
        else:
            print("閾値キーが None に設定されています。活動判定をスキップします。")
            return # 閾値なしでは活動判定ができないため中断
        
    print(f"全 {len(pair_combinations)} ペアの活動状態を判定します")
    
    for id1, id2 in pair_combinations:
        print(f"\n処理中: ペア (A = ID: {id1}, B = ID: {id2})")

        # 速度データ (df_vel) と閾値 (selected_threshold) を比較
        speed_col_A = f'speed_{id1}'
        speed_col_B = f'speed_{id2}'
        
        # カラムが存在するかチェック (念のため)
        if speed_col_A not in df_vel.columns or speed_col_B not in df_vel.columns:
            print(f"エラー: 速度の列が見つかりません ({speed_col_A} or {speed_col_B})。このペアをスキップします。")
            continue
            
        # 閾値より大きいフレームを 1 (活動)、それ以外を 0 (非活動) とする
        # .astype(int) で True/False を 1/0 に変換
        A = (df_vel[speed_col_A] > selected_threshold).astype(int)
        B = (df_vel[speed_col_B] > selected_threshold).astype(int)
        
        print(f"  - 活動状態 (A, B) を判定しました。")

# 4.COS(Combination Of States)の計算,Hayashi,2012を参照
        COS = A - B + 2 * A * B
        
        print(f"  - COS (Combination of States) を計算しました。")

# 5.接触の判定
    # 位置データ (df_pos) から、このペアのx, y座標を取得
        pos_A_x = f'x{id1}'
        pos_A_y = f'y{id1}'
        pos_B_x = f'x{id2}'
        pos_B_y = f'y{id2}'

        # 座標カラムが存在するかチェック
        if not all(col in df_pos.columns for col in [pos_A_x, pos_A_y, pos_B_x, pos_B_y]):
            print(f"エラー: 座標カラムが見つかりません。このペアをスキップします。")
            continue

        # .to_numpy() を使って高速なNumpy計算
        pos_A = df_pos[[pos_A_x, pos_A_y]].to_numpy()
        pos_B = df_pos[[pos_B_x, pos_B_y]].to_numpy()

        # 全フレームのユークリッド距離を計算
        distances = np.sqrt(np.sum((pos_A - pos_B)**2, axis=1))
        
        # 距離が contact_threshold 以下のフレームを特定 (True/FalseのSeries)
        total_frames = len(df_pos)
        frames = df_pos['position']
        contact_frames = frames[distances <= contact_threshold]
        
        print(f"  - 接触判定 (全 {total_frames} Frame) を実行しました。")

# 7.グラフの横軸となる時間(秒)への変換
        FPS = 2.0 # 1フレーム=1/2秒
            
        # position (フレーム番号) を 時間 (秒) に変換
        # (変数 'frames' は 'df_pos['position']' と同義)
        time_seconds = frames / FPS
        # 時間を時間(秒)から時間(分)に変更
        time_minutes = time_seconds / 60
            
        # 接触したフレーム番号 (contact_frames) も、時間 (秒) に変換
        contact_times_sec = contact_frames / FPS
        # こちらも同様に変換
        contact_times_min = contact_frames / 60

        # ▼ 修正：合計時間（秒）を計算して表示
        total_time_in_seconds = total_frames / FPS
        total_time_in_minutes = total_time_in_seconds / 60
        
        print(f"  - 時間軸 (秒) を計算しました (FPS={FPS}, 合計時間: {total_time_in_minutes:.2f} 分)。")

# 8.グラフの描画
        # ペアごとに新しい図（Figure）を作成する
        plt.figure(figsize=fig_size)
        ax = plt.gca() # 現在のAxesを取得

        # [線グラフ] COSの時系列をプロット
        ax.plot(time_minutes, COS, label=f'COS (ID:{id1} , ID:{id2})', linewidth=1.0)
        
        # [点グラフ] 接触点をY=0（横軸上）にプロット
        # Y軸の値を0にするために、contact_times_sec と同じ長さの0の配列を生成
        plot_y = np.zeros_like(contact_times_min)
        ax.plot(contact_times_min, plot_y, 'o', color='red', markersize=3, label=f'Contact (<= {contact_threshold} px)')

        # タイトル (ペアごとに動的)
        ax.set_title(f'COS_2 and Contact over Time (ID:{id1} , ID:{id2})', fontsize=14)
        
        # X軸 (秒)
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_xlim(0, total_time_in_minutes) # X軸の範囲を0から合計時間までにする
        
        # Y軸 (COS)
        ax.set_ylabel('COS (Combination of States)', fontsize=12)
        # 目盛りをFig. 11 に合わせる
        ax.set_yticks([-1, 0, 1, 2])
        ax.set_ylim(-1.5, 2.5) # 上下にも少し余白を持たせる
        
        # Y=0 の補助線
        ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
        
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7) #
        
        # 5. グラフの表示 (ペアごとに1枚ずつ)
        if auto_save:
            output_filename = f"COS_2 and Contact over Time (ID:{id1} , ID:{id2}).png"
            output_directory = os.path.dirname(velocity_csv_path)
            save_path = os.path.join(output_directory, output_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" - COS変化グラフを保存しました: {save_path}")
        else:
            print(f" - COS変化グラフを表示します: ID (ID:{id1} , ID:{id2})")
            plt.show()

    print("全てのペアの処理が完了しました")


def plot_trio_cos(position_csv_path, velocity_csv_path, velocity_threshold, contact_threshold,remove_outliers
                       , fig_size, auto_save):
    """
    3匹の個体のペアごとにCOSを計算・描画する。
    式ではなく状態を個別に定義
    """

# 1.データの読み込み
    # 位置データの読み込み
    try:
        df_pos = pd.read_csv(position_csv_path)
        print(f"'{position_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return
    # 速度データの読み込み
    try:
        # COS計算（speed）用
        df_vel = pd.read_csv(velocity_csv_path)
        print(f"'{velocity_csv_path}' を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {velocity_csv_path}")
        return
    
    # 2.個体IDの特定と個体ごとの反復処理
    x_cols = [col for col in df_pos.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]

    n_individuals = len(individual_ids)
    if n_individuals <= 1:
        print(f"エラー: 検出された個体数が {n_individuals} のため、グラフを作成できません。")
        print("プログラムを終了します")
        return
    print(f"{n_individuals} 個体を対象に、全ペアの距離変化グラフを作成します。")
    
    # 全ての個体のペアについてループ処理
    trio_combinations = list(combinations(individual_ids, 3))

    # 速度データの列を特定
    speed_cols = [col for col in df_vel.columns if col.startswith('speed_')]
    if not speed_cols:
        print("速度データの列が見つかりません。")
        return
    
    # 速度データ列の形(str)を数値(float)に変換
    for col in speed_cols:
        df_vel[col] = pd.to_numeric(df_vel[col], errors='coerce')
        df_vel[col].fillna(0, inplace=True)
    print("速度データ列を数値に変換しました。")

    # 外れ値の処理
    if remove_outliers:
        predefined_threshold = 200.0 # 閾値を設定 例: 50.0
        for col in speed_cols:
            outlier_count = df_vel[df_vel[col] > predefined_threshold].shape[0]
            if outlier_count > 0:
                print(f"  列 '{col}': {predefined_threshold:.2f} を超える {outlier_count} 個の外れ値を0に置換しました。")
                df_vel.loc[df_vel[col] > predefined_threshold, col] = 0 # 外れ値を0に置換
        print("外れ値の処理が完了しました。") 
    
# 3.活動状態の判定
    # しきい値の取得
    threshold_values, _ = calculate_thresholds.get_threshold_values(velocity_csv_path)
    if threshold_values is None:
        print("閾値の計算に失敗したため、プログラムを終了します。") 
        return
    else:
        if velocity_threshold is not None:
            selected_threshold = threshold_values.get(velocity_threshold)
            print(f"使用する閾値のキー: {velocity_threshold}, 値: {selected_threshold}")
        else:
            print("閾値キーが None に設定されています。活動判定をスキップします。")
            return # 閾値なしでは活動判定ができないため中断
        
    print(f"全 {len(trio_combinations)} トリオの活動状態を判定します")
    
    for id1, id2, id3 in trio_combinations:
        print(f"\n処理中: トリオ (A = ID: {id1}, B = ID: {id2}, C = ID: {id3})")

        # 速度データ (df_vel) と閾値 (selected_threshold) を比較
        speed_col_A = f'speed_{id1}'
        speed_col_B = f'speed_{id2}'
        speed_col_C = f'speed_{id3}'
        
        # カラムが存在するかチェック (念のため)
        if speed_col_A not in df_vel.columns or speed_col_B not in df_vel.columns or speed_col_C not in df_vel.columns:
            print(f"エラー: 速度の列が見つかりません ({speed_col_A} or {speed_col_B} or {speed_col_C})。このトリオをスキップします。")
            continue
            
        # 閾値より大きいフレームを 1 (活動)、それ以外を 0 (非活動) とする
        # .astype(int) で True/False を 1/0 に変換
        A = (df_vel[speed_col_A] > selected_threshold).astype(int)
        B = (df_vel[speed_col_B] > selected_threshold).astype(int)
        C = (df_vel[speed_col_C] > selected_threshold).astype(int)
        
        print(f"  - 活動状態 (A, B, C) を判定しました。")

# 4.COS(Combination Of States)の計算,3匹用に定義
            # conditions と choices を使って計算します
        
        conditions = [
            (A == 0) & (B == 0) & (C == 0), # (0, 0, 0) -> 0
            (A == 1) & (B == 0) & (C == 0), # (1, 0, 0) -> 1
            (A == 0) & (B == 1) & (C == 1), # (0, 1, 1) -> -1
            (A == 1) & (B == 0) & (C == 1), # (1, 0, 1) -> -2
            (A == 0) & (B == 1) & (C == 0), # (0, 1, 0) -> 2
            (A == 1) & (B == 1) & (C == 0), # (1, 1, 0) -> -3
            (A == 0) & (B == 0) & (C == 1), # (0, 0, 1) -> 3
            (A == 1) & (B == 1) & (C == 1)  # (1, 1, 1) -> 4
        ]
        
        choices = [
            0,   # (0, 0, 0)
            1,   # (1, 0, 0)
            -1,  # (0, 1, 1)
            -2,  # (1, 0, 1)
            2,   # (0, 1, 0)
            -3,  # (1, 1, 0)
            3,   # (0, 0, 1)
            4    # (1, 1, 1)
        ]
        
        # 条件に基づいて値を割り当て (該当なしはデフォルト0)
        COS_3 = np.select(conditions, choices, default=0)
        
        print(f"  - 3体COSを計算しました。")

# 5.接触の判定
    # 位置データ (df_pos) から、このトリオのx, y座標を取得
        pos_A_x = f'x{id1}'
        pos_A_y = f'y{id1}'
        pos_B_x = f'x{id2}'
        pos_B_y = f'y{id2}'
        pos_C_x = f'x{id3}'
        pos_C_y = f'y{id3}' 

        #  接触判定のための距離計算
        total_frames = len(df_pos)
        frames = df_pos['position']

        # 座標カラムが存在するかチェック
        if not all(col in df_pos.columns for col in [pos_A_x, pos_A_y, pos_B_x, pos_B_y, pos_C_x, pos_C_y]):
            print(f"エラー: 座標カラムが見つかりません。このトリオをスキップします。")
            continue

        # .to_numpy() を使って高速なNumpy計算
        pos_A = df_pos[[pos_A_x, pos_A_y]].to_numpy()
        pos_B = df_pos[[pos_B_x, pos_B_y]].to_numpy()
        pos_C = df_pos[[pos_C_x, pos_C_y]].to_numpy()

        # 全フレームのユークリッド距離を計算
        distances_AB = np.sqrt(np.sum((pos_A - pos_B)**2, axis=1))
        distances_BC = np.sqrt(np.sum((pos_B - pos_C)**2, axis=1))
        distances_CA = np.sqrt(np.sum((pos_C - pos_A)**2, axis=1))
        # 2個体の接触はそれぞれ計算
        contact_pair_frames = []
        contact_AB = frames[distances_AB <= contact_threshold]
        contact_BC = frames[distances_BC <= contact_threshold]
        contact_CA = frames[distances_CA <= contact_threshold]
        contact_pair_frames = [contact_AB, contact_BC, contact_CA]
        # 3個体の接触は2パターンを考慮
        contact_trio_frames = []
            # 全員が接触
        triangle_contact = (distances_AB <= contact_threshold) & (distances_BC <= contact_threshold) & (distances_CA <= contact_threshold)
        triangle_contact_frames = frames[triangle_contact]
        contact_trio_frames.append(triangle_contact_frames)
            # 2個体が接触(1個体を介した接触)
        any_chain_contact =(distances_AB <= contact_threshold) & (distances_BC <= contact_threshold) | (distances_BC <= contact_threshold) & (distances_CA <= contact_threshold) | (distances_CA <= contact_threshold) & (distances_AB <= contact_threshold)
        chain_contact = any_chain_contact & (~triangle_contact)
        chain_contact_frames = frames[chain_contact]
        contact_trio_frames.append(chain_contact_frames)

        print(f"  - 接触判定 (全 {total_frames} Frame) を実行しました。")

# 7.グラフの横軸となる時間(秒)への変換
        FPS = 2.0 # 1フレーム=1/2秒
            
        # 全体の時間軸（X軸用）
        time_minutes = frames / FPS / 60
        total_time_in_minutes = total_frames / FPS / 60

        # 接触フレームのリストを整理 (フラットな構造にする)
        # 構造: [0:ペア接触(全体), 1:全結合(Triangle), 2:鎖状(Chain)]
        contact_frame_list = []
        contact_frame_list.append(contact_trio_frames[0]) # 0: Triangle
        contact_frame_list.append(contact_trio_frames[1]) # 1: Chain
        contact_frame_list.append(contact_pair_frames[0])  # 2: A-B
        contact_frame_list.append(contact_pair_frames[1])  # 3: B-C
        contact_frame_list.append(contact_pair_frames[2])  # 4: C-A

        # 各接触タイプごとに時間(分)に変換
        contact_times_min_list = []
        for frames_array in contact_frame_list:
            # numpy配列に対して計算
            times_min = frames_array / FPS / 60
            contact_times_min_list.append(times_min)

        # これで以下のデータが揃いました:
        # contact_times_min_list[0] -> ペア接触の時間
        # contact_times_min_list[1] -> 全結合の時間
        # contact_times_min_list[2] -> 鎖状の時間
        
        print(f"  - 時間軸 (秒) を計算しました (FPS={FPS}, 合計時間: {total_time_in_minutes:.2f} 分)。")

# 8.グラフの描画
        # ペアごとに新しい図（Figure）を作成する
        plt.figure(figsize=fig_size)
        ax = plt.gca() # 現在のAxesを取得

        # [線グラフ] COSの時系列をプロット
        ax.plot(time_minutes, COS_3, label=f'COS (ID:{id1} , ID:{id2})', linewidth=1.0)
        
        # [点グラフ] 接触点をY=0（横軸上）にプロット
        # プロット設定: (データインデックス, 色, ラベル, Y位置)
        pairs_config = [
                (2, 'green',  f'Pair (ID:{id1} , ID:{id2})'),
                (3, 'cyan',   f'Pair (ID:{id2} , ID:{id3})'),
                (4, 'purple', f'Pair (ID:{id3} , ID:{id1})')
            ]
        for idx, color, label in pairs_config:
            times = contact_times_min_list[idx]
            if len(times) > 0:
                ax.plot(times, np.zeros_like(times), 'o', color=color, markersize=8, label=label, alpha=0.5, zorder=2)

        # 仲介接触 Chain (オレンジ)
        times_chain = contact_times_min_list[1]
        if len(times_chain) > 0:
            ax.plot(times_chain, np.zeros_like(times_chain), 'o', color='red', markersize=6, label='Chain', zorder=3)

        # 全結合 Triangle (赤)
        times_triangle = contact_times_min_list[0]
        if len(times_triangle) > 0:
            ax.plot(times_triangle, np.zeros_like(times_triangle), 'o', color='red', markersize=6, label='Triangle', zorder=4)

        # タイトル (ペアごとに動的)
        ax.set_title(f'COS_3 and Contact over Time (ID:{id1} , ID:{id2} , ID:{id3})', fontsize=14)
        
        # X軸 (秒)
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_xlim(0, total_time_in_minutes) # X軸の範囲を0から合計時間までにする
        
        # Y軸 (COS)
        ax.set_ylabel('COS (Combination of States)', fontsize=12)
        # 目盛り
        ax.set_yticks([-3, -2, -1, 0, 1, 2, 3, 4])
        ax.set_ylim(-3.5, 4.5) # 上下にも少し余白を持たせる
        
        # Y=0 の補助線
        ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
        
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7) #
        
        # 5. グラフの表示 (ペアごとに1枚ずつ)
        if auto_save:
            output_filename = f"COS_2 and Contact over Time (ID:{id1} , ID:{id2}).png"
            output_directory = os.path.dirname(velocity_csv_path)
            save_path = os.path.join(output_directory, output_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" - COS変化グラフを保存しました: {save_path}")
        else:
            print(f" - COS変化グラフを表示します: ID (ID:{id1} , ID:{id2})")
            plt.show()


    print("全てのトリオの処理が完了しました")





# メイン処理
if __name__ == "__main__":
    # 位置データと速度データの両方を入力
    INPUT_CSV = "20251101_01"
    INPUT_POSITION_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position.csv"
    INPUT_VELOCITY_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position_velocity.csv"
    CONTACT_THRESHOLD = 50.0
    # 外れ値を除去するかどうか (True: 除去する, False: 除去しない)
    REMOVE_OUTLIERS = True
    # 活動状態の判定に使用する速度の閾値 STATE_THRESHOLD を選択
    # 'q1', 'median_q2', 'q3', 'avg_half' などから閾値のキーを選択(calculate_thresholdsで計算されるもの)
    VELOCITY_THRESHOLD = "avg_half"
    # 接触判定に使用する距離のしきい値DISTANCE＿THRESHOLD を設定
    # CONTACT_THRESHOLD = plot_social_network.CONTACT_THRESHOLD_PIXELS
    CONTACT_THRESHOLD = 50.0  # ピクセル単位の接触しきい値
    # グラフのサイズを指定
    FIG_SIZE = (10, 5)
    # グラフの自動保存
    AUTO_SAVE = False

    # 個体数の取得
    position_csv_path = INPUT_POSITION_CSV
    try:
        df_pos = pd.read_csv(position_csv_path)
        print(f"'{position_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません \n 処理を終了します")
        exit()

    x_cols = [col for col in df_pos.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]

    n_individuals = len(individual_ids)
    if n_individuals == 1:
        RUN_PLOT_SINGLE_COS = True
        print("1匹の個体が検出されました。1匹用のCOSグラフを作成します。")
    elif n_individuals == 2:
        RUN_PLOT_PAIR_COS = True
        print("2匹の個体が検出されました。2匹用のCOSグラフを作成します。")
    elif n_individuals ==3:
        print("3匹の個体が検出されました。3匹用のCOSグラフを作成します。")
        RUN_PLOT_TRIPLE_COS = True
    else:
        print(f"エラー: 検出された個体数が {n_individuals} のため、COSグラフを作成できません。")
        print("プログラムを終了します")
        exit()

    if RUN_PLOT_SINGLE_COS:
        print("\n--- [実行中] 1匹用COSグラフの作成 ---")
        plot_solo_cos(INPUT_POSITION_CSV, INPUT_VELOCITY_CSV, VELOCITY_THRESHOLD, CONTACT_THRESHOLD, remove_outliers=REMOVE_OUTLIERS
                            , fig_size=FIG_SIZE, auto_save=AUTO_SAVE)
        
    elif RUN_PLOT_PAIR_COS:
        print("\n--- [実行中] 2匹用COSグラフの作成 ---")
        plot_pair_cos(INPUT_POSITION_CSV, INPUT_VELOCITY_CSV, VELOCITY_THRESHOLD, CONTACT_THRESHOLD, remove_outliers=REMOVE_OUTLIERS
                            , fig_size=FIG_SIZE, auto_save=AUTO_SAVE)
        
    elif RUN_PLOT_TRIPLE_COS:
        print("\n--- [実行中] 3匹用COSグラフの作成 ---")
        plot_trio_cos(INPUT_POSITION_CSV, INPUT_VELOCITY_CSV, VELOCITY_THRESHOLD, CONTACT_THRESHOLD, remove_outliers=REMOVE_OUTLIERS
                            , fig_size=FIG_SIZE, auto_save=AUTO_SAVE)
