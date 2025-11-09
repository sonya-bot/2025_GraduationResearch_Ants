# -*- coding utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import os
import sympy as sym
import calculate_thresholds # 閾値計算用のモジュールをインポート
import plot_social_network
from matplotlib.ticker import MaxNLocator, LogLocator
from itertools import combinations

def plot_state_distribution(position_csv_path, velocity_csv_path, velocity_threshold, contact_threshold, remove_outliers, plot_glaph):
    """
    ある初期状態 ( $\pi(0)$ ) からスタートした場合、時間が経過するにつれて8状態の確率分布 
    $\pi(t)$ がどのように推移するかをシミュレーションする。
    """

# 1.データ入力
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
    print(f"{n_individuals} 個体を対象に、全ペアの状態確率遷移グラフを作成します。")
    
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

# 3.活動状態の定義
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

    # 8状態のラベル (グラフの凡例で使用)
    state_labels = [
        "s1: Apart - Both_Inactive",   # COS=0, Contact=False
        "s2: Contact - Both_Inactive", # COS=0, Contact=True
        "s3: Apart - Both_Active",     # COS=2, Contact=False
        "s4: Contact - Both_Active",   # COS=2, Contact=True
        "s5: Apart - Mixed_Active_A",  # COS=1, Contact=False
        "s6: Contact - Mixed_Active_A",  # COS=1, Contact=True
        "s7: Apart - Mixed_Active_B",  # COS=-1, Contact=False
        "s8: Contact - Mixed_Active_B"   # COS=-1, Contact=True
    ]

    print(f"全 {len(state_labels)} 状態を定義しました。それぞれの状態を以下に示します")
    for i, label in enumerate(state_labels):
        # (i+1) で s1 から s8 と番号を合わせる
        print(f"  - {label}")

    # 各ペアの計算結果を保存する辞書
    transition_matrices = {} # ステップ5用
    state_series_all_pairs = {} # ステップ4の結果

# 4.全ペアの反復処理
    # 活動状態の判定
    for id1, id2 in pair_combinations:
        pair_name = f"Pair {id1} , {id2}"
        print(f"\n処理中: ペア (ID: {id1}, ID: {id2})")

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

# 5.COS(Combination Of States)の計算,Hayashi,2012を参照
        COS = A - B + 2 * A * B
        
        print(f"  - COS (Combination of States) を計算しました。")

# 6.接触の判定
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
        is_contact = (distances <= contact_threshold)
        
        print(f"  - 接触判定 (全 {total_frames} Frame) を実行しました。")

# 7.8状態へのマッピング
        # numpy.select を使って、条件に基づいて全フレームを 1〜8 の状態に分類
        
        conditions = [
            (COS == 0) & (is_contact == False), # s1: Apart - Both_Inactive
            (COS == 0) & (is_contact == True),  # s2: Contact - Both_Inactive
            (COS == 2) & (is_contact == False), # s3: Apart - Both_Active
            (COS == 2) & (is_contact == True),  # s4: Contact - Both_Active
            (COS == 1) & (is_contact == False), # s5: Apart - Mixed_Active_A
            (COS == 1) & (is_contact == True),  # s6: Contact - Mixed_Active_A
            (COS == -1) & (is_contact == False),# s7: Apart - Mixed_Active_B
            (COS == -1) & (is_contact == True)  # s8: Contact - Mixed_Active_B
        ]
        
        # 状態ラベル (1から8まで。0始まりではなく1始まりにする)
        choices = [1, 2, 3, 4, 5, 6, 7, 8]
        
        # state_series は [1, 1, 2, 2, 6, 2, ...] というフレームごとの状態番号の配列
        state_series = np.select(conditions, choices, default=0) # default=0 はエラーチェック用
        
        # エラーが検出された場合、処理を中断
        if 0 in state_series:
            print("警告: 不明な状態 (default=0) が検出されました。")
            return

        # 結果を辞書に保存
        state_series_all_pairs[pair_name] = state_series
        print(f"  - 「状態系列」(全 {len(state_series)} Frame) の生成が完了しました。")

# 8.遷移確率行列Pの計算
        # 8状態 (s1-s8) + 1 (s0=エラー用) のため、9x9 の行列を初期化
        transition_counts = np.zeros((9, 9))
        
        for i in range(len(state_series) - 1):
            current_state = state_series[i]
            next_state = state_series[i+1]
            
            if current_state == 0 or next_state == 0:
                continue # エラー状態 (default=0) の遷移は無視
                
            transition_counts[current_state, next_state] += 1
            
        print(f"  - 遷移回数行列を計算しました。")

        # 遷移確率行列 P を計算
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 # 0除算を回避
        P = transition_counts / row_sums
        
        # s1〜s8 の 8x8 行列 P を保存
        transition_matrices[pair_name] = P[1:, 1:]
        
        print(f"  - 8x8 遷移確率行列 P を計算しました。")

# 9.マルコフ連鎖のシミュレーション
        # 状態の定義
        states = np.arange(1, 9) # [1, 2, 3, 4, 5, 6, 7, 8]の全8状態

        # 遷移確率行列の取得
        P = transition_matrices[pair_name]

        # シミュレーションに使用するFrame数を指定
        N = len(frames) #Frames

        # 初期状態（s1）を設定 (place = [0])
        initial_state = [1] # (s1 からスタート)

        # マルコフ連鎖のシミュレーション (for _ in range(N-1):)
        for _ in range(N - 1):
            # 現在の状態 (1〜8) (current_state = place[-1])
            # P_matrix は 0〜7 でアクセスするため、-1 する
            current_state_index = initial_state[-1] - 1 
            
            # 遷移確率行列から次の状態をサンプリング (next_state = np.random.choice(...))
            # P_matrix[current_state_index] は、現在の状態から遷移する確率の行 (8要素)
            next_state = np.random.choice(states, p=P[current_state_index])
            
            initial_state.append(next_state) # (place.append(next_state))

        print(f"  - マルコフ連鎖(全 {N} Steps) のシミュレーションが完了しました 。")

# 10.確率推移の計算 
        # 各状態の推移（累積平均）を格納するリストを準備
        # (8状態分のリストを保持する辞書)
        probability_trends = {s: [] for s in states} # 'states' は [1, 2...8]

        # 各状態の推移（累積平均）を計算 (for i in range(N):)
        # (N はシミュレーションステップ数 N=1000)
        for i in range(1, N + 1):
            # 現在までの履歴スライス (place[:i+1])
            # (変数 'initial_state' を使用)
            current_history_slice = initial_state[:i] 
            current_length = len(current_history_slice)
            
            for s in states: # 'states' = [1, 2...8]
                # 履歴スライス内に s が出現した回数をカウント (place[:i+1].count(0))
                count = current_history_slice.count(s)
                # 累積確率を計算 (... / len(place[:i+1]))
                probability = count / current_length
                probability_trends[s].append(probability)
        
        print(f"  - 累積確率の推移を計算しました。")

# 11.グラフの横軸となる時間(秒)への変換
        FPS = 2.0 # 1フレーム=1/2秒
            
        # position (フレーム番号) を 時間 (秒) に変換
        # (変数 'frames' は 'df_pos['position']' と同義)
        time_seconds = frames / FPS
        # 時間を時間(秒)から時間(分)に変更
        time_minutes = time_seconds / 60

        # ▼ 修正：合計時間（秒）を計算して表示
        total_time_in_seconds = total_frames / FPS
        total_time_in_minutes = total_time_in_seconds / 60
        
        print(f"  - 時間軸 (秒) を計算しました (FPS={FPS}, 合計時間: {total_time_in_minutes:.2f} 分)。")

# 12.グラフの描画 
        if plot_glaph:
            fig, ax = plt.subplots(figsize=(15, 5))
            
            # 8状態の確率推移をすべてプロット (ax.plot(inside, ...))
            for s in states: # 'states' = [1, 2...8]
                # 'state_labels' (s1: ...) から正しいラベルを取得
                label_name = state_labels[s-1] 
                ax.plot(time_minutes, probability_trends[s], label=label_name, linewidth=1.5)

            # グラフの体裁 (ax.set(...))
            # (変数 'pair_name' を使用)
            ax.set_title(f'Markov Chain Simulation (ID:{id1} , {id2})', fontsize=14)
            ax.set_xlabel('Time (minutes)', fontsize=12) #
            ax.set_xlim(0, time_minutes.max())
            # (Y軸ラベルを 'Probability (Cumulative Average)' に変更)
            ax.set_ylabel('Probability (Cumulative Average)', fontsize=12) #
            ax.set_ylim(0, 1.0)
            ax.grid(True) #
            
            # 凡例をグラフの「内側・右上」('upper right') に配置
            ax.legend(loc='upper right', fontsize='small') #
            
            # グラフの表示(ペアごとに1枚ずつ)
            plt.tight_layout()
            # plt.show()

            # グラフの自動保存 
            # ファイル名 (例: "State_Distribution(ID_0,1).png")
            output_filename = f"State_Distribution(ID_{id1},{id2}).png"
            # ▼ 修正: os.path.join で「保存先フォルダ」と「ファイル名」を連結
            output_directory = os.path.dirname(position_csv_path)
            save_path = os.path.join(output_directory, output_filename)
            
            try:
                # dpi=300 で高解像度保存
                plt.savefig(save_path, dpi=300)
                print(f"  - グラフを '{output_filename}' として保存しました。")
            except Exception as e:
                print(f"  - グラフの保存中にエラーが発生しました: {e}")

            # plt.show() # 保存と同時に表示も行う

            # # (メモリを節約するために、表示後に図を閉じる)
            # plt.close(fig)
        else:
            print(f" - グラフの作成をスキップします")
            pass

    print("全てのペアの処理が完了しました")
    # 計算結果の「辞書」と「ラベル」、frames を return する(後の関数で使用)
    return transition_matrices, state_labels, id1, id2

def analytical_calculation(transition_matrices, state_labels):
    """
    「遷移確率行列 P」 から、sympy を使って定常分布 (pi = pi*P) を
    解析的に計算し、コンソールに出力する。
    """
    print("\n解析計算 (定常分布)を行います")

    # P行列の辞書をループ
    for pair_name, P_matrix in transition_matrices.items():
        print(f"\n処理中: ペア (ID: {id1}, ID: {id2})")
        pi_vars = sym.symbols(f'pi_1:{9}') # (pi_1, pi_2, ..., pi_8)
        equations = []
        for j in range(8): # j = 0 から 7 (s1 から s8)
            lhs = pi_vars[j]
            rhs = 0
            for i in range(8): # i = 0 から 7 (s1 から s8)
                rhs += pi_vars[i] * P_matrix[i, j]
            equations.append(sym.Eq(lhs, rhs)) #
        
        equations.append(sym.Eq(sum(pi_vars), 1)) #
        print(f"  - 9個の連立方程式 (pi = pi*P, sum(pi)=1) を構築しました。")
        print(f"  - sympy.solve で解析計算を実行します...(時間がかかる場合があります)")

        try:
            solution = sym.solve(equations[:7] + [equations[8]], pi_vars) #
        except Exception as e:
            print(f"  - sympy.solve でエラーが発生しました: {e}")
            continue 

        if not solution:
            print(f"  - sympy.solve が解を見つけられませんでした。スキップします。")
            continue

        stationary_distribution = [solution.get(pi, 0) for pi in pi_vars]
        print(f"  - 定常分布 (解析計算) が完了しました。")
        
        print(f"    --- 定常分布 (Stationary Distribution) ペア (ID_{id1},{id2}) ---")
        for i in range(len(state_labels)):
            state_name = state_labels[i] #
            probability = stationary_distribution[i]
            print(f"      {state_name}: {float(probability)*100:.2f} %")

    print(f"\n全てのペアの解析計算 (定常分布) が完了しました。")

# メイン処理
if __name__ == "__main__":
    # 位置データと速度データの両方を入力
    INPUT_POSITION_CSV = "/Volumes/100.108.13.8/analysis_data/20251101_01/20251101_01-position.csv"
    INPUT_VELOCITY_CSV = f"{INPUT_POSITION_CSV}_velocity.csv"
    # 外れ値を除去するかどうか (True: 除去する, False: 除去しない)
    REMOVE_OUTLIERS = True
    # 活動状態の判定に使用する速度の閾値 STATE_THRESHOLD を選択
    # 'q1', 'median_q2', 'q3', 'avg_half' などから閾値のキーを選択(calculate_thresholdsで計算されるもの)
    VELOCITY_THRESHOLD = "avg_half"
    # 接触判定に使用する距離のしきい値DISTANCE＿THRESHOLD を設定
    # CONTACT_THRESHOLD = plot_social_network.CONTACT_THRESHOLD_PIXELS
    CONTACT_THRESHOLD = 50.0  # ピクセル単位の接触しきい値

    # 実行する処理を選択
    DO_PLOT_STATE_DISTRIBUTION = False
    DO_ANALYTICAL_CALCULATION = True

    # 選択された処理のみを実行する
    try:
        if DO_PLOT_STATE_DISTRIBUTION:
            transition_matrices, state_labels, id1, id2 = plot_state_distribution(INPUT_POSITION_CSV, INPUT_VELOCITY_CSV, VELOCITY_THRESHOLD, CONTACT_THRESHOLD, remove_outliers=REMOVE_OUTLIERS, plot_glaph=True)
        else:
            # プロットはスキップしたい場合でも、解析に必要なデータを取得するために関数を呼び出して戻り値を受け取ります。
            print(f"\n{plot_state_distribution}をスキップします。\n状態の計算のみを行います")
            transition_matrices, state_labels, id1, id2 = plot_state_distribution(INPUT_POSITION_CSV, INPUT_VELOCITY_CSV, VELOCITY_THRESHOLD, CONTACT_THRESHOLD, remove_outliers=REMOVE_OUTLIERS, plot_glaph=False)

        if DO_ANALYTICAL_CALCULATION:
            analytical_calculation(transition_matrices, state_labels)
        else:
            print(f"\n{analytical_calculation}をスキップします")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

    print("\n選択されたすべての処理が終了しました")