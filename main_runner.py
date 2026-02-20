# import os
# import calculate_velocity
# import plot_speed_over_time
# import plot_velocity_distribution
# import plot_msd
# import plot_social_network
# import plot_distance_over_time
# import plot_COS_Over_Time
# import plot_Contact_Spectrum

# # ◆◆◆ 実行設定 ◆◆◆
# # 実行したい分析を True に設定してください。
# RUN_CALCULATE_VELOCITY = True         # True: 位置データから速度を計算する
# RUN_PLOT_SPEED_OVER_TIME = False       # True: 時間ごとの速度変化グラフを描画する
# RUN_PLOT_VELOCITY_DISTRIBUTION = False # True: 速度の分布（ヒストグラム）を描画する
# RUN_PLOT_MSD = False                   # True: MSD（平均二乗変位）を計算・描画する
# RUN_PLOT_SOCIAL_NETWORK = False        # True: 個体間の接触ネットワークを計算・描画する
# RUN_PLOT_DISTANCE_OVER_TIME = False   # True: 個体ペア間の距離の時間変化グラフを描画する
# RUN_PLOT_COS_OVER_TIME = False        # True: COSの値の時間変化グラフを描画する
# RUN_PLOT_CONTACT_SPECTRUM = True     # True: 接触頻度のパワースペクトルグラフを描画する


# # 入力ファイル
# COLONIES = {
#     "ISO_DICT": {
#         "Colony A": "20251030_01", "Colony B": "20251104_02", "Colony C": "20251106_01",
#         "Colony D": "20251113_01", "Colony E": "20251117_02", "Colony G": "20251119_01",
#         "Colony H": "20251120_02", "Colony I": "20251127_01",
#     },
#     "PAIR_DICT": {
#         "Colony A": "20251030_02", "Colony B": "20251105_01", "Colony C": "20251107_01",
#         "Colony D": "20251113_03", "Colony E": "20251118_01", "Colony G": "20251119_02",
#         "Colony H": "20251121_01", "Colony I": "20251127_02",
#     },
#     "TRIO_DICT": {
#         "Colony A": "20251101_01", "Colony B": "20251105_02", "Colony C": "20251110_01",
#         "Colony D": "20251117_01", "Colony E": "20251118_02", "Colony G": "20251120_01",
#         "Colony H": "20251126_01", "Colony I": "20251128_01",
#     }
# }

# # 基本ファイルパス指定
# PATHS = {
#     "Windows": r"d:/analysis_data",
#     "Mac":  r"/Volumes/100.108.13.8/analysis_data",
#     "Linux":   r"/home/user/analysis_data"
# },

# # パラメータ設定

# # 外れ値の設定
# REMOVE_OUTLIERS = True           # 外れ値を除去するか (True / False)
# REMOVE_THRESHOLD = 200.0         # 外れ値とみなす速度の閾値 (ピクセル/フレーム)

# # 各種閾値の設定
# VELOCITY_THRESHOLD = "median_q2"  # 使用する速度の閾値 ('q1', 'median_q2', 'q3', 'avg_half', または None)
# CONTACT_THRESHOLD = 50.0         # 接触とみなす距離の閾値（ピクセル）,閾値以下の距離を接触とみなす

# # 対数スケールの設定
# USE_LOG_SCALE = True            # 縦軸を対数表示するか (True / False)
#     # MSD (`plot_msd.py`) 用
# USE_LOGLOG_PLOT = True           # MSDグラフを両対数プロットにするか (True / False)

# #グラフの指定
# FIG_SIZE = (10,5) #描画サイズを指定
# AUTO_SAVE = False #グラフの自動保存設定 (True / False)

# # ユーティリティ関数
# def get_path(COLONIES, PATHS):
#     """指定されたコロニータイプと環境に基づいて、位置ファイルと速度ファイルのパスを取得する関数"""
#     # パスの取得
#     PATHS = PATHS["Mac"]
#     base_path = PATHS
#     # パスの生成
#     input_position_path = os.path.join(base_path, COLONIES, f"{COLONIES}-position.csv")
#     input_velocity_path = os.path.join(base_path, COLONIES, f"{COLONIES}-position_velocity.csv")
#     output_path = os.path.join(base_path, COLONIES, f"{COLONIES}-output.csv")
#     print(f"Input Position Path: {input_position_path}, Input Velocity Path: {input_velocity_path}, Output Path: {output_path}")
#     return input_position_path, input_velocity_path, output_path


# -*- coding: utf-8 -*-

import os

# --- 各解析モジュールのインポート ---
# 基礎・空間解析モジュール
import calculate_velocity
import plot_speed_over_time
import plot_msd
import plot_social_network
import plot_distance_over_time
import plot_Activity_Interval
import plot_COS_Over_Time
import plot_State_Duration
import plot_Contact_Duration
import plot_Contact_Spectrum
import plot_Sosial_Rest


# 1.実行コード選択
RUN_CALCULATE_VELOCITY         = False  # 速度の計算
RUN_PLOT_SPEED_OVER_TIME       = False  # 速度の時間変化
RUN_PLOT_MSD                   = False  # 平均二乗変位 (MSD)
RUN_PLOT_SOCIAL_NETWORK        = False  # 接触ネットワーク
RUN_PLOT_DISTANCE_OVER_TIME    = False  # 個体間距離の時間変化

RUN_PLOT_ACTIVITY_INTERVAL     = True   # 活動インターバル解析
RUN_PLOT_COS_OVER_TIME         = True   # 同時活動個体数の時系列と接触イベント
RUN_PLOT_STATE_DURATION        = False  # 活動/非活動状態の持続時間 (CCDF)
RUN_PLOT_CONTACT_DURATION      = False  # 接触持続時間の CCDF
RUN_PLOT_CONTACT_SPECTRUM      = False  # 接触シグナルの周波数解析
RUN_PLOT_SOCIAL_REST           = False  # 社会的休止状態の解析

# 2.共通パラメータ設定

# ベースとなるパス (環境に合わせて変更)
BASE_PATH = "/Users/sonya/卒論データ/analysis_data"
OUTPUT_DIR = "/Users/sonya/卒論データ/analysis_data/graph_output"

# 物理・解析パラメータ
FPS = 2.0
CONTACT_THRESHOLD_PX = 50.0
REMOVE_OUTLIERS = True
VELOCITY_THRESHOLD_KEY = 'avg_half'
AUTO_SAVE = False
FIG_SIZE = (6, 4)
USE_LOG_SCALE =  True  # 速度プロット等でのY軸対数設定

# 出力先ディレクトリの作成
if AUTO_SAVE and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 3.データセットの定義

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

# 実行ループ用のリスト
DATA_SETS = [
    (ISO_DICT, "Isolate", "Isolate"),
    (PAIR_DICT, "Pair", "Pair"),
    (TRIO_DICT, "Trio", "Trio")
]


# 4.メイン処理
def main():
    print("解析プログラム (main_runner) を開始します")
    
    # 全ての処理対象フォルダのリストを生成（基礎解析用）
    all_folders = []
    for d, _, _ in DATA_SETS:
        all_folders.extend(list(d.values()))
    all_folders = list(set(all_folders)) # 重複削除

    for folder_id in all_folders:
        pos_csv = os.path.join(BASE_PATH, folder_id, f"{folder_id}-position.csv")
        vel_csv = os.path.join(BASE_PATH, folder_id, f"{folder_id}-position_velocity.csv")
        
        if not os.path.exists(pos_csv):
            continue

        if RUN_CALCULATE_VELOCITY:
            print(f"[{folder_id}] 速度の計算を実行中...")
            calculate_velocity.process_velocity(pos_csv)

        if RUN_PLOT_SPEED_OVER_TIME and os.path.exists(vel_csv):
            print(f"[{folder_id}] 速度の時間変化グラフを作成中...")
            plot_speed_over_time.plot_speed_over_time(
                vel_csv, REMOVE_OUTLIERS, USE_LOG_SCALE, FIG_SIZE, AUTO_SAVE
            )


        if RUN_PLOT_MSD:
            print(f"[{folder_id}] MSDの計算・描画を実行中...")
            plot_msd.USE_LOGLOG_PLOT = True
            msd_data = plot_msd.calculate_msd_from_wide_format(pos_csv)
            if msd_data:
                plot_msd.plot_msd(msd_data, pos_csv)

        if RUN_PLOT_SOCIAL_NETWORK:
            print(f"[{folder_id}] 接触ネットワークの描画を実行中...")
            plot_social_network.plot_social_network(pos_csv, CONTACT_THRESHOLD_PX)

        if RUN_PLOT_DISTANCE_OVER_TIME:
            print(f"[{folder_id}] 個体間距離の時間変化を実行中...")
            plot_distance_over_time.plot_distance_over_time(pos_csv, CONTACT_THRESHOLD_PX)

    if any([RUN_PLOT_ACTIVITY_INTERVAL, RUN_PLOT_COS_OVER_TIME, RUN_PLOT_STATE_DURATION, RUN_PLOT_CONTACT_DURATION, RUN_PLOT_CONTACT_SPECTRUM, RUN_PLOT_SOCIAL_REST]):
        print("\n--- 高度な状態・時系列解析フェーズ ---")
    
    # 1. 活動インターバル解析
    if RUN_PLOT_ACTIVITY_INTERVAL:
        print("\n[実行中] 活動インターバル解析")
        for source_dict, mode_label, file_suffix in DATA_SETS:
            save_path = os.path.join(OUTPUT_DIR, f"Activity_Interval_{file_suffix}.png")
            plot_Activity_Interval.plot_intervals_analysis(
                source_dict, mode_label, BASE_PATH, REMOVE_OUTLIERS, VELOCITY_THRESHOLD_KEY, FIG_SIZE, save_path, AUTO_SAVE
            )

    # 2. 同時活動個体数 (COS) の時系列
    if RUN_PLOT_COS_OVER_TIME:
        print("\n[実行中] 同時活動個体数の時系列解析 (Pair, Trio)")
        for source_dict, mode_label, file_suffix in DATA_SETS[1:]: # ISOは除外
            save_path = os.path.join(OUTPUT_DIR, f"Activity_Count_{file_suffix}.png")
            plot_COS_Over_Time.plot_activity_count_dashboard(
                source_dict, mode_label, BASE_PATH, FIG_SIZE, VELOCITY_THRESHOLD_KEY, CONTACT_THRESHOLD_PX, AUTO_SAVE, save_path
            )

    # 3. 状態持続時間 (State Duration CCDF)
    if RUN_PLOT_STATE_DURATION:
        print("\n[実行中] 状態持続時間(CCDF)の解析")
        # TODO: plot_State_Duration 側のメインロジックを呼び出す関数を記述

    # 4. 接触持続時間 (Contact Duration)
    if RUN_PLOT_CONTACT_DURATION:
        print("\n[実行中] 接触持続時間(CCDF)の解析")
        # TODO: plot_Contact_Duration 側のメインロジックを呼び出す関数を記述

    # 5. 接触シグナルの周波数解析 (Spectrum)
    if RUN_PLOT_CONTACT_SPECTRUM:
        print("\n[実行中] 接触シグナルの周波数解析")
        # TODO: plot_Contact_Spectrum 側のメインロジックを呼び出す関数を記述

    # 6. 社会的休止状態 (Social Rest)
    if RUN_PLOT_SOCIAL_REST:
        print("\n[実行中] 社会的休止状態の解析")
        # TODO: plot_Sosial_Rest 側のメインロジックを呼び出す関数を記述

    print("\n=== 全ての解析処理が完了しました ===")

if __name__ == '__main__':
    main()