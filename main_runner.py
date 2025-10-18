import os
import calculate_velocity
import plot_speed_over_time
import plot_velocity_distribution
import plot_msd

# -----------------------------------------------------------------------------
# ◆◆◆ 1. 実行設定 ◆◆◆
# 実行したい分析を True に設定してください。
# -----------------------------------------------------------------------------
RUN_CALCULATE_VELOCITY = True         # True: 位置データから速度を計算する
RUN_PLOT_SPEED_OVER_TIME = True       # True: 時間ごとの速度変化グラフを描画する
RUN_PLOT_VELOCITY_DISTRIBUTION = True # True: 速度の分布（ヒストグラム）を描画する
RUN_PLOT_MSD = True                   # True: MSD（平均二乗変位）を計算・描画する

# -----------------------------------------------------------------------------
# ◆◆◆ 2. パラメータ設定 ◆◆◆
# 分析に使用するファイルや条件を設定してください。
# -----------------------------------------------------------------------------
# --- 基本ファイル設定 ---
# ここでファイルパスの「ベース」部分を設定すれば、下のファイル名は自動で設定されます。
# BASE_FILE_PATH = "d:/analysis_data/20251016_02/20251016_02"
BASE_FILE_PATH = "/Volumes/100.108.13.8/analysis_data/20251014_01/20251014_01-position.csv" # Macでの実行時

# --- 各スクリプトの詳細設定 ---
# 速度計算 (`calculate_velocity.py`) 用
# (通常は変更不要です)
INPUT_POSITION_CSV = f"{BASE_FILE_PATH}"
OUTPUT_VELOCITY_CSV = f"{BASE_FILE_PATH}_velocity.csv"

# 速度グラフ (`plot_speed_over_time.py`, `plot_velocity_distribution.py`) 用
KEY_FOR_THRESHOLD = "median_q2"  # 使用する閾値 ('q1', 'median_q2', 'q3', 'avg_half', または None)
REMOVE_OUTLIERS = True           # 外れ値を除去するか (True / False)

# MSD (`plot_msd.py`) 用
USE_LOGLOG_PLOT = True           # MSDグラフを両対数プロットにするか (True / False)


# -----------------------------------------------------------------------------
# ◆◆◆ 処理の実行 ◆◆◆
# (この下は編集不要です)
# -----------------------------------------------------------------------------
def main():
    """設定に基づいて各分析処理を実行するメイン関数"""
    print("=== 分析処理を開始します ===")

    # --- 1. 速度の計算 ---
    if RUN_CALCULATE_VELOCITY:
        print("\n--- [実行中] 1. 速度計算 ---")
        calculate_velocity.calculate_velocity(INPUT_POSITION_CSV, OUTPUT_VELOCITY_CSV)
    else:
        print("\n--- [スキップ] 1. 速度計算 ---")

    # --- 2. 時間ごとの速度変化グラフの描画 ---
    if RUN_PLOT_SPEED_OVER_TIME:
        print("\n--- [実行中] 2. 時間ごと速度グラフ描画 ---")
        # 速度ファイルが存在するかチェック
        if not os.path.exists(OUTPUT_VELOCITY_CSV):
            print(f"警告: 速度ファイル '{OUTPUT_VELOCITY_CSV}' が見つかりません。")
            print("→ グラフ描画をスキップします。先に速度計算を実行してください。")
        else:
            plot_speed_over_time.plot_speed_over_time(OUTPUT_VELOCITY_CSV, KEY_FOR_THRESHOLD, REMOVE_OUTLIERS)
    else:
        print("\n--- [スキップ] 2. 時間ごと速度グラフ描画 ---")

    # --- 3. 速度分布グラフの描画 ---
    if RUN_PLOT_VELOCITY_DISTRIBUTION:
        print("\n--- [実行中] 3. 速度分布グラフ描画 ---")
        # 速度ファイルが存在するかチェック
        if not os.path.exists(OUTPUT_VELOCITY_CSV):
            print(f"警告: 速度ファイル '{OUTPUT_VELOCITY_CSV}' が見つかりません。")
            print("→ グラフ描画をスキップします。先に速度計算を実行してください。")
        else:
            # plot_velocity_distribution.py 内でグローバル変数を参照しているため、
            # 実行時に設定値を渡すように元のコードを少し変更するか、
            # ここで値を設定する必要があります。今回は関数に引数を渡します。
            plot_velocity_distribution.plot_histogram_dashboard(OUTPUT_VELOCITY_CSV, KEY_FOR_THRESHOLD)
    else:
        print("\n--- [スキップ] 3. 速度分布グラフ描画 ---")

    # --- 4. MSDの計算と描画 ---
    if RUN_PLOT_MSD:
        print("\n--- [実行中] 4. MSD計算・描画 ---")
        # 位置ファイルが存在するかチェック
        if not os.path.exists(INPUT_POSITION_CSV):
            print(f"警告: 位置ファイル '{INPUT_POSITION_CSV}' が見つかりません。")
            print("→ MSD計算をスキップします。")
        else:
            # plot_msd.py 内のグローバル変数 USE_LOGLOG_PLOT を上書き
            plot_msd.USE_LOGLOG_PLOT = USE_LOGLOG_PLOT
            msd_data = plot_msd.calculate_msd_from_wide_format(INPUT_POSITION_CSV)
            if msd_data:
                plot_msd.plot_msd(msd_data, INPUT_POSITION_CSV)
    else:
        print("\n--- [スキップ] 4. MSD計算・描画 ---")
    
    print("\n=== 全ての処理が完了しました ===")

if __name__ == '__main__':
    main()