import os
import calculate_velocity
import plot_speed_over_time
import plot_velocity_distribution
import plot_msd
import plot_social_network
import plot_distance_over_time
import plot_COS_Over_Time
import plot_Contact_Spectrum

# -----------------------------------------------------------------------------
# ◆◆◆ 1. 実行設定 ◆◆◆
# 実行したい分析を True に設定してください。
# -----------------------------------------------------------------------------
RUN_CALCULATE_VELOCITY = True         # True: 位置データから速度を計算する
RUN_PLOT_SPEED_OVER_TIME = True       # True: 時間ごとの速度変化グラフを描画する
RUN_PLOT_VELOCITY_DISTRIBUTION = True # True: 速度の分布（ヒストグラム）を描画する
RUN_PLOT_MSD = True                   # True: MSD（平均二乗変位）を計算・描画する
RUN_PLOT_SOCIAL_NETWORK = True        # True: 個体間の接触ネットワークを計算・描画する
RUN_PLOT_DISTANCE_OVER_TIME = True   # True: 個体ペア間の距離の時間変化グラフを描画する
RUN_PLOT_COS_OVER_TIME = True        # True: COSの値の時間変化グラフを描画する
RUN_PLOT_CONTACT_SPECTRUM = True     # True: 接触頻度のパワースペクトルグラフを描画する

# -----------------------------------------------------------------------------
# ◆◆◆ 2. パラメータ設定 ◆◆◆
# -----------------------------------------------------------------------------
# --- 基本ファイル設定 ---
# ここで実行ファイルの日付を入力
INPUT_CSV = "20251030_02"

# --- 各スクリプトの詳細設定 ---
# CSVファイルの設定
INPUT_POSITION_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position.csv"
INPUT_VELOCITY_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position_velocity.csv"

# 外れ値の設定
REMOVE_OUTLIERS = True           # 外れ値を除去するか (True / False)
REMOVE_THRESHOLD = 200.0         # 外れ値とみなす速度の閾値 (ピクセル/フレーム)

# 各種閾値の設定
VELOCITY_THRESHOLD = "median_q2"  # 使用する速度の閾値 ('q1', 'median_q2', 'q3', 'avg_half', または None)
CONTACT_THRESHOLD = 50.0         # 接触とみなす距離の閾値（ピクセル）,閾値以下の距離を接触とみなす

# 対数スケールの設定
USE_LOG_SCALE = True            # 縦軸を対数表示するか (True / False)
    # MSD (`plot_msd.py`) 用
USE_LOGLOG_PLOT = True           # MSDグラフを両対数プロットにするか (True / False)

#グラフの指定
FIG_SIZE = (10,5) #描画サイズを指定
AUTO_SAVE = True #グラフの自動保存設定 (True / False)



# -----------------------------------------------------------------------------
# ◆◆◆ 処理の実行 ◆◆◆
# -----------------------------------------------------------------------------
def main():
    """設定に基づいて各分析処理を実行するメイン関数"""
    print("=== 分析処理を開始します ===")
    # --- 0. 入力ファイルが存在するかチェック ---
    if not os.path.exists(INPUT_POSITION_CSV):
        print(f"エラー: 位置ファイル '{INPUT_POSITION_CSV}' が見つかりません。")
        print("→ すべての処理を中止します。")
        return
    
    # --- 1. 速度の計算 ---
    if os.path.exists(INPUT_VELOCITY_CSV):
        print(f"速度ファイルが既に存在しているため、速度計算をスキップします。")
        RUN_CALCULATE_VELOCITY = False
        print("\n--- [スキップ] 1. 速度計算 ---")
    else:
        RUN_CALCULATE_VELOCITY = True
        print("\n--- [実行中] 1. 速度計算 ---")
        calculate_velocity.calculate_velocity(INPUT_POSITION_CSV, INPUT_VELOCITY_CSV)

    # --- 2. 時間ごとの速度変化グラフの描画 ---
    if RUN_PLOT_SPEED_OVER_TIME:
        plot_speed_over_time.plot_speed_over_time(INPUT_VELOCITY_CSV, REMOVE_OUTLIERS, REMOVE_THRESHOLD, USE_LOG_SCALE, FIG_SIZE, AUTO_SAVE)
    else:
        print("\n--- [スキップ] 2. 速度変化グラフ描画 ---")
        pass

    # --- 3. 速度分布グラフの描画 ---
    if RUN_PLOT_VELOCITY_DISTRIBUTION:
        print("\n--- [実行中] 3. 速度分布グラフ描画 ---")
        plot_velocity_distribution.plot_histogram_dashboard(INPUT_VELOCITY_CSV, REMOVE_OUTLIERS, REMOVE_THRESHOLD, VELOCITY_THRESHOLD, USE_LOG_SCALE, FIG_SIZE, AUTO_SAVE)
    else:
        print("\n--- [スキップ] 3. 速度分布グラフ描画 ---")
        pass

    # --- 4. MSDの計算と描画 ---
    if RUN_PLOT_MSD:
        print("\n--- [実行中] 4. MSD計算・描画 ---")
        df = plot_msd.load_data(INPUT_POSITION_CSV)
        if df is not None:
            msd_results = plot_msd.calculate_msd_from_wide_format(INPUT_POSITION_CSV, max_lag_ratio=0.5)
            plot_msd.plot_msd(msd_results, INPUT_POSITION_CSV, USE_LOGLOG_PLOT, FIG_SIZE, AUTO_SAVE)
    else:
        print("\n--- [スキップ] 4. MSD計算・描画 ---")
        pass


    # --- 5. 個体感ネットワークの計算と描画 ---
    if RUN_PLOT_SOCIAL_NETWORK:
        print("\n--- [実行中] 5. 個体ネットワーク描画 ---")
        plot_social_network.plot_social_network(INPUT_POSITION_CSV, CONTACT_THRESHOLD, FIG_SIZE, AUTO_SAVE)
    else:
        print("\n--- [スキップ] 5. 個体ネットワーク描画 ---")
        pass

    # --- 6. 個体間距離の時間変化グラフの描画 ---
    if RUN_PLOT_DISTANCE_OVER_TIME:
        print("\n--- [実行中] 6. 個体間距離時間変化グラフ描画 ---")
        plot_distance_over_time.plot_distance_over_time(INPUT_POSITION_CSV, CONTACT_THRESHOLD, USE_LOG_SCALE, FIG_SIZE, AUTO_SAVE)
    else:
        print("\n--- [スキップ] 6. 個体間距離時間変化グラフ描画 ---")
        pass

    # --- 7. COSの値の時間変化グラフの描画 ---
    if RUN_PLOT_COS_OVER_TIME:
        print("\n--- [実行中] 7. COSの値の時間変化グラフの描画 ---")
        plot_COS_Over_Time.plot_cos_over_time(INPUT_POSITION_CSV, INPUT_VELOCITY_CSV, VELOCITY_THRESHOLD, CONTACT_THRESHOLD, REMOVE_OUTLIERS, FIG_SIZE, AUTO_SAVE)
    else:
        print("\n--- [スキップ] 7. COSの値の時間変化グラフの描画 ---")

    # --- 8. 接触頻度のパワースペクトルグラフの描画 ---
    if RUN_PLOT_CONTACT_SPECTRUM:
            print("\n--- [実行中] 8. 接触頻度のパワースペクトルグラフの描画 ---")
            plot_Contact_Spectrum.plot_contact_spectrum(INPUT_POSITION_CSV, CONTACT_THRESHOLD, FIG_SIZE, AUTO_SAVE)
    else:
        print("\n--- [スキップ] 8. 接触頻度のパワースペクトルグラフの描画 ---")

if __name__ == '__main__':
    main()
    print("\nすべての処理が終了しました。")