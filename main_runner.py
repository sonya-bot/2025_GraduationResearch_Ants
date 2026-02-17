import os
import calculate_velocity
import plot_speed_over_time
import plot_velocity_distribution
import plot_msd
import plot_social_network
import plot_distance_over_time
import plot_COS_Over_Time
import plot_Contact_Spectrum

# ◆◆◆ 実行設定 ◆◆◆
# 実行したい分析を True に設定してください。
RUN_CALCULATE_VELOCITY = True         # True: 位置データから速度を計算する
RUN_PLOT_SPEED_OVER_TIME = False       # True: 時間ごとの速度変化グラフを描画する
RUN_PLOT_VELOCITY_DISTRIBUTION = False # True: 速度の分布（ヒストグラム）を描画する
RUN_PLOT_MSD = False                   # True: MSD（平均二乗変位）を計算・描画する
RUN_PLOT_SOCIAL_NETWORK = False        # True: 個体間の接触ネットワークを計算・描画する
RUN_PLOT_DISTANCE_OVER_TIME = False   # True: 個体ペア間の距離の時間変化グラフを描画する
RUN_PLOT_COS_OVER_TIME = False        # True: COSの値の時間変化グラフを描画する
RUN_PLOT_CONTACT_SPECTRUM = True     # True: 接触頻度のパワースペクトルグラフを描画する


# 入力ファイル
COLONIES = {
    "ISO_DICT": {
        "Colony A": "20251030_01", "Colony B": "20251104_02", "Colony C": "20251106_01",
        "Colony D": "20251113_01", "Colony E": "20251117_02", "Colony G": "20251119_01",
        "Colony H": "20251120_02", "Colony I": "20251127_01",
    },
    "PAIR_DICT": {
        "Colony A": "20251030_02", "Colony B": "20251105_01", "Colony C": "20251107_01",
        "Colony D": "20251113_03", "Colony E": "20251118_01", "Colony G": "20251119_02",
        "Colony H": "20251121_01", "Colony I": "20251127_02",
    },
    "TRIO_DICT": {
        "Colony A": "20251101_01", "Colony B": "20251105_02", "Colony C": "20251110_01",
        "Colony D": "20251117_01", "Colony E": "20251118_02", "Colony G": "20251120_01",
        "Colony H": "20251126_01", "Colony I": "20251128_01",
    }
}

# 基本ファイルパス指定
PATHS = {
    "Windows": r"d:/analysis_data",
    "Mac":  r"/Volumes/100.108.13.8/analysis_data",
    "Linux":   r"/home/user/analysis_data"
},

# パラメータ設定

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
AUTO_SAVE = False #グラフの自動保存設定 (True / False)

# ユーティリティ関数
def get_path(COLONIES, PATHS):
    """指定されたコロニータイプと環境に基づいて、位置ファイルと速度ファイルのパスを取得する関数"""
    # パスの取得
    PATHS = PATHS["Mac"]
    base_path = PATHS
    # パスの生成
    input_position_path = os.path.join(base_path, COLONIES, f"{COLONIES}-position.csv")
    input_velocity_path = os.path.join(base_path, COLONIES, f"{COLONIES}-position_velocity.csv")
    output_path = os.path.join(base_path, COLONIES, f"{COLONIES}-output.csv")
    print(f"Input Position Path: {input_position_path}, Input Velocity Path: {input_velocity_path}, Output Path: {output_path}")
    return input_position_path, input_velocity_path, output_path
