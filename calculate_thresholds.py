import pandas as pd
import numpy as np

def calculate_activity_thresholds(input_filename):
    """
    論文の手法に基づき、各個体および全体の活動/非活動のしきい値を計算する。
    しきい値 = 平均速度 / 2 (Hayashi,2015に基づく)
    さらに、データの分布を把握するために四分位数を計算する。
    """
    try:
        df = pd.read_csv(input_filename)
        print(f"'{input_filename}'を正常に読み込みました。\n")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {input_filename}")
        return

    # 'speed_'で始まる列を検索して個体IDを特定
    speed_cols = [col for col in df.columns if col.startswith('speed_')]
    if not speed_cols:
        print("速さのデータ(speed_...)の列が見つかりません。")
        return

    print("--各個体の計算結果--")

    # 各個体についてループ処理
    for col_name in speed_cols:
        individual_speeds = df[col_name]
        # 0より大きい（動いている）データのみを計算対象にする
        individual_speeds = individual_speeds[individual_speeds > 0]

        # 平均値の計算
        average_speed = individual_speeds.mean()
        # しきい値の計算
        threshold = average_speed / 2
        # 四分位数の計算 (中央値も含む)
        q1_speed = individual_speeds.quantile(0.25)
        median_speed = individual_speeds.quantile(0.50) # Q2
        q3_speed = individual_speeds.quantile(0.75)

        individual_id = col_name.replace('speed_', '')
        print(f"  個体ID {individual_id}:")
        print(f"    - 平均速度: {average_speed:.4f}")
        print(f"    - 第一四分位数 (Q1): {q1_speed:.4f}")
        print(f"    - 中央値 (Q2): {median_speed:.4f}")
        print(f"    - 第三四分位数 (Q3): {q3_speed:.4f}")
        print(f"    - しきい値 (平均/2): {threshold:.4f}")
        print("-" * 20)

    # 全固体についての計算
    print("\n--全固体の計算結果--")

    # 全ての速さデータを1つのリストにまとめる
    all_speeds_flat = df[speed_cols].values.flatten()

    # 0より大きい（動いている）データのみを計算対象にする
    all_speeds = all_speeds_flat[all_speeds_flat > 0]
    
    # 全体の平均速度を計算
    overall_average_speed = all_speeds.mean()
    # 全体のしきい値を計算
    overall_threshold = overall_average_speed / 2

    # === ここから追記 ===
    # 全体の四分位数を計算
    overall_q1 = np.quantile(all_speeds, 0.25)
    overall_median = np.quantile(all_speeds, 0.50) # Q2
    overall_q3 = np.quantile(all_speeds, 0.75)

    print(f"  全個体の平均速度: {overall_average_speed:.4f}")
    print(f"  -全個体の第一四分位数 (Q1): {overall_q1:.4f}")
    print(f"  -全個体の中央値 (Q2): {overall_median:.4f}")
    print(f"  -全個体の第三四分位数 (Q3): {overall_q3:.4f}")
    print(f"  -全個体の統一しきい値 (平均/2): {overall_threshold:.4f}")
    print("-" * 20)


if __name__ == '__main__':
    # ◆◆◆ 設定 ◆◆◆
    # お客様の環境に合わせたファイルパス
    INPUT_CSV = '/Users/sonya/Library/CloudStorage/OneDrive-HiroshimaCityUniversity/2025/UMATracker/datas/c00001(edit_2)-position-velocity.csv'
    
    calculate_activity_thresholds(INPUT_CSV)

