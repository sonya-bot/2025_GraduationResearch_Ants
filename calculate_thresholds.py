import pandas as pd
import numpy as np

def get_threshold_values(input_filename):
    """
    CSVファイルから速度データを読み込み、全個体および各個体の統計量を計算して返す。

    Args:
        input_filename (str): 速度データが含まれるCSVファイル名。

    Returns:
        tuple: (全個体の結果辞書, 各個体の結果リスト) を返す。
               エラー時は (None, None) を返す。

        計算手法:
            平均値 (average)
            第1四分位数 (q1)
            中央値 (median_q2)
            第3四分位数 (q3)
            しきい値 (avg_half, Hayashi,2015に基づく)
    """
    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {input_filename}")
        return None, None

    speed_cols = [col for col in df.columns if col.startswith('speed_')]
    if not speed_cols:
        print("速さのデータ(speed_...)の列が見つかりません。")
        return None, None

    # 各個体の結果を格納するリスト
    individual_results_list = []
    
    # 各個体についてループ処理
    for col_name in speed_cols:
        individual_speeds = df[col_name]
        individual_speeds = individual_speeds[individual_speeds > 0]
        
        # データが空の場合はスキップ
        if individual_speeds.empty:
            continue

        individual_id = col_name.replace('speed_', '')
        
        # 各個体の結果を辞書にまとめる
        individual_result = {
            'id': individual_id,
            'average': individual_speeds.mean(),
            'q1': individual_speeds.quantile(0.25),
            'median_q2': individual_speeds.quantile(0.50),
            'q3': individual_speeds.quantile(0.75),
            'avg_half': individual_speeds.mean() / 2
        }
        individual_results_list.append(individual_result)

    # 全個体の計算
    all_speeds_flat = df[speed_cols].values.flatten()
    all_speeds = all_speeds_flat[all_speeds_flat > 0]
    
    overall_results = {}
    if all_speeds.size > 0:
        overall_results = {
            'average': np.mean(all_speeds),
            'q1': np.quantile(all_speeds, 0.25),
            'median_q2': np.quantile(all_speeds, 0.50),
            'q3': np.quantile(all_speeds, 0.75),
            'avg_half': np.mean(all_speeds) / 2
        }

    return overall_results, individual_results_list

# このファイルが直接実行された時だけ、結果を表示する
if __name__ == '__main__':
    INPUT_CSV = '/Users/sonya/Library/CloudStorage/OneDrive-HiroshimaCityUniversity/2025/UMATracker/datas/c00001(edit_2)-position-velocity.csv'
    
    # 関数を呼び出し、全体の結果と個別の結果をそれぞれ受け取る
    overall_data, individual_data = get_threshold_values(INPUT_CSV)

    # 結果が正常に取得できたら表示
    if overall_data and individual_data:
        print("--各個体の計算結果--")
        for res in individual_data:
            print(f"  個体ID {res['id']}:")
            print(f"    - 平均速度: {res['average']:.4f}")
            print(f"    - 第一四分位数 (Q1): {res['q1']:.4f}")
            print(f"    - 中央値 (Q2): {res['median_q2']:.4f}")
            print(f"    - 第三四分位数 (Q3): {res['q3']:.4f}")
            print(f"    - しきい値 (平均/2): {res['avg_half']:.4f}")
            print("-" * 20)
        
        print("\n--全固体の計算結果--")
        print(f"  -全個体の平均速度: {overall_data['average']:.4f}")
        print(f"  -全個体の第一四分位数 (Q1): {overall_data['q1']:.4f}")
        print(f"  -全個体の中央値 (Q2): {overall_data['median_q2']:.4f}")
        print(f"  -全個体の第三四分位数 (Q3): {overall_data['q3']:.4f}")
        print(f"  -全個体の統一しきい値 (平均/2): {overall_data['avg_half']:.4f}")
        print("-" * 20)
