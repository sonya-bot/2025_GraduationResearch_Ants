import pandas as pd
import numpy as np
import

def calculate_velocity(input_filename, output_filename):
    """
    位置情報のCSVファイルから各個体の速度を計算し、
    入力ファイルと同じヘッダー形式で新しいCSVファイルに出力する。

    Args:
        input_filename (str): 入力する位置情報CSVファイル名。
        output_filename (str): 出力する速度情報CSVファイル名。
    """
    if os.path.exists(output_filename):
        print(f"出力ファイル '{output_filename}' は既に存在しています。速度計算をスキップします。")
        return

    try:
        df_input = pd.read_csv(input_filename)
        print(f"'{input_filename}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {input_filename}")
        return

    # 速度を格納するための新しいDataFrameを準備
    df_output = pd.DataFrame()
    
    # 出力ファイルの1列目の列名を、入力ファイルと同じ 'position' に設定
    df_output['position'] = df_input['position']

    # 個体番号を特定
    x_cols = [col for col in df_input.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]
    
    print(f"{len(individual_ids)} 個体の速度を計算します...")

    # 各個体の速度を計算
    for i_id in individual_ids:
        pos_x_col = f'x{i_id}'
        pos_y_col = f'y{i_id}'

        # 1. 速度のx, y成分を計算 (1フレーム前の位置との差分)
        vx = df_input[pos_x_col].diff()
        vy = df_input[pos_y_col].diff()

        # 2. 速さ(Speed)を計算
        speed = np.sqrt(vx**2 + vy**2)

        # 3. DataFrameに vx, vy, speed の3列を追加
        df_output[f'velocity_x{i_id}'] = vx
        df_output[f'velocity_y{i_id}'] = vy
        df_output[f'speed_{i_id}'] = speed

    # 最初のフレームの速度(NaN)を0で埋める
    df_output.fillna(0, inplace=True)

    print(df_output.head())  # デバッグ用に最初の数行を表示

    # 結果を新しいCSVファイルに出力
    try:
        df_output.to_csv(output_filename, index=False)
        print(f"計算が完了し、速度情報を '{output_filename}' に保存しました。")
    except Exception as e:
        print(f"ファイルの保存中にエラーが発生しました: {e}")


if __name__ == '__main__':
    # ◆◆◆ 設定 ◆◆◆
    # INPUT_CSV = "d:/analysis_data/20251016_02/20251016_02-position.csv"
    INPUT_CSV = "/Volumes/100.108.13.8/analysis_data/20251014_01/20251014_01-position.csv" # Macでの実行時
    OUTPUT_CSV = f'{INPUT_CSV.replace("position", "position_velocity")}'
    
    calculate_velocity(INPUT_CSV, OUTPUT_CSV)