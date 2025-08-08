import pandas as pd
import matplotlib.pyplot as plt

# ◆◆◆ 設定: ご自身のPCのファイルパスに書き換えてください ◆◆◆
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

# 速度データが含まれるCSVファイルのフルパス
VELOCITY_FILE_PATH = '/Users/sonya/Library/CloudStorage/OneDrive-HiroshimaCityUniversity/2025/UMATracker/datas/c00001(edit_2)-position-velocity.csv'

# 元の位置データが含まれるCSVファイルのフルパス
POSITION_FILE_PATH = '/Users/sonya/Library/CloudStorage/OneDrive-HiroshimaCityUniversity/2025/UMATracker/datas/c00001(edit_2)-position.csv'

# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# 調査したい個体のID
INDIVIDUAL_ID_TO_CHECK = 6

# ----------------------------------------------------------------

def find_speed_outlier(vel_file, pos_file, individual_id):
    """
    指定された個体の速度データから、最大値を特定してその前後の位置情報を表示する。
    """
    column_to_check = f'speed_{individual_id}'
    pos_x_col = f'x{individual_id}'
    pos_y_col = f'y{individual_id}'

    try:
        df_velocity = pd.read_csv(vel_file)
        df_position = pd.read_csv(pos_file)
        print(f"'{vel_file}' を正常に読み込みました。")
        print(f"'{pos_file}' を正常に読み込みました。")

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。コード内のファイルパスを再度確認してください。")
        print(f"詳細: {e}")
        return
    except Exception as e:
        print(f"ファイルの処理中にエラーが発生しました: {e}")
        return

    if column_to_check not in df_velocity.columns:
        print(f"エラー: 速度ファイルに '{column_to_check}' という列が見つかりません。")
        return

    outlier_index = df_velocity[column_to_check].idxmax()
    max_speed = df_velocity.loc[outlier_index, column_to_check]
    frame_number = df_velocity.loc[outlier_index, 'position']
    
    print("\n--- 異常値の検出結果 ---")
    print(f"個体ID: {individual_id}")
    print(f"異常な速度が記録されたフレーム: {frame_number}番")
    print(f"記録された最大速度: {max_speed:.2f}")
    print("------------------------\n")

    print("原因と思われる、位置情報の急激な変化：")
    
    prev_index = outlier_index - 1
    if prev_index >= 0:
        prev_pos_x = df_position.loc[prev_index, pos_x_col]
        prev_pos_y = df_position.loc[prev_index, pos_y_col]
        print(f"フレーム {frame_number - 1}番の位置: x={prev_pos_x:.2f}, y={prev_pos_y:.2f}")

    current_pos_x = df_position.loc[outlier_index, pos_x_col]
    current_pos_y = df_position.loc[outlier_index, pos_y_col]
    print(f"フレーム {frame_number}番の位置: x={current_pos_x:.2f}, y={current_pos_y:.2f}  <-- ここで座標が飛んでいる可能性")
    print("------------------------")

def plot_single_individual(input_filename,individual_id):
    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"エラー: '{input_filename}'が見つかりません。ステップ2を先に実行してください。")
        return
        
    speed_col = f'speed_{individual_id}'
    if speed_col not in df.columns:
        print(f"エラー: '{speed_col}'列が見つかりません。")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df['position'], df[speed_col])
    
    ax.set_title(f'個体ID: {individual_id} の速さの詳細グラフ', fontsize=16)
    ax.set_xlabel('時間 (Frame)', fontsize=12)
    ax.set_ylabel('速さ (Speed)', fontsize=12)
    ax.grid(True, linestyle='--')
    
    # データをテキストで確認するため、最大値を表示
    max_speed = df[speed_col].max()
    ax.text(0.95, 0.95, f'Max Speed: {max_speed:.2f}', 
            transform=ax.transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.show()
    # plt.savefig(output_filename, dpi=150)
    # print(f"ID {individual_id}のみのグラフ '{output_filename}' を作成しました。")

if __name__ == '__main__':
    find_speed_outlier(VELOCITY_FILE_PATH, POSITION_FILE_PATH, INDIVIDUAL_ID_TO_CHECK)
    plot_single_individual(VELOCITY_FILE_PATH, individual_id=INDIVIDUAL_ID_TO_CHECK)