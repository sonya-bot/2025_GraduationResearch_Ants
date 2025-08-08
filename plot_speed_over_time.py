import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

# 【文字化け対策】日本語フォントを設定
try:
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Meiryo'
    elif platform.system() == 'Darwin': # macOS
        plt.rcParams['font.family'] = 'Hiragino Sans'
    else: # Linux
        plt.rcParams['font.family'] = 'IPAexGothic'
except Exception as e:
    print(f"日本語フォントの設定中にエラーが発生しました: {e}")

def plot_speed_over_time(input_filename, output_filename=None):
    """
    速度・速さ情報ファイルから、各個体の速さの時間変化をプロットする。

    Args:
        input_filename (str): 入力ファイル名 (c00001_velocity_and_speed.csv)
        output_filename (str): 出力するグラフのファイル名
    """
    try:
        df = pd.read_csv(input_filename)
        print(f"'{input_filename}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {input_filename}")
        return
        
    # 'speed'を含む列を探して個体IDを特定する
    speed_cols = [col for col in df.columns if 'speed' in col]
    individual_ids = [col.replace('speed', '') for col in speed_cols]
    n_individuals = len(individual_ids)

    if n_individuals == 0:
        print("グラフ化するspeedデータが見つかりません。")
        return

    # === グラフのレイアウトを自動で決定 ===
    # 個体数に応じて、できるだけ正方形に近いレイアウトにする
    n_cols = int(np.ceil(np.sqrt(n_individuals)))
    n_rows = (n_individuals + n_cols - 1) // n_cols
    
    # 図全体のサイズを定義
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True)
    # グラフが1つの場合でも対応できるように、axesを1次元配列に変換
    axes_flat = axes.flatten() if n_individuals > 1 else [axes]

    # 図全体のタイトル
    fig.suptitle('Speed over Time (individuals)', fontsize=16)

    # 最初のグラフ(axes_flat[0])に、全個体の速さを重ねてプロット
    ax_overlay = axes_flat[0]
    ax_overlay.set_title('All Individuals')
    ax_overlay.set_xlabel('Frame')
    ax_overlay.set_ylabel('Speed')
    ax_overlay.grid(True, linestyle='--', alpha=0.6)

    # ループして、同じaxに色分けしてプロット
    for i_id in individual_ids:
        speed_col_name = f'speed{i_id}'
        ax_overlay.plot(df['position'], df[speed_col_name], linewidth=0.8, label=f'ID: {i_id}')

    # 個体数が多すぎない場合のみ凡例を表示
    if len(individual_ids) <= 10:
        ax_overlay.legend(fontsize='small')

    # 2番目以降のグラフに個別の速さをプロット
    for i, i_id in enumerate(individual_ids):
        # プロットする位置をi+1にずらす
        ax = axes_flat[i + 1] 
        
        speed_col_name = f'speed{i_id}'
        
        ax.plot(df['position'], df[speed_col_name], label=f'Speed of {i_id}')

        ax.set_title(f'Individual ID: {i_id}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Speed')
        ax.grid(True, linestyle='--', alpha=0.6)

    # 余った描画領域を非表示にする
    # ループの開始を n_individuals + 1に修正
    for i in range(n_individuals + 1, len(axes_flat)):
        axes_flat[i].axis('off')

    # グラフの表示(デバッグ用)
    plt.show()

    # # グラフを画像ファイルとして保存
    # try:
    #     plt.savefig(output_filename, dpi=300)
    #     print(f"グラフを '{output_filename}' として保存しました。")
    # except Exception as e:
    #     print(f"ファイルの保存中にエラーが発生しました: {e}")


if __name__ == '__main__':
    # ◆◆◆ 設定 ◆◆◆
    INPUT_CSV = '/Users/sonya/Library/CloudStorage/OneDrive-HiroshimaCityUniversity/2025/UMATracker/datas/c00001(edit_2)-position-velocity.csv'
    # INPUT_CSV = 'C:\Users\13sou\OneDrive - Hiroshima City University\2025\UMATracker\datas\c00001(edit_2)-position-velocity.csv'
    # OUTPUT_PNG = 'speed_over_time.png'

    plot_speed_over_time(INPUT_CSV)