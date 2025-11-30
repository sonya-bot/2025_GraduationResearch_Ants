# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import numpy.fft as fft # ▼ 要件定義(v5) 6. Numpy FFTをインポート
import matplotlib.pyplot as plt
import platform
import os
from itertools import combinations 

def plot_contact_spectrum(position_csv_path, contact_threshold, fig_size, auto_save, use_loglog_plot):
    """
    入力データの確認及び
    個体数のカウント,
    FPSの定義を行う
    (動作の共通化)
    """
# 1.データ入力,出力
    # 入力
    try:
        #
        df_pos = pd.read_csv(position_csv_path)
        print(f"'{position_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return
    # 出力
    output_dir = os.path.dirname(position_csv_path)
    
# 2.FPSの定義
    FPS = 2.0 # 1フレーム=1/2秒
    sample_spacing_second = 1.0 / FPS # 周波数計算用にサンプリング間隔(秒)を計算
    # 時間を時間(秒)から時間(分)に変更
    sample_spacing_minutes = sample_spacing_second / 60 
    # print(f"  - サンプリング周波数: {FPS} Hz (サンプリング間隔: {sample_spacing_minutes} m)")

# 3.個体IDの特定と個体ごとの反復処理
    x_cols = [col for col in df_pos.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]

    n_individuals = len(individual_ids)
    if n_individuals <= 1:
        print(f"エラー: 検出された個体数が {n_individuals} のため、グラフを作成できません。")
        print("プログラムを終了します")
        return
    elif n_individuals == 2:
        print("2匹の個体が検出されました。2匹用のスペクトラムグラフを作成します。")
        plot_pair_contact_spectrum(df_pos, individual_ids, sample_spacing_minutes, contact_threshold, fig_size, use_loglog_plot, auto_save, output_dir)
    elif n_individuals ==3:
        print("3匹の個体が検出されました。3匹用のスペクトラムグラフを作成します。")
        plot_trio_contact_spectrum(df_pos, individual_ids, sample_spacing_minutes, contact_threshold, fig_size, use_loglog_plot, auto_save, output_dir)



def calculate_power_spectrum(contact_signal, sample_spacing_minutes):
    """ 
    二値シグナル(接触/非接触)からパワースペクトル(dB)を計算 
    """
# 5.フーリエ変換の実行
    # サンプリング数
    N = len(contact_signal)

    # 実フーリエ変換
    F = np.fft.rfft(contact_signal) * (2/N)

    # 周波数軸の値を計算
    freq_per_min = fft.rfftfreq(N, d=sample_spacing_minutes)

    # 周波数スペクトルの複素数を絶対値に変換
    F_abs = np.abs(F)
    # 見やすいように常用対数に変換
    F_log = np.log10(F_abs)

    # 6. 1/fノイズと1/f^2ノイズの基準線を計算
    # 計算用: 直流成分(0Hz)を除いた周波数データを使用
    valid_idx = freq_per_min > 0
    valid_freqs = freq_per_min[valid_idx]
    
    # 基準点（アンカー）の設定: データの最低周波数成分のパワーに合わせる
    # 直流成分(index 0)の次は index 1
    ref_power = F_log[1] 
    ref_freq = valid_freqs[0] # 対応する周波数 (=freq_per_min[1])

    # 1/f (Pink Noise) の傾き: log(P) = -1 * log(f) + C
    # 基準点 (log(ref_freq), ref_power) を通るように C を決定
    # ref_power = -1 * log10(ref_freq) + C  =>  C = ref_power + log10(ref_freq)
    # y = -log10(f) + ref_power + log10(ref_freq)
    #   = - (log10(f) - log10(ref_freq)) + ref_power
    slope_pink = -1.0 * (np.log10(valid_freqs) - np.log10(ref_freq)) + ref_power

    # 1/f^2 (Brown Noise/Random Walk) の傾き: log(P) = -2 * log(f) + C
    slope_brown = -2.0 * (np.log10(valid_freqs) - np.log10(ref_freq)) + ref_power


    return freq_per_min, F_log, valid_freqs, slope_pink, slope_brown


def draw_graph(freq_per_min, F_log, fig_size, use_loglog_plot, auto_save, title_text, save_path):
    """
    グラフの描画を行う
    """
# 6.グラフの描画
    fig, ax = plt.subplots(figsize=fig_size) # ax is defined here
    
    # [線グラフ] 振幅スペクトルを描画
    ax.plot(freq_per_min, F_log, linewidth=1.0)
    
    # グラフの体裁
    ax.set_title(title_text, fontsize=14)
    
    
    # X軸の表示範囲を調整 (0 Hz (直流成分) を除外して表示)
    # (0.01 回/分 から表示)
    # ax.set_xlim(0.01, freq_per_min.max()) 
    # ax.set_xlim(-1,10)
    
    # Y軸 (対数変換済みのため、スケールは 'linear')
    ax.set_ylabel('Amplitude(Log Scale)', fontsize=12) #
    
    ax.grid(True, linestyle='--', alpha=0.6)

        # 対数スケールの設定
    if use_loglog_plot:
        # ax.set_xlim(0.1,100)           
        # ax.set_xscale('log')
        ax.set_xlabel('Frequency(Log Scale) [/min]', fontsize=12) #
    else:
        # ax.set_xlim(0)
        ax.set_xlabel('Frequency [/min]', fontsize=12)

    # ノイズを示す線を追記
    # [基準線] 1/f (Pink Noise)
    ax.plot(valid_freqs, slope_pink, color='gray', linestyle='--', linewidth=1.2, 
            label='1/f (Pink Noise)', alpha=0.8, zorder=2)
    
    # [基準線] 1/f^2 (Brown Noise)
    ax.plot(valid_freqs, slope_brown, color='red', linestyle=':', linewidth=1.2, 
            label='1/f² (Brown Noise/Random Walk)', alpha=0.8, zorder=2)# [基準線] 1/f (Pink Noise)
    
    ax.legend(loc='upper right', fontsize=10)
        

    # 保存または表示
    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" - グラフを保存しました: {save_path}")
        plt.close()
    else:
        print(f" - グラフを表示します: {title_text}")
        plt.show()
    

def plot_pair_contact_spectrum(df_pos, individual_ids, sample_spacing_minutes, contact_threshold, fig_size, use_loglog_plot, auto_save, save_path):
    """
    位置データからペア間の接触シグナルを生成し、
    Numpy FFT を使って2個体の接触頻度パワースペクトルを計算・描画する。
    """
# 4.接触の判定
    pair_combinations = list(combinations(individual_ids, 2))
    for id1, id2 in pair_combinations:
        print(f"\n処理中: ペア (ID: {id1}, ID: {id2})")


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
        contact_frames = frames[distances <= contact_threshold]
        
        # 距離が distance_threshold 以下のフレームを 1 (接触), それ以外を 0 (非接触) とする
        contact_signal = (distances <= contact_threshold).astype(float)
        
        total_frames = len(df_pos) #
        contact_frames_count = contact_signal.sum()
                
        print(f"  - 接触判定 (全 {total_frames} Frame) を実行しました。")
    
# 5.フーリエ変換の実行
        freq_per_min,F_log = calculate_power_spectrum(contact_signal, sample_spacing_minutes)
        print(f"  - 振幅スペクトル (X軸: 回/分, Y軸: Log Amplitude) を計算しました。")

# 6.グラフの描画
        # タイトルと保存パスを生成して渡す
        title_text = f"Contact Spectrum (Pair,N={len(individual_ids)})\n(Pair ID:{id1},{id2})"
        save_title = f"Contact_Spectrum (Pair,ID:{id1},{id2}).png"
        save_path = os.path.join(save_path, save_title)
        draw_graph(freq_per_min, F_log, fig_size, use_loglog_plot, auto_save, title_text, save_path) # Removed ax from the call
        print(f"  - グラフを描画しました。")

print("全てのペアの処理が完了しました")



# メイン処理
if __name__ == "__main__":
    # 位置データの入力
    INPUT_CSV = "20251107_01"
    INPUT_POSITION_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position.csv"
    # 接触判定に使用するしきい値
    CONTACT_THRESHOLD = 50.0  # ピクセル単位の接触しきい値
    FIG_SIZE = (6, 4)
    USE_LOGLOG_PLOT = True
    AUTO_SAVE = False

    # 関数を呼び出し
    plot_contact_spectrum(INPUT_POSITION_CSV, CONTACT_THRESHOLD, FIG_SIZE, USE_LOGLOG_PLOT, AUTO_SAVE)
