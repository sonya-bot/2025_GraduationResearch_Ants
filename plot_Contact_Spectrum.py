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
    位置データからペア間の接触シグナルを生成し、
    Numpy FFT を使ってパワースペクトルを計算・描画する。
    """

# 1.データ入力
    try:
        #
        df_pos = pd.read_csv(position_csv_path)
        print(f"'{position_csv_path}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return
    
# 2.個体IDの特定と個体ごとの反復処理
    x_cols = [col for col in df_pos.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]

    n_individuals = len(individual_ids)
    if n_individuals <= 1:
        print(f"エラー: 検出された個体数が {n_individuals} のため、グラフを作成できません。")
        print("プログラムを終了します")
        return
    
    # 全ての個体のペアについてループ処理
    pair_combinations = list(combinations(individual_ids, 2))

    print(f"{n_individuals} 個体を対象に、全{len(pair_combinations)}ペアのスペクトルグラフを作成します。")

# 3.FPSの定義
    FPS = 2.0 # 1フレーム=1/2秒
    sample_spacing_second = 1.0 / FPS # 周波数計算用にサンプリング間隔(秒)を計算
    # 時間を時間(秒)から時間(分)に変更
    sample_spacing_minutes = sample_spacing_second / 60 
    print(f"  - サンプリング周波数: {FPS} Hz (サンプリング間隔: {sample_spacing_minutes} m)")

# 4.接触の判定
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
        contact_signal = (distances <= contact_threshold).astype(int)
        
        total_frames = len(df_pos) #
        contact_frames_count = contact_signal.sum()
                
        print(f"  - 接触判定 (全 {total_frames} Frame) を実行しました。")
    
# 5.フーリエ変換の実行
        # サンプリング数
        N = len(contact_signal)

        # 実フーリエ変換
        F = np.fft.rfft(contact_signal) * (2/N)

        # 周波数軸の値を計算
        freq_per_min = fft.rfftfreq(N, d=sample_spacing_minutes)

        # # 周波数スペクトルの複素数を絶対値に変換
        F_abs = np.abs(F)
        # Fkの値は絶対値ではなく2乗値を使用(11/14ゼミ)
        F_squared = F_abs**2
        # 見やすいように常用対数に変換
        F_log = np.log10(F_squared)

        print(f"  - 振幅スペクトル (X軸: 回/分, Y軸: Log Amplitude) を計算しました。")

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

# 7.グラフの描画
        plt.figure(figsize=fig_size)
        ax = plt.gca() # 新しいFigureのAxesを取得
        
        # [線グラフ] 振幅スペクトルを描画
        ax.plot(freq_per_min, F_log, label='Contact Spectrum',linewidth=1.0)
        
        # グラフの体裁
        ax.set_title(f'Contact Spectrum (ID:{id1} , {id2})', fontsize=14)
        
        # 対数スケールの設定(y軸は常用対数変換済み)
        if use_loglog_plot:
            ax.set_xscale('log') # X軸も対数スケールに設定
            ax.set_xlabel('Log Frequency (/min)', fontsize=12) # X軸 (対数スケール)
        else:
            ax.set_xlabel('Frequency (/min)', fontsize=12) # X軸 (線形スケール)
        
        # X軸の表示範囲を調整 (0 Hz (直流成分) を除外して表示)
        # (0.01 回/分 から表示)
        ax.set_xlim(0.01, 100) 
        # ax.set_xlim(-1,10)
        
        # Y軸 (対数変換済みのため、スケールは 'linear')
        ax.set_ylabel('Log Amplitude', fontsize=12) 
        # 下限を少し下げて表示(下限は-5.5まで)
        ax.set_ylim(max(F_log.min()-0.5, -12.0), F_log.max()+0.5)
        
        ax.grid(True, linestyle='--', alpha=0.6)

        # ノイズを示す線を追記
        # [基準線] 1/f (Pink Noise)
        ax.plot(valid_freqs, slope_pink, color='gray', linestyle='--', linewidth=1.2, 
                label='1/f (Pink Noise)', alpha=0.8, zorder=2)
        
        # [基準線] 1/f^2 (Brown Noise)
        ax.plot(valid_freqs, slope_brown, color='red', linestyle=':', linewidth=1.2, 
                label='1/f² (Brown Noise/Random Walk)', alpha=0.8, zorder=2)# [基準線] 1/f (Pink Noise)
        
        ax.legend(loc='upper right', fontsize=10)
        
        # グラフの表示 (ペアごとに1枚ずつ)
        if auto_save:
            output_filename = f"Contact Spectrum (ID_{id1},{id2}).png"
            output_directory = os.path.dirname(position_csv_path)
            save_path = os.path.join(output_directory, output_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" - スペクトラム変化グラフを保存しました: {save_path}")
        else:
            print(f" - スペクトラム変化グラフを表示します: ID (ID:{id1} , ID:{id2})")
            plt.show()

print("全てのペアの処理が完了しました")



# メイン処理
if __name__ == "__main__":
    # 位置データの入力
    INPUT_CSV = "20251101_01"
    INPUT_POSITION_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position.csv"
    # 接触判定に使用するしきい値
    CONTACT_THRESHOLD = 50.0  # ピクセル単位の接触しきい値
    # グラフのサイズを指定
    FIG_SIZE = (10, 5)
    # グラフの自動保存設定
    AUTO_SAVE = True
    # 対数スケールの設定
    USE_LOGLOG_PLOT = True  # True: 両対数プロット, False: 半対数プロット

    # 関数を呼び出し
    plot_contact_spectrum(INPUT_POSITION_CSV, CONTACT_THRESHOLD, FIG_SIZE, AUTO_SAVE, USE_LOGLOG_PLOT)