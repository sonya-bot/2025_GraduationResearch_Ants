# # -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
from itertools import combinations
import matplotlib.ticker as ticker


# 1.計算・解析
def calculate_power_spectrum(contact_signal, sample_spacing_minutes):
    """ 
    二値シグナル(接触/非接触)からパワースペクトル(dB)と傾きを計算 
    """
    # --- 1. トレンド除去 (Linear Detrend) ---
    # 1次関数の傾向（ドリフト）を除去
    detrended_signal = signal.detrend(contact_signal, type='linear')
    
    # --- 2. フーリエ変換 ---
    N = len(detrended_signal)
    if N == 0:
        return None, None, None, None, None, None

    # 実フーリエ変換
    F = np.fft.rfft(detrended_signal) * (2/N)
    freq_per_min = fft.rfftfreq(N, d=sample_spacing_minutes)

    # 振幅スペクトル（絶対値） & 対数変換
    F_abs = np.abs(F)
    F_log = np.log10(F_abs + 1e-10)

    # --- 3. 解析用データの抽出 (0Hzを除く) ---
    valid_idx = freq_per_min > 0
    valid_freqs = freq_per_min[valid_idx]
    valid_F_log = F_log[valid_idx]
    
    if len(valid_freqs) < 2:
        return None, None, None, None, None, None

    # --- 4. 傾きの計算 (Regression Slope) ---
    log_freqs = np.log10(valid_freqs)
    slope, intercept = np.polyfit(log_freqs, valid_F_log, 1)
    
    # --- 5. 基準線 (1/f, 1/f^2) の計算 ---
    ref_power = F_log[1] if len(F_log) > 1 else 0
    ref_freq = valid_freqs[0]

    slope_pink = -1.0 * (log_freqs - np.log10(ref_freq)) + ref_power
    slope_brown = -2.0 * (log_freqs - np.log10(ref_freq)) + ref_power

    return freq_per_min, F_log, valid_freqs, slope_pink, slope_brown, slope

# 2.個体ごとの処理
def process_contact_spectrum(csv_path, contact_threshold, mode='pair'):
    """
    1つのコロニーのCSVを読み込み、スペクトルデータと傾きリストを返す
    mode: 'pair' (N=2) or 'trio' (N=3)
    """
    try:
        df_pos = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {csv_path}")
        return None

    # FPS設定
    FPS = 2.0
    sample_spacing_minutes = (1.0 / FPS) / 60

    # 個体ID特定
    x_cols = [col for col in df_pos.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]
    n_individuals = len(individual_ids)

    results = [] # {'label': str, 'freq': [], 'amp': [], 'slope': float, 'valid_freqs': [], 'pink': [], 'brown': []}

    # 座標データの展開
    coords = {}
    for uid in individual_ids:
        coords[uid] = df_pos[[f'x{uid}', f'y{uid}']].to_numpy()

    # --- ペア解析 (N=2) ---
    if mode == 'pair':
        if n_individuals < 2:
            return None
            
        pair_combinations = list(combinations(individual_ids, 2))
        for id1, id2 in pair_combinations:
            # 距離計算
            distances = np.sqrt(np.sum((coords[id1] - coords[id2])**2, axis=1))
            contact_signal = (distances <= contact_threshold).astype(float)
            
            # スペクトル計算
            freq, amp, v_freq, pink, brown, slope = calculate_power_spectrum(contact_signal, sample_spacing_minutes)
            
            if slope is not None:
                results.append({
                    'label': f"ID:{id1},{id2}",
                    'freq': freq,
                    'amp': amp,
                    'slope': slope,
                    'valid_freqs': v_freq,
                    'pink': pink,
                    'brown': brown
                })

    # --- トリオ解析 (N=3) ---
    elif mode == 'trio':
        if n_individuals < 3:
            return None
            
        trio_combinations = list(combinations(individual_ids, 3))
        for id1, id2, id3 in trio_combinations:
            # 3ペアの距離計算
            d_AB = np.sqrt(np.sum((coords[id1] - coords[id2])**2, axis=1))
            d_BC = np.sqrt(np.sum((coords[id2] - coords[id3])**2, axis=1))
            d_CA = np.sqrt(np.sum((coords[id3] - coords[id1])**2, axis=1))
            
            c_AB = (d_AB <= contact_threshold)
            c_BC = (d_BC <= contact_threshold)
            c_CA = (d_CA <= contact_threshold)
            
            # トリオ全体の接触シグナル (Chain or Triangle)
            triangle = (c_AB & c_BC & c_CA)
            chain = ((c_AB & c_BC) | (c_BC & c_CA) | (c_CA & c_AB)) & (~triangle)
            trio_signal = (triangle | chain).astype(float)
            
            # スペクトル計算
            freq, amp, v_freq, pink, brown, slope = calculate_power_spectrum(trio_signal, sample_spacing_minutes)
            
            if slope is not None:
                results.append({
                    'label': f"ID:{id1},{id2},{id3}",
                    'freq': freq,
                    'amp': amp,
                    'slope': slope,
                    'valid_freqs': v_freq,
                    'pink': pink,
                    'brown': brown
                })

    return results

# 3.グラフ描画共通関数
def plot_spectrum_on_ax(ax, results, col_name, use_x_log, show_xlabel=True, show_ylabel=True):
    """
    1つのAxes(グラフエリア)に対してスペクトルを描画する共通関数
    """
    slope_texts = []
    
    # データ描画
    for idx, res in enumerate(results):
        ax.plot(res['freq'], res['amp'], linewidth=1.2, alpha=0.7)
        
        # 基準線 (最初の1回だけ)
        if idx == 0:
            ax.plot(res['valid_freqs'], res['pink'], color='gray', linestyle='--', label="1/f(Pink)", linewidth=1.5, alpha=0.5, zorder=1)
            ax.plot(res['valid_freqs'], res['brown'], color='gray', linestyle=':', label="1/f²(Brown)", linewidth=1.5, alpha=0.5, zorder=1)
        
        slope_texts.append(f"Slope: {res['slope']:.2f}")

    # デザイン調整
    ax.set_title(col_name, fontsize=20)
    ax.set_ylim(-5.5, 0)
    ax.grid(True, linestyle='--', alpha=0.4)
    if len(results) > 0:
        ax.legend(loc='upper right', fontsize=15)

    # 軸ラベル
    if show_ylabel:
        ax.set_ylabel('Amplitude', fontsize=20)
    if show_xlabel:
        ax.set_xlabel('Frequency [/min]', fontsize=20)

    # 軸スケール
    if use_x_log:
        ax.set_xscale('log')
        ax.set_xlim(1e-2, 1e2)
    else:
        ax.set_xlim(0, 100)

    # 傾きテキスト表示
    if slope_texts:
        full_text = "\n".join(slope_texts)
        ax.text(0.02, 0.05, full_text, transform=ax.transAxes, 
                fontsize=15, verticalalignment='bottom', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray'))
        
# 4.ダッシュボード描画
def draw_graph(target_dict, mode, base_path, contact_threshold, use_x_log, single_fig_size, save_path, auto_save):
    colony_names = list(target_dict.keys())
    n_plots = len(colony_names)
    n_cols = 4
    n_rows = 2
    
    # 全体サイズの計算
    total_width = single_fig_size[0] * n_cols
    total_height = single_fig_size[1] * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, total_height))
    axes = axes.flatten()

    print(f"\n - 一覧グラフ作成 ({mode})")
    for i in range(n_cols * n_rows):
        ax = axes[i]
        if i < n_plots:
            col_name = colony_names[i]
            folder_id = target_dict[col_name]
            csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
            print(f" - Processing {col_name}")
            
            results = process_contact_spectrum(csv_path, contact_threshold, mode)
            if results:
                # 軸ラベルの制御 (端っこだけ表示する)
                show_y = (i % n_cols == 0)
                show_x = (i >= n_cols)
                plot_spectrum_on_ax(ax, results, col_name, use_x_log, show_x, show_y)
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_title(col_name)
        else:
            ax.axis('off')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.suptitle(f"Contact Spectrum({mode.capitalize()})", fontsize=20, y=0.98)
    plt.tight_layout()

    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"保存完了: {save_path}")
        plt.close()
    else:
        plt.show()

# 5.特定のコロニーのみのグラフ作成
def draw_specific_colonies(target_dict, target_names, mode, base_path, contact_threshold, use_x_log, fig_size, output_dir, auto_save):
    """
    指定されたコロニー名(リスト)のデータのみを個別に描画し、1枚ずつ保存する
    """
    print(f"\n - 個別グラフ作成 ({mode})")
    
    for name in target_names:
        if name not in target_dict:
            print(f"スキップ: '{name}' は辞書に含まれていません")
            continue
            
        folder_id = target_dict[name]
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        print(f" - Processing {name} (Single Graph)")
        
        results = process_contact_spectrum(csv_path, contact_threshold, mode)
        
        if results:
            # 1枚のグラフを作成
            fig, ax = plt.subplots(figsize=fig_size)
            
            # 共通描画関数を使用 (ラベルは常に表示)
            plot_spectrum_on_ax(ax, results, name, use_x_log, show_xlabel=True, show_ylabel=True)
            
            plt.title(f"Contact Spectrum ({mode.capitalize()}) : {name}", fontsize=20, y=1.03)
            plt.tight_layout()
            
            # 個別保存
            if auto_save:
                safe_name = name.replace(" ", "_")
                filename = f"Contact_Spectrum:{safe_name}({mode}).png"
                save_path = os.path.join(output_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"グラフを保存しました: {filename}")
                plt.close()
            else:
                print(f"グラフを表示します: {name}")
                plt.show()
        else:
            print(f" - データがありません: {name}")

def plot_slope_boxplot(slope_data, mode, save_path, auto_save):
    """
    傾きデータのボックスプロットを作成
    """
    plt.figure(figsize=(6, 4))
    plt.boxplot(slope_data, labels=[mode.capitalize()])
    plt.ylabel('Slope', fontsize=15)
    plt.title(f'Slope Distribution ({mode.capitalize()})', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.4)

    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"傾きボックスプロットを保存しました: {save_path}")
        plt.close()
    else:
        plt.show()

# メイン処理
if __name__ == '__main__':
    # --- 設定 ---
    # 入力ファイル辞書
    PAIR_DICT = {
        "Colony A": "20251030_02", "Colony B": "20251105_01", "Colony C": "20251107_01",
        "Colony D": "20251113_03", "Colony E": "20251118_01", "Colony G": "20251119_02",
        "Colony H": "20251121_01", "Colony I": "20251127_02",
    }
    TRIO_DICT = {
        "Colony A": "20251101_01", "Colony B": "20251105_02", "Colony C": "20251110_01",
        "Colony D": "20251117_01", "Colony E": "20251118_02", "Colony G": "20251120_01",
        "Colony H": "20251126_01", "Colony I": "20251128_01",
    }

    # ベースパス (環境に合わせて変更してください)
    # BASE_PATH = "/Volumes/100.108.13.8/analysis_data"
    BASE_PATH = "/Users/sonya/卒論データ/analysis_data"
    
    CONTACT_THRESHOLD = 50.0 # ピクセル単位の接触しきい値
    USE_X_LOG = True
    FIG_SIZE = (6, 4)         # 個別のグラフサイズ
    AUTO_SAVE = False   # True: ファイル保存, False: 画面表示
    
    # 出力先
    OUTPUT_DIR = "/Users/sonya/卒論データ/analysis_data/graph_output" # または任意のフォルダ

    # --- 実行 ---
    # 1. ペア (N=2) のダッシュボード作成
    save_path_pair = os.path.join(OUTPUT_DIR, "Contact_Spectrum(Pair).png")
    draw_graph(PAIR_DICT, 'pair', BASE_PATH, CONTACT_THRESHOLD, USE_X_LOG, FIG_SIZE, save_path_pair, AUTO_SAVE)
    
    # 2. トリオ (N=3) のダッシュボード作成
    save_path_trio = os.path.join(OUTPUT_DIR, "Contact_Spectrum(Trio).png")
    draw_graph(TRIO_DICT, 'trio', BASE_PATH, CONTACT_THRESHOLD, USE_X_LOG, FIG_SIZE, save_path_trio, AUTO_SAVE)

    # 個体数別傾きボックスプロット作成
    plot_slope_boxplot(
        [res['slope'] for col in PAIR_DICT.keys() 
         for res in process_contact_spectrum(
             os.path.join(BASE_PATH, PAIR_DICT[col], f"{PAIR_DICT[col]}-position.csv"), 
             CONTACT_THRESHOLD, 'pair') or []],
        'pair',
        os.path.join(OUTPUT_DIR, "Slope_Boxplot(Pair).png"),
        AUTO_SAVE
    )
    
    TARGET_SPECIFIC = []

    # 実行 (個別ファイルとして保存されます)
    draw_specific_colonies(PAIR_DICT, TARGET_SPECIFIC, 'pair', BASE_PATH, CONTACT_THRESHOLD, USE_X_LOG, FIG_SIZE, OUTPUT_DIR, AUTO_SAVE)
    draw_specific_colonies(TRIO_DICT, TARGET_SPECIFIC, 'trio', BASE_PATH, CONTACT_THRESHOLD, USE_X_LOG, FIG_SIZE, OUTPUT_DIR, AUTO_SAVE)

    print("\n全ての処理が完了しました。")
