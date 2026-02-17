# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import linregress

# 1. 計算・前処理関数
def data_input(position_csv_path):
    """
    入力データの読み込み、ロング形式への変換。
    FPSはここで固定値として定義。
    """
    try:
        df_wide = pd.read_csv(position_csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return None, None, None
    # 実装簡易化: 自力でmeltする方が確実
    df_temp = df_wide.copy()
    if 'position' in df_temp.columns:
        df_temp = df_temp.rename(columns={'position': 'frame'})
    
    # IDの抽出 (x_0, y_0 ... または 0_x, 0_y ...)
    # ここでは 'x_0', 'y_0' 形式を想定
    x_cols = [c for c in df_temp.columns if c.startswith('x')]
    ids = [c.replace('x', '') for c in x_cols]
    
    # ロング形式へ変換
    frames = []
    for uid in ids:
        sub_df = df_temp[['frame', f'x{uid}', f'y{uid}']].copy()
        sub_df.columns = ['frame', 'position_x', 'position_y']
        sub_df['id'] = uid
        frames.append(sub_df)
    
    if not frames:
        return None, None, None
        
    df_long = pd.concat(frames)
    df_long['id'] = pd.to_numeric(df_long['id'])
    
    fps = 2.0 # 固定FPS
    
    return df_long, ids, fps

def calculate_msd_internal(df_long, ids, fps, max_lag_ratio=0.5):
    """
    MSD計算ロジック
    """
    results = {}
    n_frames = df_long['frame'].max()
    max_lag = int(n_frames * max_lag_ratio)
    if max_lag == 0: max_lag = 1

    for uid in ids:
        # 個体ごとのデータ抽出と補間
        uid_num = int(uid)
        boid_df = df_long[df_long['id'] == uid_num].sort_values(by='frame').set_index('frame')
        
        # 欠損フレームの補間
        all_frames = pd.DataFrame(index=np.arange(boid_df.index.min(), boid_df.index.max() + 1))
        boid_df = boid_df.reindex(all_frames.index).interpolate(method='linear')
        coords = boid_df[['position_x', 'position_y']].to_numpy()
        
        msd_list = []
        lag_times_frames = list(range(1, max_lag))

        # MSD計算 (numpyで高速化)
        for lag in lag_times_frames:
            diff = coords[lag:] - coords[:-lag]
            squared_disp = np.sum(diff**2, axis=1)
            # 平均をとる
            if len(squared_disp) > 0:
                msd_list.append(np.mean(squared_disp))
            else:
                msd_list.append(np.nan)
        
        # NaN除去
        valid_indices = [i for i, val in enumerate(msd_list) if not np.isnan(val)]
        valid_lags = [lag_times_frames[i] for i in valid_indices]
        valid_msd = [msd_list[i] for i in valid_indices]
        
        if valid_lags:
            # 分単位のラグ時間に変換
            lags_min = np.array(valid_lags) / fps / 60.0
            results[uid] = {'lags_min': lags_min, 'msd': np.array(valid_msd)}

    return results

def collect_msd_statistics(target_dict, mode, base_path, stats_list):
    """
    MSDの傾き(Slope)などを計算してリストに追加
    """
    print(f" - MSD統計データ収集中 ({mode})...")
    
    for name, folder_id in target_dict.items():
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        df_long, ids, fps = data_input(csv_path)

        if df_long is None: continue

        msd_results = calculate_msd_internal(df_long, ids, fps)
        
        for uid in ids:
            res = msd_results.get(str(uid)) # IDはstrで管理
            if not res: 
                # 数値IDの場合もあるのでトライ
                res = msd_results.get(int(uid))
            
            if res is None: continue

            lags = res['lags_min']
            msd = res['msd']
            
            # ログログ回帰で傾き(α)を算出
            # ノイズを避けるため、ある程度データがある場合のみ
            if len(lags) > 10:
                slope, intercept, r_value, p_value, std_err = linregress(np.log10(lags), np.log10(msd))
            else:
                slope = np.nan
            
            stats_list.append({
                "Group_Type": mode,
                "Colony_Name": name,
                "ID": uid,
                "Slope_LogLog": round(slope, 4) if not np.isnan(slope) else None,
                # 簡易的な拡散係数 D = MSD / 4t (最後の点の値を使用)
                "Diff_Coeff_Last": round(msd[-1] / (4 * lags[-1]), 2) if len(lags) > 0 else None
            })

# 2. グラフ描画関数
def get_layout_params(n_plots, single_fig_size):
    """
    プロット数に応じて最適な行数・列数・figsizeを計算
    """
    if n_plots <= 1:
        return 1, 1, single_fig_size

    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    
    total_width = single_fig_size[0] * n_cols
    total_height = single_fig_size[1] * n_rows
    
    return n_rows, n_cols, (total_width, total_height)

def plot_msd_overview(target_dict, mode, base_path, single_fig_size, save_path, auto_save):
    """
    1コロニーにつき1枚のMSDグラフを作成 (全個体重ね合わせ)
    凡例にIDと傾きを表示し、右下に参考線の説明を表示する。
    """
    n_plots = len(target_dict)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, single_fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    title_suffix = f" ({mode})" if " - " not in mode else ""
    print(f"\n - MSDグラフ作成: {mode}")

    for i, (name, folder_id) in enumerate(target_dict.items()):
        if i >= len(axes): break
        ax = axes[i]
        
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        df_long, ids, fps = data_input(csv_path)

        if df_long is None:
            ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center')
            continue

        msd_results = calculate_msd_internal(df_long, ids, fps)
        
        has_data = False
        
        # --- 個体ごとのプロット ---
        for uid in ids:
            res = msd_results.get(str(uid)) or msd_results.get(int(uid))
            if not res: continue
            
            has_data = True
            lags = res['lags_min']
            msd = res['msd']
            
            # 傾き(Slope)計算
            slope_str = "?"
            if len(lags) > 10:
                try:
                    slope, _, _, _, _ = linregress(np.log10(lags), np.log10(msd))
                    slope_str = f"{slope:.2f}"
                except:
                    pass
            
            # 凡例用ラベル: ID:0 (α=1.05)
            label_txt = f"ID:{uid} {slope_str}"
            
            ax.loglog(lags, msd, '-', linewidth=1.5, alpha=0.7, label=label_txt)

        # --- 参考線とテキストボックス ---
        if has_data:
            # データの中心付近を取得して線を引く
            first_res = list(msd_results.values())[0]
            lags = first_res['lags_min']
            mid_idx = len(lags) // 2
            mid_x = lags[mid_idx]
            mid_y = first_res['msd'][mid_idx]
            
            # Slope 1 (Diffusive)
            y_s1 = mid_y * (lags / mid_x) ** 1
            ax.loglog(lags, y_s1, 'k:', linewidth=1.0, alpha=0.5) # 凡例には載せない
            
            # Slope 2 (Ballistic)
            y_s2 = mid_y * (lags / mid_x) ** 2
            ax.loglog(lags, y_s2, 'k-', linewidth=1.0, alpha=0.5) # 凡例には載せない
            
            # 右下に参考線の説明テキストボックスを配置
            ref_text = "··· 1(Brown)\n" \
                       "--- 2(Ballistic)"
            
            ax.text(0.60, 0.05, ref_text, transform=ax.transAxes, 
                    fontsize=15, horizontalalignment='left', verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))

        ax.set_title(name, fontsize=20)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.set_ylim(1e-1, 1e6) 

        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Lag Time [min]', fontsize=20)
        if i % n_cols == 0:
            ax.set_ylabel('MSD [pixel²]', fontsize=20)

        # メイン凡例 (IDと傾き) - 左上
        if len(ids) <= 10:
            ax.legend(loc='upper left', title="Slope", fontsize=15, title_fontsize=15)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if n_plots > 1:
        plt.suptitle(f"Mean Squared Displacement{title_suffix}", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()
        
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {save_path}")
        plt.close()
    else:
        print(f"グラフを表示します: {save_path}")
        plt.show()
def plot_msd_slope_boxplot(stats_list, output_dir, auto_save):
    """
    【修正版】MSDの傾き(Slope)を比較する箱ひげ図
    - Kruskal-Wallis検定 (全体) と Mann-Whitney U検定 (ペア) を実施
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import mannwhitneyu, kruskal

    print(f"\n - MSD傾き比較箱ひげ図を作成中...")

    # 1. データの抽出・整理
    groups = ["Isolate", "Pair", "Trio"]
    plot_data = {g: [] for g in groups}
    
    for rec in stats_list:
        gtype = rec.get("Group_Type")
        slope = rec.get("Slope_LogLog")
        
        if gtype == "Trioed": gtype = "Trio"
        
        if gtype in groups and slope is not None and not np.isnan(slope):
            plot_data[gtype].append(slope)

    # データチェック
    if not all(len(plot_data[g]) > 0 for g in groups):
        print("データが不足しているため、3群比較ができません。")
        return

    # 2. 統計検定
    
    # A. 全体検定 (Kruskal-Wallis)
    # 3群すべてにデータがある場合のみ実施
    kw_stat, kw_p = kruskal(plot_data["Isolate"], plot_data["Pair"], plot_data["Trio"])
    
    # B. ペア検定 (Mann-Whitney U) - パターン2: 全組み合わせ
    p_values = {}
    pairs = [("Isolate", "Pair"), ("Pair", "Trio"), ("Isolate", "Trio")]
    
    for g1, g2 in pairs:
        data1 = plot_data[g1]
        data2 = plot_data[g2]
        stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
        p_values[f"{g1} vs {g2}"] = p

    # 3. グラフ描画
    fig, ax = plt.subplots(figsize=(6, 6))

    data_list = [plot_data[g] for g in groups]
    
    # 箱ひげ図
    box = ax.boxplot(data_list, tick_labels=groups, patch_artist=True,
                     widths=0.6, showfliers=False,
                     medianprops=dict(color='black', linewidth=1.5))

    colors = ['lightblue', 'orange', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 散布図 (Jitter)
    for i, g in enumerate(groups):
        y = plot_data[g]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.6, s=40, color='gray', edgecolors='black', linewidth=0.5)

    # 4. 検定結果の表示作成
    # 星マーク変換用関数
    def get_sig(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        return "n.s."

    text_lines = []
    
    # 全体 (Kruskal-Wallis)
    text_lines.append(f"Overall (Kruskal-Wallis): p={kw_p:.3f} ({get_sig(kw_p)})")
    text_lines.append("-" * 30) # 区切り線
    
    # ペアごとの結果
    for g1, g2 in pairs:
        p = p_values[f"{g1} vs {g2}"]
        text_lines.append(f"{g1} vs {g2}: p={p:.3f} ({get_sig(p)})")
    
    full_text = "\n".join(text_lines)
    
    # グラフタイトルと結果テキストの配置
    ax.set_title(f"Comparison of MSD Slope (α)\n(P={kw_p:.3f})", fontsize=20) # padを広げる


    # 軸設定
    ax.set_ylabel("MSD Slope (α)", fontsize=20)
    ax.set_xlabel("Group Size", fontsize=20)
    ax.set_ylim(0.0, 0.6)
    
    ax.set_yticks(np.arange(0, 0.7, 0.1))
    ax.set_yticklabels([f"{x:.1f}" for x in np.arange(0, 0.7, 0.1)], fontsize=15)
    ax.set_xticklabels(groups, fontsize=15)
    
    ax.axhline(y=1.0, color='gray', linewidth=1.0, alpha=0.5, label='α≈1 (Diffusive)')
    ax.axhline(y=2.0, color='gray', linewidth=1.0, alpha=0.5, label='α≈2 (Ballistic)')
    ax.axhline(y=0.5, color='gray', linewidth=1.0, alpha=0.5, label='1<α<2 (Subdiffusive)')
    ax.legend(loc='upper left', fontsize=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if auto_save:
        save_path = os.path.join(output_dir, "MSD_Slope_Comparison_Boxplot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"箱ひげ図を保存しました: {save_path}")
        plt.close()
    else:
        plt.show()

# メイン処理

if __name__ == "__main__":
    
    # 1. データ定義
    ISO_DICT = {
        "Colony A": "20251030_01", "Colony B": "20251104_02", "Colony C": "20251106_01",
        "Colony D": "20251113_01", "Colony E": "20251117_02", "Colony G": "20251119_01",
        "Colony H": "20251120_02", "Colony I": "20251127_01",
    }
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

    # 2. ユーザー設定
    # TARGET_SPECIFIC = ["Colony A"] # 個別指定
    TARGET_SPECIFIC = []           # 全実行

    BASE_PATH = "/Users/sonya/卒論データ/analysis_data"
    OUTPUT_DIR = "/Users/sonya/卒論データ/analysis_data/graph_output"

    FIG_SIZE = (6, 4)
    AUTO_SAVE = True

    # 3. 準備
    # if AUTO_SAVE and not os.path.exists(OUTPUT_DIR):
    #     os.makedirs(OUTPUT_DIR)

    all_msd_stats = []

    DATA_SETS = [
        (ISO_DICT, "Isolate", "Isolate"),
        (PAIR_DICT, "Pair", "Pair"),
        (TRIO_DICT, "Trio", "Trio")
    ]

    # 4. 実行ループ
    for source_dict, mode_label, file_suffix in DATA_SETS:
        
        # 処理対象の辞書を作成
        if TARGET_SPECIFIC:
            process_dict = {k: v for k, v in source_dict.items() if k in TARGET_SPECIFIC}
            if not process_dict: continue
            is_specific_mode = True
        else:
            process_dict = source_dict
            is_specific_mode = False

        # --- A. 統計データの収集 ---
        collect_msd_statistics(process_dict, mode_label, BASE_PATH, all_msd_stats)

        # --- B. グラフ描画 ---
        if is_specific_mode:
            # 個別ファイル出力
            for target_colony, folder_id in process_dict.items():
                single_dict = {target_colony: folder_id}
                save_name = f"MSD_{target_colony}_{file_suffix}.png"
                
                plot_msd_overview(single_dict, f"{target_colony} - {mode_label}", 
                                  BASE_PATH, FIG_SIZE, 
                                  os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
        else:
            # 一覧ファイル出力
            save_name = f"MSD_All_{file_suffix}.png"
            # plot_msd_overview(process_dict, mode_label, BASE_PATH, FIG_SIZE, 
            #                   os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
            
            plot_msd_slope_boxplot(all_msd_stats, OUTPUT_DIR, AUTO_SAVE)

    # 5. 統計データのCSV保存
    if all_msd_stats and AUTO_SAVE:
        df_stats = pd.DataFrame(all_msd_stats)
        cols_order = ["Group_Type", "Colony_Name", "ID", "Slope_LogLog", "Diff_Coeff_Last"]
        # 存在する列のみ
        final_cols = [c for c in cols_order if c in df_stats.columns]
        df_stats = df_stats[final_cols]
        
        csv_save_path = os.path.join(OUTPUT_DIR, "msd_summary.csv")
        df_stats.to_csv(csv_save_path, index=False)
        print(f"\n★ MSD統計データを保存しました: {csv_save_path}")

    print("\nAll processes completed.")