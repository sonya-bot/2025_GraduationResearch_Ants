# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from matplotlib.ticker import MultipleLocator

# 1. 計算・前処理関数


def data_input(position_csv_path):
    """
    入力データの読み込み、ロング形式への変換。
    x0, y0 形式と x_0, y_0 形式の両方に対応。
    FPSはここで固定値として定義。
    """
    try:
        df_wide = pd.read_csv(position_csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {position_csv_path}")
        return None, None, None

    df_temp = df_wide.copy()
    
    # 'position'列があれば 'frame' にリネーム
    if 'position' in df_temp.columns:
        df_temp = df_temp.rename(columns={'position': 'frame'})
    
    # カラム名の正規化 (x0 -> x_0, y0 -> y_0)
    new_columns = {}
    for col in df_temp.columns:
        if col == 'frame':
            continue
        # x0, x1... の形式の場合
        if col.startswith('x') and col[1:].isdigit():
            new_columns[col] = f"x_{col[1:]}"
        elif col.startswith('y') and col[1:].isdigit():
            new_columns[col] = f"y_{col[1:]}"
    
    if new_columns:
        df_temp.rename(columns=new_columns, inplace=True)

    # IDの抽出 (x_0, y_0 ... )
    x_cols = [c for c in df_temp.columns if c.startswith('x_')]
    ids = [c.replace('x_', '') for c in x_cols]
    
    # ロング形式へ変換
    frames = []
    for uid in ids:
        col_x = f'x_{uid}'
        col_y = f'y_{uid}'
        
        if col_x in df_temp.columns and col_y in df_temp.columns:
            sub_df = df_temp[['frame', col_x, col_y]].copy()
            sub_df.columns = ['frame', 'position_x', 'position_y']
            sub_df['id'] = uid
            frames.append(sub_df)
    
    if not frames:
        print(f"警告: 有効な座標データ列が見つかりませんでした - {position_csv_path}")
        return None, None, None
        
    df_long = pd.concat(frames)
    df_long['id'] = pd.to_numeric(df_long['id'])
    
    fps = 2.0 # 固定FPS
    
    return df_long, ids, fps

def calculate_iid_internal(df_long, ids, contact_threshold):
    """
    個体間距離(IID)および接触状態を計算するロジック
    Pair(N=2)の場合は1通り、Trio(N=3)の場合は3通りのペアを計算
    """
    results = {}
    
    # IDリストを数値に変換してソート
    ids_int = sorted([int(id_str) for id_str in ids])
    
    # ペアの組み合わせ作成
    # N=2: (0,1)
    # N=3: (0,1), (1,2), (2,0)
    pairs = []
    if len(ids_int) == 2:
        pairs.append((ids_int[0], ids_int[1]))
    elif len(ids_int) == 3:
        pairs.append((ids_int[0], ids_int[1]))
        pairs.append((ids_int[1], ids_int[2]))
        pairs.append((ids_int[2], ids_int[0])) # ループ
    else:
        # N=1 または N>3 の場合は今回は対象外とするか、全組み合わせにする
        # ここでは単純な全組み合わせ (itertools.combinations) は使わず、
        # 指定された仕様(Trioはループ)に従う
        return {}

    # 計算
    for p1, p2 in pairs:
        pair_key = f"{p1}-{p2}"
        
        # 各個体のデータ抽出
        df1 = df_long[df_long['id'] == p1].sort_values('frame').set_index('frame')
        df2 = df_long[df_long['id'] == p2].sort_values('frame').set_index('frame')
        
        # フレーム合わせ (共通フレームのみ)
        common_index = df1.index.intersection(df2.index)
        if len(common_index) == 0:
            continue
            
        pos1 = df1.loc[common_index, ['position_x', 'position_y']].to_numpy()
        pos2 = df2.loc[common_index, ['position_x', 'position_y']].to_numpy()
        
        # ユークリッド距離
        dist = np.sqrt(np.sum((pos1 - pos2)**2, axis=1))
        
        # 接触状態 (0 or 1)
        contact = (dist < contact_threshold).astype(int)
        
        results[pair_key] = {
            'frame': common_index,
            'distance': dist,
            'contact': contact
        }
        
    return results

def collect_contact_statistics(target_dict, mode, base_path, contact_threshold, stats_list):
    """
    接触に関する統計データを計算してリストに追加
    """
    print(f" - IID統計データ収集中 ({mode})...")
    
    for name, folder_id in target_dict.items():
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        df_long, ids, fps = data_input(csv_path)

        if df_long is None or len(ids) < 2:
            continue

        iid_results = calculate_iid_internal(df_long, ids, contact_threshold)
        
        for pair_key, res in iid_results.items():
            dist = res['distance']
            contact = res['contact']
            
            # --- 統計量 ---
            mean_dist = np.mean(dist)
            min_dist = np.min(dist)
            
            # 接触回数 (0 -> 1 に変化した回数)
            # 先頭が1ならそれも1回とカウントするか、変化のみ見るか。ここでは変化回数+初期状態
            diff_contact = np.diff(contact, prepend=0)
            contact_count = np.sum(diff_contact == 1)
            
            # 接触率
            contact_ratio = np.mean(contact)
            
            # 平均接触時間 (分)
            # 総接触フレーム数 / 接触回数 / FPS / 60
            total_contact_frames = np.sum(contact)
            if contact_count > 0:
                mean_contact_duration = total_contact_frames / contact_count / fps / 60.0
            else:
                mean_contact_duration = 0
            
            stats_list.append({
                "Group_Type": mode,
                "Colony_Name": name,
                "Pair_ID": pair_key,
                "Mean_Distance": round(mean_dist, 2),
                "Min_Distance": round(min_dist, 2),
                "Contact_Count": contact_count,
                "Contact_Ratio": round(contact_ratio, 4),
                "Mean_Contact_Duration_Min": round(mean_contact_duration, 2)
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

def plot_distance_over_time(target_dict, mode, base_path, use_y_log, contact_threshold, single_fig_size, save_path, auto_save):
    """
    グラフA. 個体間距離の時系列
    """
    n_plots = len(target_dict)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, single_fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    title_suffix = f" ({mode})" if " - " not in mode else ""
    print(f"\n - 個体間距離時系列グラフ作成: {mode}")

    for i, (name, folder_id) in enumerate(target_dict.items()):
        if i >= len(axes): break
        ax = axes[i]
        
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        df_long, ids, fps = data_input(csv_path)

        if df_long is None or len(ids) < 2:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
            continue

        iid_results = calculate_iid_internal(df_long, ids, contact_threshold)
        
        # 時間軸 (分)
        # ※全ペアで共通のフレームインデックスと仮定、あるいはペアごとにプロット
        
        for pair_key, res in iid_results.items():
            frames = res['frame']
            dist = res['distance']
            time_min = frames / fps / 60.0
            
            ax.plot(time_min, dist, linewidth=0.8, alpha=0.4, label=f'Pair ID:{pair_key}')
            
        # 接触閾値ライン
        ax.axhline(y=contact_threshold, color='red', linestyle='--', linewidth=1.0, alpha=0.6, label=None)

        ax.set_title(name, fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if use_y_log:
            ax.set_yscale('log')
            ax.set_ylim(1, 1e3)
        else:
            ax.set_ylim(0, None)


        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Time [min]', fontsize=20)
        if i % n_cols == 0:
            ax.set_ylabel('Distance [pixel]', fontsize=20)

        # 凡例
        if len(ids) <= 10:
            ax.legend(loc='upper right', fontsize=15)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if n_plots > 1:
        plt.suptitle(f"Distance Over Time{title_suffix}", fontsize=20)
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

def plot_distance_distribution(target_dict, mode, base_path, contact_threshold, single_fig_size, save_path, auto_save):
    """
    グラフB. 個体間距離の分布 (ヒストグラム)
    """
    n_plots = len(target_dict)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, single_fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    title_suffix = f" ({mode})" if " - " not in mode else ""
    print(f"\n - 個体間距離分布グラフ作成: {mode}")

    for i, (name, folder_id) in enumerate(target_dict.items()):
        if i >= len(axes): break
        ax = axes[i]
        
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        df_long, ids, fps = data_input(csv_path)

        if df_long is None or len(ids) < 2:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
            continue

        iid_results = calculate_iid_internal(df_long, ids, contact_threshold)

        # ビン設定のために最大値を取得
        local_max_dist = 0
        for res in iid_results.values():
            local_max_dist = max(local_max_dist, np.max(res['distance']))
        
        bins = np.linspace(0, local_max_dist, 50)

        for pair_key, res in iid_results.items():
            dist = res['distance']
            weights = np.ones_like(dist) / len(dist)
            
            ax.hist(dist, bins=bins, weights=weights, histtype='step', 
                    linewidth=1.2, alpha=0.7, label=f'Pair ID:{pair_key}')

        # 接触閾値ライン
        ax.axvline(x=contact_threshold, color='red', linestyle='--', linewidth=1.0, alpha=0.6, label=None)

        ax.set_title(name, fontsize=20)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylim(0, 0.5)
        ax.set_xlim(0, 1000)
        
        
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Distance [pixel]', fontsize=20)
        if i % n_cols == 0:
            ax.set_ylabel('Probability', fontsize=20)

        if len(ids) <= 10:
            ax.legend(loc='upper right', fontsize=15)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if n_plots > 1:
        plt.suptitle(f"Distance Distribution{title_suffix}", fontsize=20)
        plt.tight_layout()
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

def plot_contact_state_over_time(target_dict, mode, base_path, contact_threshold, single_fig_size, save_path, auto_save):
    """
    グラフC. 接触状態の時系列 (0 or 1)
    """
    n_plots = len(target_dict)
    if n_plots == 0: return

    n_rows, n_cols, total_figsize = get_layout_params(n_plots, single_fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)
    
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    title_suffix = f" ({mode})" if " - " not in mode else ""
    print(f"\n - 接触状態グラフ作成: {mode}")

    for i, (name, folder_id) in enumerate(target_dict.items()):
        if i >= len(axes): break
        ax = axes[i]
        
        csv_path = os.path.join(base_path, folder_id, f"{folder_id}-position.csv")
        df_long, ids, fps = data_input(csv_path)

        if df_long is None or len(ids) < 2:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
            continue

        iid_results = calculate_iid_internal(df_long, ids, contact_threshold)

        for pair_key, res in iid_results.items():
            frames = res['frame']
            contact = res['contact']
            time_min = frames / fps / 60.0
            
            # オフセットなし、透明度調整のみ
            ax.plot(time_min, contact, linewidth=1.0, alpha=0.4, label=f'Pair:{pair_key}')

        ax.set_title(name, fontsize=20)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Noncontact\n(0)', 'Contact\n(1)'], fontsize=20)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.set_ylim(-0.1, 1.4)
        ax.set_xlim(85,185)
        ax.grid(True, linestyle='--', alpha=0.4)

        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Time [min]', fontsize=20)
        if i % n_cols == 0:
            ax.set_ylabel('Contact State', fontsize=20)

        if len(ids) <= 10:
            ax.legend(loc='upper right', fontsize=15)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if n_plots > 1:
        plt.suptitle(f"Contact State Over Time (latter half) {title_suffix}", fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 1]) 
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

    CONTACT_THRESHOLD = 50.0  # 接触判定閾値 (pixel)
    USE_Y_LOG = True
    FIG_SIZE = (6, 4)
    AUTO_SAVE = True

    # 3. 準備
    if AUTO_SAVE and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_contact_stats = []

    # ISOは個体間距離がないためスキップ
    DATA_SETS = [
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
        collect_contact_statistics(process_dict, mode_label, BASE_PATH, CONTACT_THRESHOLD, all_contact_stats)

        # --- B. グラフ描画 ---
        if is_specific_mode:
            # 個別ファイル出力
            for target_colony, folder_id in process_dict.items():
                single_dict = {target_colony: folder_id}
                print(f"\nProcessing Specific Target: {target_colony} ({mode_label})")
                
                # (1) IID Over Time
                save_name = f"Distance_Over_Time_{target_colony}_{file_suffix}.png"
                plot_distance_over_time(single_dict, f"{target_colony} - {mode_label}", 
                                   BASE_PATH, USE_Y_LOG,CONTACT_THRESHOLD, FIG_SIZE, 
                                   os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
                
                # (2) IID Distribution
                save_name = f"Distance_Distribution_{target_colony}_{file_suffix}.png"
                plot_distance_distribution(single_dict, f"{target_colony} - {mode_label}", 
                                      BASE_PATH, CONTACT_THRESHOLD, FIG_SIZE, 
                                      os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
                
                # (3) Contact State
                save_name = f"Contact_State_{target_colony}_{file_suffix}.png"
                plot_contact_state_over_time(single_dict, f"{target_colony} - {mode_label}", 
                                             BASE_PATH, CONTACT_THRESHOLD, FIG_SIZE, 
                                             os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
        else:
            # 一覧ファイル出力
            print(f"\nProcessing All Targets ({mode_label})")
            
            # Distance Over Time
            save_name = f"Distance_Over_Time_All_{file_suffix}.png"
            plot_distance_over_time(process_dict, mode_label, BASE_PATH, USE_Y_LOG, CONTACT_THRESHOLD, FIG_SIZE, 
                               os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
            
            # # Distance Distribution
            save_name = f"Distance_Distribution_All_{file_suffix}.png"
            plot_distance_distribution(process_dict, mode_label, BASE_PATH, CONTACT_THRESHOLD, FIG_SIZE, 
                                  os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)
            
            # Contact State
            # save_name = f"Contact_State_All_{file_suffix}.png"
            # plot_contact_state_over_time(process_dict, mode_label, BASE_PATH, CONTACT_THRESHOLD, FIG_SIZE, 
            #                              os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)

            save_name = f"Contact_State_All_{file_suffix},(latter half).png"
            plot_contact_state_over_time(process_dict, mode_label, BASE_PATH, CONTACT_THRESHOLD, FIG_SIZE, 
                                         os.path.join(OUTPUT_DIR, save_name), AUTO_SAVE)            

    # 5. 統計データのCSV保存
    if all_contact_stats and AUTO_SAVE:
        df_stats = pd.DataFrame(all_contact_stats)
        cols_order = [
            "Group_Type", "Colony_Name", "Pair_ID", 
            "Mean_Distance", "Min_Distance", "Contact_Count", "Contact_Ratio", "Mean_Contact_Duration_Min"
        ]
        # 存在する列のみ抽出
        final_cols = [c for c in cols_order if c in df_stats.columns]
        df_stats = df_stats[final_cols]
        
        csv_save_path = os.path.join(OUTPUT_DIR, "Contact_summary.csv")
        df_stats.to_csv(csv_save_path, index=False)
        print(f"\n★ 統計データを保存しました: {csv_save_path}")

    print("\nAll processes completed.")