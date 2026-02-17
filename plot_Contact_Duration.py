# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import matplotlib.ticker as ticker
from itertools import combinations 
from matplotlib.ticker import LogLocator, ScalarFormatter


# 1. データ計算・解析関数

def calculate_durations_for_colony(csv_path, contact_threshold):
    """
    1つのコロニーのデータから、グループサイズごとの持続時間リストを計算する。
    
    Returns:
        duration_storage (dict): {size: [duration_list]}
        n_individuals (int): 個体数
    """
    try:
        df_pos = pd.read_csv(csv_path)
        print(f"データ読み込み: '{os.path.basename(csv_path)}'")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {csv_path}")
        return None, 0

    # 個体IDの特定
    x_cols = [col for col in df_pos.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]
    n_individuals = len(individual_ids)

    if n_individuals <= 1:
        print(f"スキップ: 個体数が少なすぎます ({n_individuals})")
        return None, n_individuals

    # データをメモリに展開
    coords_dict = {}
    for uid in individual_ids:
        coords_dict[uid] = df_pos[[f'x{uid}', f'y{uid}']].to_numpy()

    total_frames = len(df_pos)
    FPS = 2.0 

    # 持続時間格納用
    duration_storage = {size: [] for size in range(1, n_individuals + 1)}
    
    # 状態追跡用: {uid: {'current_size': 0, 'frame_count': 0}}
    tracker = {uid: {'current_size': 0, 'frame_count': 0} for uid in individual_ids}

    # --- フレームごとの解析 ---
    # (NetworkXを使用した連結成分分解)
    for i in range(total_frames):
        G = nx.Graph()
        G.add_nodes_from(individual_ids)
        
        # 距離判定によるエッジ追加
        # (高速化のため、本来はscipy.spatial.distance.pdist推奨だが、既存ロジックを維持)
        for j in range(n_individuals):
            for k in range(j + 1, n_individuals):
                id_a = individual_ids[j]
                id_b = individual_ids[k]
                pos_a = coords_dict[id_a][i]
                pos_b = coords_dict[id_b][i]
                dist = np.sqrt(np.sum((pos_a - pos_b)**2))
                
                if dist <= contact_threshold:
                    G.add_edge(id_a, id_b)

        # 連結成分（グループ）の特定
        connected_components = list(nx.connected_components(G))
        current_frame_sizes = {}
        
        for component in connected_components:
            size = len(component)
            for uid in component:
                current_frame_sizes[uid] = size

        # 状態更新
        for uid in individual_ids:
            new_size = current_frame_sizes[uid]
            prev_size = tracker[uid]['current_size']
            count = tracker[uid]['frame_count']

            if new_size == prev_size:
                tracker[uid]['frame_count'] += 1
            else:
                # 状態が変わった場合、前の状態の持続時間を記録
                if prev_size > 0:
                    duration_sec = count / FPS
                    duration_storage[prev_size].append(duration_sec)
                
                tracker[uid]['current_size'] = new_size
                tracker[uid]['frame_count'] = 1

    # 最後のフレームの処理
    for uid in individual_ids:
        size = tracker[uid]['current_size']
        count = tracker[uid]['frame_count']
        if size > 0:
            duration_storage[size].append(count / FPS)
            
    return duration_storage, n_individuals


# 2. グラフ描画関数 (複数コロニー対応)

def plot_combined_durations(all_results, use_x_log, use_y_log, output_dir, fig_size, auto_save):
    """
    集約された全コロニーのデータをグループサイズごとにCCDF（生存関数）としてプロットする。
    
    Args:
        graph_type (str): 'loglog' (両対数), 'semilog_y' (片対数), 'linear' (線形)
    """
    # if not all_results:
    #     print("表示するデータがありません。")
    #     return

    # # 全データの中で最大の個体数（=最大のグループサイズ）を特定
    # max_n = max(res['n_inds'] for res in all_results)
    
    # # カラーマップの生成
    # colony_names = [res['name'] for res in all_results]
    # colors = plt.cm.jet(np.linspace(0, 0.9, len(colony_names))) # 0.9で止めて黄色すぎないように
    # color_map = {name: color for name, color in zip(colony_names, colors)}

    # # グループサイズごとにグラフを作成 (Size 1, Size 2, Size 3...)
    # for size in range(1, max_n + 1):
        
    #     # このサイズにおける有効なデータが1つでもあるか確認
    #     has_data = False
    #     for res in all_results:
    #         if size in res['durations'] and len(res['durations'][size]) > 0:
    #             has_data = True
    #             break
        
    #     if not has_data:
    #         continue

    #     # --- グラフ描画開始 ---
    #     plt.figure(figsize=fig_size)
    #     ax = plt.gca()

    #     label_base = ""
    #     if size == 1: label_base = "Isolate"
    #     elif size == 2: label_base = "Pair"
    #     elif size == 3: label_base = "Trio"
    #     else: label_base = f"Group Size {size}"


    #     label_base += f" (N={max_n})"

    #     print(f" - Plotting {label_base}")

    #     # コロニーごとにループ
    #     for res in all_results:
    #         colony_name = res['name']
    #         durations_dict = res['durations']
            
    #         # データがない場合はスキップ
    #         if size not in durations_dict:
    #             continue
                
    #         durs = durations_dict[size]
    #         if len(durs) == 0:
    #             continue

    #         # --- データの処理 (累積和と確率) ---
    #         # 1. ソート (昇順)
    #         durs.sort()
    #         x = np.array(durs)
            
    #         # 2. 累積確率 (P >= x)
    #         # 大きい方から何番目か / 全体数
    #         # np.arange(len, 0, -1) -> [N, N-1, ..., 1]
    #         n_events = len(durs)
    #         y = np.arange(n_events, 0, -1) / n_events
            
    #         color = color_map[colony_name]

    #         # --- プロット ---
    #         # 1. 点 (Marker): 凡例には含めない (label=None)
    #         ax.plot(x, y,
    #                 marker='o', markersize=4, markeredgewidth=0,
    #                 linestyle='',
    #                 label=None, 
    #                 color=color, alpha=0.6)
            
    #         # 2. 線 (Line): 凡例に名前を表示 (label=colony_name)
    #         ax.plot(x, y, 
    #                 linestyle='-', linewidth=1.5,
    #                 label=colony_name, 
    #                 color=color, alpha=0.4)

    #         print(f"  - {colony_name}: Count={n_events}, MaxDuration={max(durs):.1f}s")

    #     # --- グラフ体裁の調整 ---
    #     ax.set_title(f'Contact Duration CCDF:{label_base}', fontsize=20)
    #     ax.set_ylabel('Probability', fontsize=20)
    #     ax.set_xlabel('Duration time [sec]', fontsize=20)
        
    #     if use_y_log:
    #         ax.set_yscale("log")
    #         ax.set_ylim(0, 1.05)
    #     else:
    #         ax.set_ylim(0, 1.05)

    #     # X軸の範囲（適宜調整）
    #     if use_x_log:
    #         ax.set_xscale("log")
    #         ax.set_xlim(right=1e4)
    #     else:
    #         ax.set_xlim(left=0)
    #         ax.xaxis.set_major_locator(ticker.MultipleLocator(1000)) # メモリ幅の指定
    #         ax.grid(True, linestyle='--', alpha=0.4) 
    #         ax.set_xlim(0,5000)
 
        
    #     # 凡例を外に出す
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title="Colony",fontsize=12)
    #     plt.tight_layout()

    #     # 保存または表示
    #     if auto_save:
    #         # ベースフォルダに保存
    #         filename = f"Contact_Duration_CCDF:{label_base}.png"
    #         save_path = os.path.join(output_dir, filename)
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #         print(f"グラフを保存しました: {save_path}")
    #         plt.close()
    #     else:
    #         plt.show()

    if not all_results:
        print("表示するデータがありません。")
        return

    max_n = max(res['n_inds'] for res in all_results)
    colony_names = [res['name'] for res in all_results]
    colors = plt.cm.jet(np.linspace(0, 0.9, len(colony_names)))
    color_map = {name: color for name, color in zip(colony_names, colors)}

    for size in range(1, max_n + 1):
        has_data = False
        for res in all_results:
            if size in res['durations'] and len(res['durations'][size]) > 0:
                has_data = True
                break
        
        if not has_data:
            continue

        # --- グラフ描画開始 ---
        plt.figure(figsize=fig_size)
        ax = plt.gca()

        label_base = ""
        if size == 1: label_base = "Isolate (N=1)"
        elif size == 2: label_base = "Pair (N=2)"
        elif size == 3: label_base = "Trio (N=3)"
        else: label_base = f"Group Size {size}"

        print(f" - Plotting CCDF for {label_base}")

        min_prob_percent = 100.0 # Y軸の下限調整用

        for res in all_results:
            colony_name = res['name']
            durations_dict = res['durations']
            
            if size not in durations_dict:
                continue
                
            durs = durations_dict[size]
            if len(durs) == 0:
                continue

            # --- データの処理 (CCDF計算・パーセント化) ---
            durs.sort()
            x = np.array(durs)
            n_events = len(durs)
            
            # Y軸: 1.0 -> 100.0 に変換
            y = (np.arange(n_events, 0, -1) / n_events) * 100.0
            
            # 最小値を記録（軸範囲の調整用）
            if len(y) > 0:
                min_prob_percent = min(min_prob_percent, y[-1])
            
            color = color_map[colony_name]

            # --- プロット ---
            ax.plot(x, y, 
                    marker='o', markersize=3, markeredgewidth=0,
                    linestyle='-', linewidth=1.5,
                    label=colony_name, 
                    color=color, alpha=0.7)

            print(f"  - {colony_name}: Count={n_events}, MaxDuration={max(durs):.1f}s")

        # --- グラフ体裁の調整 ---
        ax.set_title(f'Contact Duration CCDF: {label_base}', fontsize=15)
        
        # Y軸ラベル: パーセント表記に修正
        ax.set_ylabel('$1 - P(X<τ)×100$ [%]', fontsize=15)
        ax.set_xlabel('Duration Time $τ$ [sec]', fontsize=15)
        
        # 軸スケールとフォーマット設定
        if use_y_log and use_x_log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Y軸: 1, 10, 100 のようなスカラ表記にする
            ax.set_yticks([1, 10, 100])
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            
            # 範囲設定
            ax.set_ylim(1, 110)
            ax.set_xlim(0, 1e4)

        else:
            # 線形スケールの場合
            ax.set_ylim(0, 105)
            ax.grid(True, linestyle='--', alpha=0.4)

        ax.grid(True, which="major", linestyle='--', alpha=0.6)
        
        # 凡例
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, title="Colony",fontsize=12)
        plt.tight_layout()

        if auto_save:
            filename = f"Contact_Duration_CCDF_{label_base}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"グラフを保存しました: {save_path}")
            plt.close()
        else:
            plt.show()

# ==========================================
# 3. メイン実行ブロック
# ==========================================

if __name__ == '__main__':
    # ◆◆◆ 設定箇所 ◆◆◆
    
    # 入力ファイル

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

    TARGET_COLONIES_DICT = TRIO_DICT  # 解析対象のコロニー辞書を指定

    # データフォルダのルートパス (環境に合わせて変更)
    # Mac
    BASE_PATH = "/Users/sonya/卒論データ/analysis_data"
    # Windows
    # BASE_PATH = "d:/analysis_data"
    OUTPUT_DIR = "/Users/sonya/卒論データ/analysis_data/graph_output"

    # 解析パラメータ
    CONTACT_THRESHOLD = 50.0  # ピクセル
    USE_X_LOG = True       # X軸を対数表示するかどうか
    USE_Y_LOG = True      # Y軸を対数表示するかどうか
    FIG_SIZE = (8, 4)         # グラフサイズ
    AUTO_SAVE = True       # 画像保存するかどうか
    
    # ----------------------------------------------------
    
    print("複数コロニー接触持続時間解析を開始します")
    
    # 全データを格納するリスト
    all_colony_results = []
    
    # 1. 各コロニーのデータを計算して収集
    for display_name, folder_id in TARGET_COLONIES_DICT.items():
        # CSVパスの構築
        csv_path = os.path.join(BASE_PATH, folder_id, f"{folder_id}-position.csv")
        
        print(f"\n - Processing: {display_name}")
        durations, n_inds = calculate_durations_for_colony(csv_path, CONTACT_THRESHOLD)
        
        if durations:
            all_colony_results.append({
                'name': display_name,
                'durations': durations,
                'n_inds': n_inds
            })

    # 2. まとめてグラフ描画
    if all_colony_results:
        print("\n -  グラフ描画開始")
        # 保存先として、最初のコロニーの親フォルダ（analysis_data直下など）を指定、
        # あるいはスクリプト実行場所などを指定
        output_dir = "/Users/sonya/卒論データ/analysis_data/graph_output"
        
        plot_combined_durations(all_colony_results, USE_X_LOG, USE_Y_LOG, output_dir, FIG_SIZE, AUTO_SAVE)
        print("\n全ての処理が完了しました。")
    else:
        print("\n有効なデータがありませんでした。")