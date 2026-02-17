# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import networkx as nx
from itertools import combinations

# ==========================================
# 1. データ読み込み・前処理
# ==========================================

def load_data(folder_path, folder_id):
    """
    位置データを読み込み、データフレームとIDリストを返す
    """
    pos_path = os.path.join(folder_path, f"{folder_id}-position.csv")
    
    try:
        df_pos = pd.read_csv(pos_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {pos_path}")
        return None, None

    # カラム名の正規化
    pos_map = {}
    for col in df_pos.columns:
        if col.startswith('x') and col[1:].isdigit():
            pos_map[col] = f"x_{col[1:]}"
        elif col.startswith('y') and col[1:].isdigit():
            pos_map[col] = f"y_{col[1:]}"
    if pos_map:
        df_pos.rename(columns=pos_map, inplace=True)

    # ID抽出
    ids = []
    for col in df_pos.columns:
        if col.startswith('x_'):
            ids.append(col.replace('x_', ''))
            
    # 数値ID順にソート
    ids.sort(key=lambda x: int(x))
            
    return df_pos, ids

# ==========================================
# 2. 計算ロジック (ネットワーク構築)
# ==========================================

def calculate_network_stats(df_pos, ids, contact_threshold_px):
    """
    個体間の距離を計算し、接触ネットワークの統計量を算出する
    """
    n_frames = len(df_pos)
    if n_frames == 0:
        return [], {}, 0

    # 座標データの抽出
    coords = {}
    for uid in ids:
        col_x, col_y = f"x_{uid}", f"y_{uid}"
        if col_x in df_pos.columns and col_y in df_pos.columns:
            coords[uid] = df_pos[[col_x, col_y]].values
        else:
            return [], {}, 0

    edges = []
    # ノードごとの統計初期化
    node_stats = {uid: {'total_contact_count': 0} for uid in ids}

    # ペアごとの距離計算
    for id1, id2 in combinations(ids, 2):
        # ユークリッド距離
        dist = np.sqrt(np.sum((coords[id1] - coords[id2])**2, axis=1))
        
        # 閾値以下のフレーム数をカウント
        contact_frames = np.sum(dist <= contact_threshold_px)
        contact_rate = (contact_frames / n_frames) * 100.0
        contact_sec  = contact_frames / 2.0  # 2.0 fps換算
        
        # エッジリストに追加
        edges.append({
            'u': id1,
            'v': id2,
            'count': contact_sec,
            'rate': contact_rate
        })
        
        # ノードごとの延べ接触回数加算
        node_stats[id1]['total_contact_count'] += contact_sec
        node_stats[id2]['total_contact_count'] += contact_sec

    return edges, node_stats, n_frames

# ==========================================
# 3. グラフ描画 (ダッシュボード)
# ==========================================

def get_layout_params(n_plots, single_fig_size):
    if n_plots <= 1: return 1, 1, single_fig_size
    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    total_w = single_fig_size[0] * n_cols
    total_h = single_fig_size[1] * n_rows
    return n_rows, n_cols, (total_w, total_h)

def plot_social_dashboard(target_dict, mode_label, base_path, contact_thresh, fig_size, save_path, auto_save, stats_container):
    """
    指定されたコロニー群のネットワーク図をダッシュボード形式で作成
    """
    print(f"\n - 解析開始: {mode_label} ")
    
    n_plots = len(target_dict)
    if n_plots == 0: return

    # レイアウト計算
    n_rows, n_cols, figsize = get_layout_params(n_plots, fig_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1: axes = [axes]
    else: axes = axes.flatten()

    sorted_colonies = sorted(target_dict.keys())
    
    # フレームレート定義 (他のコードと同様に2.0fpsとする)
    fps = 2.0
    
    for i, colony_name in enumerate(sorted_colonies):
        if i >= len(axes): break
        ax = axes[i]
        folder_id = target_dict[colony_name]
        folder_path = os.path.join(base_path, folder_id)
        
        # データ読み込み
        df_pos, ids = load_data(folder_path, folder_id)
        
        if df_pos is None or len(ids) < 2:
            ax.text(0.5, 0.5, 'No Data or Single Ant', ha='center')
            ax.set_title(colony_name)
            ax.axis('off')
            continue

        # 計算
        edges_data, node_stats, total_frames = calculate_network_stats(df_pos, ids, contact_thresh)
        
        # --- CSV用データ収集 ---
        for e in edges_data:
            stats_container.append({
                "Group_Type": mode_label,
                "Colony_Name": colony_name,
                "ID_A": e['u'],
                "ID_B": e['v'],
                "Contact_Count": e['count'],
                "Total_Frames": total_frames,
                "Contact_Rate_%": round(e['rate'], 2),
                "Total_Contact_Count_A": node_stats[e['u']]['total_contact_count'],
                "Total_Contact_Count_B": node_stats[e['v']]['total_contact_count']
            })

        # --- ネットワークグラフ構築 ---
        G = nx.Graph()
        G.add_nodes_from(ids)
        
        # ノードラベルの作成 (ID, Rate, Time)
        node_labels = {}
        for uid in ids:
            count = node_stats[uid]['total_contact_count']
            # 延べ接触率 (Total Rate)
            rate = (count / total_frames * 100.0) if total_frames > 0 else 0
            
            # 時間換算 (分)
            time_sec = count / (fps)
            time_min = time_sec / 60.0

            # 改行を入れて3段表示にする: 
            node_labels[uid] = f"{uid}"

        # エッジの追加
        valid_edges = []
        edge_weights = []
        edge_labels_dict = {}
        
        for e in edges_data:
            if e['count'] > 0:
                G.add_edge(e['u'], e['v'], weight=e['rate'])
                valid_edges.append((e['u'], e['v']))
                edge_weights.append(e['rate'])
                edge_labels_dict[(e['u'], e['v'])] = f"{e['rate']:.1f}%\n({e['count']}s)"

        # レイアウト: 円周配置
        pos = nx.circular_layout(G)
        
        # 描画設定
        # ノード (サイズを少し大きくして文字が入るようにする)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', edgecolors='black', node_size=1400)
        
        # ノードラベル (フォントサイズ調整)
        nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels, font_size=12, font_weight='bold')
        
        # エッジ
        widths = [1.0 + (w * 0.1) for w in edge_weights] 
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=valid_edges, width=widths, alpha=0.7)
        
        # エッジラベル
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels_dict, font_size=12, label_pos=0.5)

        ax.set_title(f"{colony_name}", fontsize=20)
        ax.axis('off')

    # 余白処理
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Social Network Interaction ({mode_label})", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if auto_save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"グラフ保存完了: {save_path}")
        plt.close()
    else:
        plt.show()


# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    
    # 1. データ定義
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
    
    # 2. 設定
    BASE_PATH = "/Users/sonya/卒論データ/analysis_data"
    OUTPUT_DIR = "/Users/sonya/卒論データ/analysis_data/graph_output"
    
    FIG_SIZE = (4, 4)
    CONTACT_THRESHOLD_PX = 50.0 
    AUTO_SAVE = True

    if AUTO_SAVE and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_network_stats = []

    # 3. 実行ループ
    DATA_SETS = [
        (PAIR_DICT, "Pair", "Pair"),
        (TRIO_DICT, "Trio", "Trio")
    ]
    
    for target_dict, label, suffix in DATA_SETS:
        save_name = f"Social_Network_{suffix}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        plot_social_dashboard(
            target_dict, 
            label, 
            BASE_PATH, 
            CONTACT_THRESHOLD_PX,
            FIG_SIZE, 
            save_path, 
            AUTO_SAVE,
            all_network_stats
        )

    # 4. CSV保存
    if all_network_stats:
        csv_path = os.path.join(OUTPUT_DIR, "social_network_summary.csv")
        df_summary = pd.DataFrame(all_network_stats)
        
        cols_order = [
            "Group_Type", "Colony_Name", "ID_A", "ID_B", 
            "Contact_Count", "Total_Frames", "Contact_Rate_%", 
            "Total_Contact_Count_A", "Total_Contact_Count_B"
        ]
        df_summary = df_summary[cols_order]
        
        df_summary.to_csv(csv_path, index=False)
        print(f"\nネットワーク統計データを保存しました: {csv_path}")

    print("\nAll processes completed.")