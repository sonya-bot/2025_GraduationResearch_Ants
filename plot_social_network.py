# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from itertools import combinations

def plot_social_network(input_position_csv, contact_threshold, fig_size, auto_save):
    """
    個体の位置データから接触ネットワークを計算し、グラフとして描画する。

    Args:
        input_position_csv (str): 位置情報(x, y)が含まれるCSVファイル名。
        contact_threshold (float): 2個体が「接触している」と判断する最大距離（ピクセル）。
    """
# 1. データの読み込み
    try:
        df = pd.read_csv(input_position_csv)
        print(f"'{input_position_csv}'を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {input_position_csv}")
        return

# 2. 個体IDの特定
    x_cols = [col for col in df.columns if col.startswith('x')]
    individual_ids = [col[1:] for col in x_cols]
    n_individuals = len(individual_ids)
    print(f"{n_individuals} 個体を対象に接触回数を計算します。")
    print(f"接触距離のしきい値: {contact_threshold} ピクセル")
    # 個体数が1の場合、エラーを表示して終了
    if n_individuals < 2:
        print("エラー: 個体が2つ未満のため、接触ネットワークを計算できません。")
        return

# 3. 接触回数を格納する辞書を準備
    frames = df["position"]
    contact_counts = {}
    
# 4. 全ての個体のペアについてループ
    for id1, id2 in combinations(individual_ids, 2):
        pos1 = df[[f'x{id1}', f'y{id1}']].to_numpy()
        pos2 = df[[f'x{id2}', f'y{id2}']].to_numpy()

        distances = np.sqrt(np.sum((pos1 - pos2)**2, axis=1))
        contact_frames = np.sum(distances <= contact_threshold)
        
        if contact_frames > 0:
            contact_counts[(id1, id2)] = contact_frames

# 5. ネットワークグラフの構築
    G = nx.Graph()
    G.add_nodes_from(individual_ids)

    for pair, count in contact_counts.items():
        G.add_edge(pair[0], pair[1], weight=count)
    
# 6. グラフの描画
    plt.figure(figsize=fig_size)
    
    # ノードを円周上に、ID '0' を基準に時計回りで配置するレイアウトを計算
    sorted_ids = sorted(G.nodes(), key=int)
    n_nodes = len(sorted_ids)
    pos = {}
    
    # 円の角度を設定 (真上が90度 = pi/2)
    angle_start = np.pi / 2 
    # 時計回りに配置するため、角度を減算していく
    angle_step = -2 * np.pi / n_nodes 

    for i, node_id in enumerate(sorted_ids):
        angle = angle_start + i * angle_step
        # 角度からx, y座標を計算
        x = np.cos(angle)
        y = np.sin(angle)
        pos[node_id] = (x, y)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    if weights:
        max_weight = max(weights)
        edge_widths = [w / max_weight * 15 for w in weights]
    else:
        edge_widths = []

    # ノードを描画
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='skyblue',
                             edgecolors='black', linewidths=1.5)
    
    # エッジを描画
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.8)
    
    # ノードラベル（個体ID）を "ID: (id)" 形式で描画
    custom_labels = {node: f"ID:{node}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=custom_labels, font_size=12, font_family='sans-serif', font_weight='bold')
    
    # エッジラベル（接触回数）を描画
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10,font_color='skyblue',
                                 bbox=dict(facecolor='white', alpha=0, edgecolor='none', pad=0.1))
    
    # bboxの各変数を明記
    """
    bbox=dict(
        facecolor='white', # 背景色
        alpha=0.9,        # 透明度
        edgecolor='none', # 枠線の色
        pad=0.1           # 余白
    )
    """

     # タイトルと表示設定

    plt.title(f"Contact Frequency (Total {len(frames)} frames)", fontsize=16)
    plt.axis('off')

    # グラフの自動保存設定
    if auto_save:
        output_filename = f"Contact_Frequency.png"
        output_directory = os.path.dirname(input_position_csv)
        save_path = os.path.join(output_directory, output_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" - 接触ネットワークグラフを保存しました: {save_path}")
    else:
        print(f" - 接触ネットワークグラフを表示します:")
        plt.show()

# メイン処理
if __name__ == '__main__':
    INPUT_CSV = "20251101_01"
    INPUT_POSITION_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position.csv"
    CONTACT_THRESHOLD = 50.0
    FIG_SIZE = (10, 5)
    AUTO_SAVE = False
    
    plot_social_network(INPUT_POSITION_CSV, CONTACT_THRESHOLD, FIG_SIZE, AUTO_SAVE)