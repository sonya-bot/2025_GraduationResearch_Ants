# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import numpy.fft as fft 
import matplotlib.pyplot as plt
import platform
import os
import networkx as nx
import sys
from itertools import combinations 

def plot_contact_duration_distribution(position_csv_path, contact_threshold, fig_size, auto_save, use_loglog_plot, show_pair_breakdown):
    """
    位置データから、個体の「所属グループサイズ」ごとの持続時間分布を計算し、
    片対数グラフ（Y軸対数）で描画する。
    NetworkXを用いて連結成分分解を行い、排他的なグループサイズを判定する。
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
    
    print(f"{n_individuals} 個体を検出しました。グループサイズの持続時間を解析します...")

    # 持続時間の格納用辞書(全体)
    # Key: サイズ (1, 2, ..., N), Value: 持続時間(秒)のリスト
    duration_storage = {size: [] for size in range(1, n_individuals + 1)}
    # 個体の状態追跡用辞書
    # Key: 個体ID, Value: {'current_size': int, 'frame_count': int}
    tracker = {uid: {'current_size': 0, 'frame_count': 0} for uid in individual_ids}
    # ペア別 (Size 2の内訳用)
    pair_combinations = list(combinations(individual_ids, 2))
    pair_duration_storage = {pair: [] for pair in pair_combinations}
    pair_tracker = {pair: {'active': False, 'frame_count': 0} for pair in pair_combinations}

    # 座標データをNumpy配列化して高速化 (shape: フレーム数 x 2)
    coords_dict = {}
    for uid in individual_ids:
        coords_dict[uid] = df_pos[[f'x{uid}', f'y{uid}']].to_numpy()

    total_frames = len(df_pos)

# 3.FPSの定義
    FPS = 2.0 # 1フレーム=1/2秒
    sample_spacing_second = 1.0 / FPS # 周波数計算用にサンプリング間隔(秒)を計算
    # 時間を時間(秒)から時間(分)に変更
    sample_spacing_minutes = sample_spacing_second / 60 
    print(f"  - サンプリング周波数: {FPS} Hz (サンプリング間隔: {sample_spacing_minutes} m)")

# 4.フレームごとのネットワーク解析と状態追跡
    print("フレームごとの解析を実行中...")
    # カウンターの初期化
    stats_trio_chain_frames = 0
    stats_trio_triangle_frames = 0
    
    for i in range(total_frames):
        # 4-1. ネットワーク構築 (距離判定)
        G = nx.Graph()
        G.add_nodes_from(individual_ids)
        
        # 全ペアの距離を計算し、閾値以下ならエッジを追加
        for j in range(n_individuals):
            for k in range(j + 1, n_individuals):
                id_a = individual_ids[j]
                id_b = individual_ids[k]
                
                pos_a = coords_dict[id_a][i]
                pos_b = coords_dict[id_b][i]
                
                dist = np.sqrt(np.sum((pos_a - pos_b)**2))
                
                if dist <= contact_threshold:
                    G.add_edge(id_a, id_b)

        if i == total_frames - 1:
            print(f" - ネットワーク(全 {total_frames} Frame)を構築しました。")

        # 4-2. 連結成分の特定と状態更新
        connected_components = list(nx.connected_components(G))
        
        # 個体ID -> 現在のグループサイズ のマッピングを作成
        current_frame_sizes = {}

        # size.2のフレームを特定
        current_size2_pairs = set()
        
        for component in connected_components:
            size = len(component)
            # pair状態の判定
            if size == 2:
                # componentはセットなので、ID順にソートしてタプル化し、キーとする
                # (individual_idsの順序に従う)
                comp_list = list(component)
                comp_list.sort(key=lambda x: individual_ids.index(x))
                pair_key = tuple(comp_list)
                current_size2_pairs.add(pair_key)

            # trio状態の判定
            if size == 3:
                # この成分（グループ）の部分グラフを作成
                subgraph = G.subgraph(component)
                num_edges = subgraph.number_of_edges()
                
                # 接触状態別の判定
                # エッジが3本 = Triangle (全結合)
                if num_edges == 3:
                    stats_trio_triangle_frames += 1
                # エッジが2本 = Chain (鎖状)
                elif num_edges == 2: # chainをtrio結合としてみなさない場合は、これをコメントアウト
                    stats_trio_chain_frames += 1

                # グラフ描画用には、形状に関わらず「Size 3」として統合して記録
            for uid in component:
                current_frame_sizes[uid] = size

            if i == total_frames - 1:
                print(f" - 連結成分(全 {total_frames} Frame)を特定しました。")

        # 4-3. 各個体の状態更新と持続時間記録
        for uid in individual_ids:
            new_size = current_frame_sizes[uid]
            prev_size = tracker[uid]['current_size']
            count = tracker[uid]['frame_count']

            if new_size == prev_size:
                # 状態継続: カウントアップ
                tracker[uid]['frame_count'] += 1
            else:
                # 状態変化: 前の状態を記録 (初期状態0は除外)
                if prev_size > 0:
                    duration_sec = count / FPS
                    duration_storage[prev_size].append(duration_sec)
                
                # 新しい状態にリセット
                tracker[uid]['current_size'] = new_size
                tracker[uid]['frame_count'] = 1

        # 4-4.ペア別の状態記録
        for pair in pair_combinations:
            # このペアが現在 Size 2 のグループを形成しているか？
            is_active = pair in current_size2_pairs
            
            if is_active:
                if pair_tracker[pair]['active']:
                    # 継続中
                    pair_tracker[pair]['frame_count'] += 1
                else:
                    # 開始
                    pair_tracker[pair]['active'] = True
                    pair_tracker[pair]['frame_count'] = 1
            else:
                if pair_tracker[pair]['active']:
                    # 終了
                    dur = pair_tracker[pair]['frame_count'] / FPS
                    pair_duration_storage[pair].append(dur)
                    pair_tracker[pair]['active'] = False
                    pair_tracker[pair]['frame_count'] = 0

    # ループ終了後の後処理 (最後の継続時間を記録)
    # 個体全体
    for uid in individual_ids:
        size = tracker[uid]['current_size']
        count = tracker[uid]['frame_count']
        if size > 0:
            duration_sec = count / FPS
            duration_storage[size].append(duration_sec)
    
    # ペアごと
    for pair in pair_combinations:
        if pair_tracker[pair]['active']:
            dur = pair_tracker[pair]['frame_count'] / FPS
            pair_duration_storage[pair].append(dur)

# 5.グラフの描画

    for size in range(1, n_individuals + 1):
        durations = duration_storage[size]
        
        if len(durations) > 0:
            # 個別のFigureを作成
            plt.figure(figsize=fig_size)
            ax = plt.gca()

            label_text = f""
            if size == 1: label_text += "Isolated"
            elif size == 2: label_text += "Pair"
            elif size == 3: label_text += "Trio"


            # 最大値と最小値の取得
            data_max = np.max(durations)
            data_min = np.min(durations)
            # bin数のパラメータ化
            loglog_bin = 30
            semilog_bin = 40
            
            # 対数スケールの設定
            if use_loglog_plot:
                # ax.set_xlim(0.1,100)
                bins = np.logspace(np.log10(data_min), np.log10(data_max), loglog_bin)
                ax.set_xscale('log')
                ax.set_xlabel('Duration (s) [Log Scale]', fontsize=12)
            else:
                bins = np.linspace(data_min, data_max, semilog_bin)
                ax.set_xlabel('Duration (s)', fontsize=12)
                ax.set_xlim(data_min, data_max)

            # ヒストグラム描画
            weights = np.ones_like(durations) / len(durations) * 100
            # bins = np.logspace(np.log10(0.1), np.log10(100), 30) if use_loglog_plot else 30
            ax.hist(durations, bins=bins, weights=weights,
                    label=label_text,
                    alpha=0.7, edgecolor='black', histtype='bar', log=True)
            
            # ペアごとの内訳表示 (Size 2の場合のみ)
            if size == 2 and show_pair_breakdown:
                total_count_size2 = len(durations)
                
                for i, pair in enumerate(pair_combinations):
                    pair_durs = pair_duration_storage[pair]
                    if len(pair_durs) > 0:
                        # 重み: Size 2 全体に対する割合を表示 (内訳なので)
                        pair_weights = np.ones_like(pair_durs) / total_count_size2 * 100
                        
                        pair_label = f"Pair {pair[0]},{pair[1]}"
                        # ステッププロット (階段状の線) で重ねる
                        ax.hist(pair_durs, bins=bins, weights=pair_weights,
                                label=pair_label,
                                histtype='step', linewidth=2.0, log=True)
    
            ax.set_ylim(0.1, 100)  # Y軸は対数スケールなので下限を0.1に設定
            ax.set_yscale('log')
            ax.set_ylabel('Frequency [Log Scale]', fontsize=12)
            
            ax.set_title(f'Contact Duration Histogram: {label_text} (N={n_individuals})', fontsize=14)
            ax.grid(True, which="major", axis='y', linestyle='--', alpha=0.7)
            ax.grid(True, which="major", axis='x', linestyle=':', alpha=0.4)
            ax.legend(fontsize=10, loc='upper right')
            
            # コンソール出力
            mean_val = np.mean(durations)
            print(f" [Size {size}] Count: {len(durations)}, Mean: {mean_val:.2f}s")

            # 保存または表示
            if auto_save:
                output_filename = f"Contact_Duration_Distribution({label_text},N={n_individuals}).png"
                output_directory = os.path.dirname(position_csv_path)
                save_path = os.path.join(output_directory, output_filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f" - グラフを保存しました: {save_path}")
                plt.close()
            else:
                print(" - グラフを表示します")
                plt.show()

def plot_contact_duration_cumlative_sum(position_csv_path, contact_threshold, fig_size, auto_save, use_loglog_plot, show_pair_breakdown):
    """
    位置データから、個体の「所属グループサイズ」ごとの持続時間を計算し、
    持続時間が短い順にソートを行い、累積分布を見る
    NetworkXを用いて連結成分分解を行い、排他的なグループサイズを判定する。

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
    
    print(f"{n_individuals} 個体を検出しました。グループサイズの持続時間を解析します...")

    # 持続時間の格納用辞書(全体)
    # Key: サイズ (1, 2, ..., N), Value: 持続時間(秒)のリスト
    duration_storage = {size: [] for size in range(1, n_individuals + 1)}
    # 個体の状態追跡用辞書
    # Key: 個体ID, Value: {'current_size': int, 'frame_count': int}
    tracker = {uid: {'current_size': 0, 'frame_count': 0} for uid in individual_ids}
    # ペア別 (Size 2の内訳用)
    pair_combinations = list(combinations(individual_ids, 2))
    pair_duration_storage = {pair: [] for pair in pair_combinations}
    pair_tracker = {pair: {'active': False, 'frame_count': 0} for pair in pair_combinations}

    # 座標データをNumpy配列化して高速化 (shape: フレーム数 x 2)
    coords_dict = {}
    for uid in individual_ids:
        coords_dict[uid] = df_pos[[f'x{uid}', f'y{uid}']].to_numpy()

    total_frames = len(df_pos)

# 3.FPSの定義
    FPS = 2.0 # 1フレーム=1/2秒
    sample_spacing_second = 1.0 / FPS # 周波数計算用にサンプリング間隔(秒)を計算
    # 時間を時間(秒)から時間(分)に変更
    sample_spacing_minutes = sample_spacing_second / 60 
    print(f"  - サンプリング周波数: {FPS} Hz (サンプリング間隔: {sample_spacing_minutes} m)")

# 4.フレームごとのネットワーク解析と状態追跡
    print("フレームごとの解析を実行中...")
    # カウンターの初期化
    stats_trio_chain_frames = 0
    stats_trio_triangle_frames = 0
    
    for i in range(total_frames):
        # 4-1. ネットワーク構築 (距離判定)
        G = nx.Graph()
        G.add_nodes_from(individual_ids)
        
        # 全ペアの距離を計算し、閾値以下ならエッジを追加
        for j in range(n_individuals):
            for k in range(j + 1, n_individuals):
                id_a = individual_ids[j]
                id_b = individual_ids[k]
                
                pos_a = coords_dict[id_a][i]
                pos_b = coords_dict[id_b][i]
                
                dist = np.sqrt(np.sum((pos_a - pos_b)**2))
                
                if dist <= contact_threshold:
                    G.add_edge(id_a, id_b)

        if i == total_frames - 1:
            print(f" - ネットワーク(全 {total_frames} Frame)を構築しました。")

        # 4-2. 連結成分の特定と状態更新
        connected_components = list(nx.connected_components(G))
        
        # 個体ID -> 現在のグループサイズ のマッピングを作成
        current_frame_sizes = {}

        # size.2のフレームを特定
        current_size2_pairs = set()
        
        for component in connected_components:
            size = len(component)
            # pair状態の判定
            if size == 2:
                comp_list = list(component)
                comp_list.sort(key=lambda x: individual_ids.index(x))
                pair_key = tuple(comp_list)
                current_size2_pairs.add(pair_key)

            # trio状態の判定
            if size == 3:
                # この成分（グループ）の部分グラフを作成
                subgraph = G.subgraph(component)
                num_edges = subgraph.number_of_edges()
                
                # 接触状態別の判定
                # エッジが3本 = Triangle (全結合)
                if num_edges == 3:
                    stats_trio_triangle_frames += 1
                # エッジが2本 = Chain (鎖状) ,chainをtrio結合としてみなさない場合は、これをコメントアウト
                elif num_edges == 2: 
                    stats_trio_chain_frames += 1

                # グラフ描画用には、形状に関わらず「Size 3」として統合して記録
            for uid in component:
                current_frame_sizes[uid] = size

            if i == total_frames - 1:
                print(f" - 連結成分(全 {total_frames} Frame)を特定しました。")

        # 4-3. 各個体の状態更新と持続時間記録
        for uid in individual_ids:
            new_size = current_frame_sizes[uid]
            prev_size = tracker[uid]['current_size']
            count = tracker[uid]['frame_count']

            if new_size == prev_size:
                # 状態継続: カウントアップ
                tracker[uid]['frame_count'] += 1
            else:
                # 状態変化: 前の状態を記録 (初期状態0は除外)
                if prev_size > 0:
                    duration_sec = count / FPS
                    duration_storage[prev_size].append(duration_sec)
                
                # 新しい状態にリセット
                tracker[uid]['current_size'] = new_size
                tracker[uid]['frame_count'] = 1

        if i == total_frames - 1:
            print(f" - 持続時間の記録が完了しました。")

        # 4-4.ペア別の状態記録
        for pair in pair_combinations:
            # このペアが現在 Size 2 のグループを形成しているか？
            is_active = pair in current_size2_pairs
            
            if is_active:
                if pair_tracker[pair]['active']:
                    # 継続中
                    pair_tracker[pair]['frame_count'] += 1
                else:
                    # 開始
                    pair_tracker[pair]['active'] = True
                    pair_tracker[pair]['frame_count'] = 1
            else:
                if pair_tracker[pair]['active']:
                    # 終了
                    dur = pair_tracker[pair]['frame_count'] / FPS
                    pair_duration_storage[pair].append(dur)
                    pair_tracker[pair]['active'] = False
                    pair_tracker[pair]['frame_count'] = 0

        if i == total_frames - 1:
            print(f" - ペア別の状態記録が完了しました。")

    # 4-5. ループ終了後の後処理 (最後の継続時間を記録)
    # 個体全体
    for uid in individual_ids:
        size = tracker[uid]['current_size']
        count = tracker[uid]['frame_count']
        if size > 0:
            duration_sec = count / FPS
            duration_storage[size].append(duration_sec)
    
    # ペアごと
    for pair in pair_combinations:
        if pair_tracker[pair]['active']:
            dur = pair_tracker[pair]['frame_count'] / FPS
            pair_duration_storage[pair].append(dur)
        

    # 4-5 記録したデータを昇順にソート
    calculated_group_data = {} # キー: size, 値: (ranks, cumsum)
    calculated_pair_data = {}  # キー: pair, 値: (ranks, cumsum)

    # (1) グループサイズごとの計算
    for s in range(1, n_individuals + 1):
        durs = duration_storage[s]
        if len(durs) > 0:
            durs.sort() # 昇順ソート
            cs = np.cumsum(durs) # 累積和
            r = np.arange(1, len(durs) + 1) # ランク
            # 結果を保存
            calculated_group_data[s] = (r, cs)

    # (2) ペアごとの計算 (Size 2 内訳用)
    for pair in pair_combinations:
        p_durs = pair_duration_storage[pair]
        if len(p_durs) > 0:
            p_durs.sort() # 昇順ソート
            p_cs = np.cumsum(p_durs) # 累積和
            p_r = np.arange(1, len(p_durs) + 1) # ランク
            # 結果を保存
            calculated_pair_data[pair] = (p_r, p_cs)

    print(f" - 持続時間のソートと累積和計算が完了しました")

# 4.グラフの描画 (サイズごと)
    for size in range(1, n_individuals + 1):
        durations = duration_storage[size]
        
        if len(durations) > 0:
            # 保存しておいたデータを取り出す
            ranks, cumsum = calculated_group_data[size]
            # 個別のFigureを作成
            plt.figure(figsize=fig_size)
            ax = plt.gca()

            label_text = f""
            if size == 1: label_text += "Isolated"
            elif size == 2: label_text += "Pair"
            elif size == 3: label_text += "Trio"
        
            # 対数スケールの設定
            if use_loglog_plot:
                # ax.set_xlim(0.1,100)           
                # ax.set_xscale('log')
                ax.set_ylabel('Rank (ascending order)', fontsize=12)
            else:
                # ax.set_xlim(0)
                ax.set_ylabel('Rank (ascending order)', fontsize=12)
        
        
            ax.plot(cumsum, ranks, label=label_text, linewidth=3.0, color='black', alpha=0.5)

            # Size 2 の場合、ペアごとの内訳を表示
            if size == 2 and show_pair_breakdown:

                for pair in pair_combinations:
                    pair_durs = pair_duration_storage[pair]
                    if len(pair_durs) > 0:
                        pair_durs.sort() # 昇順ソート
                        p_cumsum = np.cumsum(pair_durs)
                        p_ranks = np.arange(1, len(pair_durs) + 1)
                        
                        pair_label = f"Pair {pair[0]},{pair[1]}"
                        ax.plot(p_cumsum, p_ranks, label=pair_label, linewidth=1.5)

            # グラフ体裁
            ax.set_title(f'Contact Duration Cumulative Sum: {label_text} (N={n_individuals})', fontsize=14)
            ax.set_xlabel('Cumulative Duration (sec)', fontsize=12)
            ax.set_xlim(0,10000)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

            # コンソール出力(デバッグ用)
            # コンソール出力の追加案
            print(f" [Size {size}] Max Duration: {np.max(durations):.2f}s")
            print(f" [Size {size}] Top 10 longest: {sorted(durations, reverse=True)[:10]}")

            # 保存または表示
            if auto_save:
                output_filename = f"Contact_Duration_Cumulative_Sum({label_text},N={n_individuals}).png"
                output_directory = os.path.dirname(position_csv_path)
                save_path = os.path.join(output_directory, output_filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f" - グラフを保存しました: {save_path}")
                plt.close()
            else:
                print(" - グラフを表示します")
                plt.show()

    print("\n処理が完了しました。")


# 実行プログラムの設定
RUN_PLOT_CONTACT_DURATION_DISTRIBUTION = False
RUN_PLOT_CONTACT_DURATION_CUMULATIVE_SUM = True


if __name__ == "__main__":
    # 位置データの入力
    INPUT_CSV = "20251101_01"
    INPUT_POSITION_CSV = f"/Volumes/100.108.13.8/analysis_data/{INPUT_CSV}/{INPUT_CSV}-position.csv"
    # 接触判定に使用するしきい値
    CONTACT_THRESHOLD = 50.0  # ピクセル単位の接触しきい値
    # グラフのサイズを指定
    FIG_SIZE = (10, 5)
    # グラフの自動保存設定
    AUTO_SAVE = False
    # 対数スケールの設定
    USE_LOGLOG_PLOT = False  # True: 両対数プロット, False: 半対数プロット

    # ペアごとの分布を表示
    SHOW_PAIR_BREAKDOWN = False

    # 関数を呼び出し
    if RUN_PLOT_CONTACT_DURATION_DISTRIBUTION:
        plot_contact_duration_distribution(INPUT_POSITION_CSV, CONTACT_THRESHOLD, FIG_SIZE, AUTO_SAVE, USE_LOGLOG_PLOT, SHOW_PAIR_BREAKDOWN)
    else:
        pass

    if RUN_PLOT_CONTACT_DURATION_CUMULATIVE_SUM:
        plot_contact_duration_cumlative_sum(INPUT_POSITION_CSV, CONTACT_THRESHOLD, FIG_SIZE, AUTO_SAVE, USE_LOGLOG_PLOT, SHOW_PAIR_BREAKDOWN)
    else:
        pass