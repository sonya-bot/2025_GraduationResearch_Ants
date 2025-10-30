import cv2
import glob
import os
import shutil
import time
import datetime
from datetime import datetime
import queue
import threading
import subprocess # FFmpegを呼び出すために追加
import re # 追加: 正規表現モジュール

### 画像のタイムラプス化 (FFmpegを使用) ###
def timelaps(input_path, use_gpu):
    """
    指定されたフォルダ内の画像をFFmpegを使ってタイムラプス動画に変換する。
    GPU (h264_nvenc) を使用して高速処理を行う。
    """
    output_path = input_path
    print(f'output_path: {output_path}') #動画を作成する対象のファイル名を表示
    images = sorted(glob.glob(os.path.join(output_path, '*.jpg')))
    print(f"画像の総枚数: {len(images)}")

    if not images:
        print("画像がないため、動画を作成できません。")
        return

    # 1枚目の画像を読み込んで、動画の解像度を取得
    img_for_size = cv2.imread(images[0])
    height, width, _ = img_for_size.shape
    print(f"動画の解像度: {width}x{height}")
    
    # フォルダ名から動画ファイル名を決定
    date = os.path.basename(output_path)
    video_path = os.path.join(output_path, f"{date}.mp4")

    # フレームレートを設定
    frame_rate = 2.0 #1枚の画像を0.5秒表示 (2fps)

    # --- FFmpegのコマンドを組み立てる ---
    command = [
        '/usr/local/bin/ffmpeg', # ★★★ FFmpegのフルパスを直接指定 ★★★
        '-y',  # 出力ファイルを無条件に上書き
        '-f', 'rawvideo',  # 入力フォーマットをrawvideoに指定
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',  # OpenCVの画像フォーマット(BGR)を指定
        '-s', f'{width}x{height}',  # 動画の解像度を指定
        '-r', str(frame_rate),  # フレームレートを指定
        '-i', '-',  # 標準入力からデータを受け取る
    ]

    # GPU使用時のオプションを追加
    if use_gpu:
        print("GPUを使用してFFmpegで動画を作成します。")
        # --- ビデオコーデックとオプション (ここがGPU設定の核心) ---
        command.extend([
            '-c:v', 'h264_nvenc',  # NVIDIA GPUのH.264エンコーダを使用
            '-preset', 'p5',      # プリセット: p1(高品質) ~ p7(最速) の中でバランス型
            '-cq:v', '23',        # 固定品質モード (18-28が一般的。数値が低いほど高品質)
            '-pix_fmt', 'yuv420p',# 互換性の高いピクセルフォーマット
        ])
    else:
    # CPU使用時のオプションを追加
        print("CPUを使用してFFmpegで動画を作成します。")
        command.extend([
            '-c:v', 'libx264',    # CPUベースのH.264エンコーダ
            '-preset', 'medium',  # CPU負荷と品質のバランス
            '-crf', '23',         # 品質指定
            '-pix_fmt', 'yuv420p',# 互換性の高いピクセルフォーマット
        ])

    command.append(video_path) # 出力ファイルパスを追加

    print("FFmpegコマンドを実行します:")
    print(" ".join(command)) # 実行するコマンドを表示

    # FFmpegプロセスを開始
    # stdin=subprocess.PIPE で、PythonからFFmpegにデータを送れるようにする
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    
    # 全ての画像を読み込み、FFmpegの標準入力に書き込む
    try:
        for image_path in images:
            img = cv2.imread(image_path)
            # サイズが異なる場合はリサイズ (念のため)
            if (img.shape[1], img.shape[0]) != (width, height):
                img = cv2.resize(img, (width, height))
            
            # 画像データをバイト列としてプロセスに書き込む
            process.stdin.write(img.tobytes())
    except BrokenPipeError:
        print("FFmpegのプロセスが予期せず終了しました。FFmpegからのエラーメッセージを確認してください。")
    finally:
        # 全ての画像を書き込んだら、入力を閉じる
        if process.stdin:
            process.stdin.close()
        
        # FFmpegプロセスが完了するのを待つ
        process.wait()
    
    print(f"動画変換完了 動画を'{video_path}'に保存しました。")

if __name__ == '__main__':
    start_time = time.time()
    
    # --- 撮影設定 ---
    # INPUT_PATH = "/mnt/d/datas/test_data/20251015_05"
    INPUT_PATH =  "/Users/sonya/Library/CloudStorage/OneDrive-HiroshimaCityUniversity/2025/UMATracker/datas/test_data/20251015_08"

    # --- 実行する処理の選択 ---
    USE_GPU = True  # GPUを使用してFFmpegで動画を作成するかどうか

    cap = None
    output_path_full = ""
    try:
        # GPUが使用可能かチェック
        if USE_GPU:
            ffmpeg_check_cmd = ['/usr/local/bin/ffmpeg', '-hide_banner', '-hwaccels']
            result = subprocess.run(ffmpeg_check_cmd, capture_output=True, text=True)
            if 'cuda' not in result.stdout:
                print("警告: FFmpegがGPUアクセラレーション(cuda)をサポートしていません。CPUモードで実行します。")
                USE_GPU = False
            else:
                print("FFmpegがGPUアクセラレーション(cuda)をサポートしています。GPUモードで実行します。")
                # USE_GPU = True

            if os.path.exists(INPUT_PATH) and os.path.isdir(INPUT_PATH):
                print(f"タイムラプス動画の作成を開始します \n 動画ファイルは '{INPUT_PATH}' に保存されます。")
                timelaps(INPUT_PATH, USE_GPU)
            else:
                print(f"エラー: 指定されたパス '{INPUT_PATH}' が存在しません。")
    except Exception as e:
        print(f"FFmpegのチェック中にエラーが発生しました: {e}")
        USE_GPU = False

    elapsed_time = time.time() - start_time
    print(f"全ての処理が完了しました。処理時間: {elapsed_time:.2f}秒")