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

dt_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # 現在の日時を取得


# --- 定数定義 ---
# プレビュー用のウィンドウ名
WINDOW_NAME = "UMATracker Timelapse Creator"


### 初期設定
def setup_camera(camera_num, output_path):
    """カメラの準備と保存先フォルダを作成する"""
    today_str = datetime.now().strftime("%Y%m%d")
    count = 1
    base_path = output_path

    while True:
        folder_name = f"{today_str}_{str(count).zfill(2)}"
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True) # exist_ok=Trueを追加
            break
        count += 1

    print("--------------------------------------------------")
    print("デバッグ用のプレビューを表示します。")
    cap = cv2.VideoCapture(camera_num)
    if not cap.isOpened():
        print(f"エラー: カメラ {camera_num} を開けませんでした。")
        return None, None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 100, 100)

    ret, frame = cap.read()
    if not ret:
        print("エラー: プレビュー用のフレームを取得できませんでした。")
        cap.release()
        return None, None

    cv2.imshow(WINDOW_NAME, frame)
    print("カメラ映像が表示されます。問題なければ、エンターキーを押してください。")
    print("（このキー入力で、タイムラプス撮影が開始されます）")
    while True:
        if cv2.waitKey(1) & 0xFF == 13: # 13はEnterキー
            break
    print("--------------------------------------------------")
    print(f"画像をフォルダ '{folder_path}' に保存します。")
    return folder_path, cap


### 画像の撮影
def capture(output_path, cap, capture_interval, burst_num):
    """指定された間隔で画像を撮影し、バックグラウンドで保存する"""
    q = queue.Queue()

    def saver():
        while True:
            path, frame = q.get()
            if frame is None:
                break
            cv2.imwrite(path, frame)
            q.task_done()

    saver_thread = threading.Thread(target=saver, daemon=True)
    saver_thread.start()
    count = 1
    stop_requested = False
    print(f"撮影を開始します。約{capture_interval}秒ごとに{burst_num}枚撮影します。")
    print("撮影を終了する場合はエンターキーを押してください。")
    print("recording start", dt_now)

    while not stop_requested:
        for _ in range(burst_num):
            ret, frame = cap.read()
            if not ret:
                print("エラー: フレームを読み込めませんでした。")
                stop_requested = True
                break
            
            cv2.imshow(WINDOW_NAME, frame)
            k = cv2.waitKey(1) & 0xff
            
            print(f"撮影枚数:{count}")

            path = os.path.join(output_path, f"{count:04d}.jpg")
            q.put((path, frame))
            count += 1

            if k == 13: # Enterキー
                stop_requested = True
                break
        
        if not stop_requested:
            time.sleep(capture_interval)

    print("撮影完了、残りの画像の保存を待っています...")
    q.put((None, None))
    saver_thread.join()
    print("全ての画像の保存が完了しました。")
    cap.release()
    cv2.destroyAllWindows()

### ファイル名をゼロ埋め連番にリネーム
def rename_files(output_path):
    """ファイル名を連番にリネームする"""
    files = sorted(glob.glob(os.path.join(output_path, '*.jpg')))
    total_files = len(files)
    if total_files == 0:
        print("リネーム対象のファイルがありません。")
        return

    padding = len(str(total_files))
    print(f"{total_files}枚の画像を{padding}桁の連番にリネームします...")
    folder_basename = os.path.basename(output_path)
    for i, file_path in enumerate(files):
        new_name = os.path.join(output_path, f"{folder_basename}_{i+1:0{padding}d}.jpg")
        os.rename(file_path, new_name)
    print("リネーム完了")

### 画像のタイムラプス化 (FFmpegバージョン)
def timelaps(output_path):
    """
    指定されたフォルダ内の画像をFFmpegを使ってタイムラプス動画に変換する。
    GPU (h264_nvenc) を使用して高速処理を行う。
    """
    print(f'output_path') #動画を作成する対象のファイル名を表示
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
        '/usr/local/bin/ffmpeg', # ★★★ 変更点: FFmpegのフルパスを直接指定 ★★★
        '-y',  # 出力ファイルを無条件に上書き
        '-f', 'rawvideo',  # 入力フォーマットをrawvideoに指定
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',  # OpenCVの画像フォーマット(BGR)を指定
        '-s', f'{width}x{height}',  # 動画の解像度を指定
        '-r', str(frame_rate),  # フレームレートを指定
        '-i', '-',  # 標準入力からデータを受け取る
        
        # --- ビデオコーデックとオプション (ここがGPU設定の核心) ---
        '-c:v', 'h264_nvenc',  # NVIDIA GPUのH.264エンコーダを使用
        '-preset', 'p5',      # プリセット: p1(高品質) ~ p7(最速) の中でバランス型
        '-cq:v', '23',        # 固定品質モード (18-28が一般的。数値が低いほど高品質)
        '-pix_fmt', 'yuv420p',# 互換性の高いピクセルフォーマット
        
        video_path,  # 出力ファイルパス
    ]

    print("FFmpegコマンドを実行します:")
    print(" ".join(command)) # 実行するコマンドを表示

    # FFmpegプロセスを開始
    # stdin=subprocess.PIPE で、PythonからFFmpegにデータを送れるようにする
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    print(f"動画を '{video_path}' に変換中...")
    
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


### キャプチャした画像の削除
def delete_captured_images(output_path):
    """フォルダ内のJPG画像をすべて削除する"""
    print(f"フォルダ '{output_path}' 内の元画像を削除します。")
    image_files = glob.glob(os.path.join(output_path, '*.jpg'))
    for file_path in image_files:
        os.remove(file_path)
    print("画像ファイルの削除が完了しました。")


if __name__ == '__main__':
    start_time = time.time()

    # --- 撮影設定 ---
    CAMERA_NUM = 0
    CAPTURE_INTERVAL = 1.0
    CAPTURE_NUM_OF_INTERVAL = 2
    OUTPUT_PATH = "/mnt/d/datas/capture_data/" #ファイルパス(Windows_SSD)
    # OUTPUT_PATH = "/mnt/d/datas/test_data" #ファイルパス(Windows_SSD),テスト用

    # --- 実行する処理の選択 ---
    DO_CAPTURE = False
    DO_RENAME = True
    DO_TIMELAPSE = True # ← ここをTrueにしてGPUエンコードを試す
    DO_DELETE_IMAGES = False # 動画が正しくできていることを確認してからTrueにする

    cap = None
    output_path_full = ""
    try:
        if DO_CAPTURE:
            output_path_full, cap = setup_camera(CAMERA_NUM, OUTPUT_PATH)
            if output_path_full and cap:
                capture(output_path_full, cap, CAPTURE_INTERVAL, CAPTURE_NUM_OF_INTERVAL)
        else:
            # 撮影しない場合は、処理対象のフォルダをここに手動で指定
            output_path_full = '/mnt/d/datas/capture_data/20251008_11' 

        if DO_RENAME and os.path.exists(output_path_full):
            rename_files(output_path_full)

        if DO_TIMELAPSE and os.path.exists(output_path_full):
            timelaps(output_path_full)

        if DO_DELETE_IMAGES and os.path.exists(output_path_full):
            delete_captured_images(output_path_full)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap and cap.isOpened():
            print("カメラを解放しています。")
            cap.release()
        cv2.destroyAllWindows()
        print("クリーンアップ処理が完了しました。")

    elapsed_time = time.time() - start_time
    print(f"全ての処理が完了しました。処理時間: {elapsed_time:.2f}秒")

