import cv2
import glob
import os
import shutil
import time
import datetime
from datetime import datetime
import queue
import threading

# --- 定数定義 ---
# プレビュー用のウィンドウ名
WINDOW_NAME = "UMATracker Timelapse Creator"
dt_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # 現在の日時を取得


### 初期設定
def setup_camera(camera_num, output_path):
    """
    カメラの準備と保存用フォルダの作成を行う
    """
    # 今日の日付を取得 (例: 20250831)
    today_str = datetime.now().strftime("%Y%m%d")
    count = 1

    # ベースの出力先を保持
    base_path = output_path

    # "日付_連番" のフォルダが存在しないかチェックするループ
    while True:
        # フォルダ名を生成 (例: 20250831_01)
        folder_name = f"{today_str}_{str(count).zfill(2)}"
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            break

        # フォルダが存在すれば、次の番号を試す
        count += 1

    # 撮影前にカメラの動作確認
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
        if cv2.waitKey(1) & 0xFF == 13: # 13はEnterキーのキーコード
            break
    print("--------------------------------------------------")

    print(f"画像をフォルダ '{folder_path}' に保存します。")
    return folder_path, cap


### 画像の撮影
### 画像の撮影
def capture(output_path, cap, capture_interval, burst_num):
    """
    指定された間隔で画像を撮影し、バックグラウンドで保存する
    """
    # --- 共有変数と同期オブジェクト ---
    latest_frame = None
    ret_value = False
    lock = threading.Lock()
    reader_stopped = threading.Event()
    new_frame_event = threading.Event()

    # --- リーダー・スレッド（カメラからの読み込み） ---
    def _reader_loop():
        nonlocal latest_frame, ret_value
        while not reader_stopped.is_set():
            ret, frame = cap.read()
            with lock:
                ret_value = ret
                if ret:
                    latest_frame = frame
                    new_frame_event.set()
                else:
                    new_frame_event.clear()
            time.sleep(0.001)
        print("カメラ読み込みスレッドが停止しました。")

    reader_thread = threading.Thread(target=_reader_loop, daemon=True)
    reader_thread.start()

    # --- セーバー・スレッド（ファイルへの保存） ---
    q = queue.Queue(maxsize=10)
    def _saver_loop():
        while True:
            path, frame = q.get()
            if frame is None:
                q.task_done()
                break
            cv2.imwrite(path, frame)
            q.task_done()
    saver_thread = threading.Thread(target=_saver_loop, daemon=True)
    saver_thread.start()

    # --- メインの撮影ループ ---
    count = 1
    stop_requested = False
    print(f"撮影を開始します。約{capture_interval}秒ごとに{burst_num}枚撮影します。")
    print("撮影を終了する場合はエンターキーを押してください。")
    print(f"recording start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()
    
    while not stop_requested:
        for _ in range(burst_num):
            new_frame_event.wait()
            new_frame_event.clear()
            
            with lock:
                ret, frame = ret_value, latest_frame.copy() if ret_value and latest_frame is not None else (False, None)
            
            if not ret:
                print("警告: フレームの取得に失敗しました。撮影を続行します。")
                continue

            cv2.imshow(WINDOW_NAME, frame)
            k = cv2.waitKey(1) & 0xff

            if count % 10 == 0:
                print(f"撮影枚数:{count}")

            path = os.path.join(output_path, f"{count:04d}.jpg")
            q.put((path, frame))
            count += 1

            if k == 13:
                stop_requested = True
                break

        if stop_requested:
            break

        time.sleep(capture_interval)

    # --- 終了処理 ---
    end_time = time.time()
    elapsed_time = end_time - start_time # 経過時間（秒）
    total_images = count - 1
    
    print("撮影完了、各種スレッドの終了と残りの画像の保存を待っています...")
    reader_stopped.set()
    reader_thread.join(timeout=1)
    q.put((None, None))
    q.join()

    print(f"全ての画像の保存が完了しました。総撮影枚数: {total_images}")


### ファイル名をゼロ埋め連番にリネーム
def rename_files(output_path):
    """
    撮影したJPGファイル名を連番にリネームする
    """
    print("ファイル名をリネームしています...")
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


### 画像のタイムラプス化
def timelaps(output_path):
    """
    指定フォルダ内の画像からタイムラプス動画を生成する
    """
    images = sorted(glob.glob(os.path.join(output_path, '*.jpg')))
    print(f"画像の総枚数: {len(images)}")

    if not images:
        print("画像がないため、動画を作成できません。")
        return

    date = os.path.basename(output_path)
    frame_rate = 60.0 # フレームレートを60に固定（必要に応じて調整）

    try:
        img_for_size = cv2.imread(images[0])
        height, width, _ = img_for_size.shape
    except (IndexError, cv2.error) as e:
        print(f"画像サイズの取得に失敗しました: {e}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_path, f"{date}.mp4")
    video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    print(f"動画を '{video_path}' に変換中...")
    for image_path in images:
        img = cv2.imread(image_path)
        video.write(img)

    video.release()
    print("動画変換完了")


### キャプチャした画像の削除
def delete_captured_images(output_path):
    """
    動画生成後の元画像を削除する
    """
    print(f"フォルダ '{output_path}' 内の元画像を削除します。")
    image_files = glob.glob(os.path.join(output_path, '*.jpg'))
    for file_path in image_files:
        os.remove(file_path)
    print("画像ファイルの削除が完了しました。")


if __name__ == '__main__':
    start_time = time.time()

    # --- 撮影設定 ---
    CAMERA_NUM = 0  # PC内蔵カメラは0、USBカメラは1, 2...
    CAPTURE_INTERVAL = 1.0  # 画像取得間隔（秒）
    CAPTURE_NUM_OF_INTERVAL = 2  # 1間隔あたりの撮影枚数
    # OUTPUT_PATH = "/mnt/d/datas/capture_data/" #ファイルパス(Windows_SSD)
    OUTPUT_PATH = "/mnt/d/datas/test_data" #ファイルパス(Windows_SSD),テスト用

    # --- 実行する処理の選択 ---
    DO_CAPTURE = True
    DO_RENAME = True
    DO_TIMELAPSE = False # 必要に応じてTrueに変更
    DO_DELETE_IMAGES = False # 必要に応じてTrueに変更

    cap = None
    output_path_full = ""
    try:
        if DO_CAPTURE:
            output_path_full, cap = setup_camera(CAMERA_NUM, OUTPUT_PATH)
            if output_path_full and cap:
                capture(output_path_full, cap, CAPTURE_INTERVAL, CAPTURE_NUM_OF_INTERVAL)
        else:
             # 撮影しない場合は、処理対象のフォルダを手動で指定
            output_path_full = "/path/to/your/image_folder" # 例: "./capture_data/20251008_01"

        if DO_RENAME and os.path.exists(output_path_full):
            rename_files(output_path_full)

        if DO_TIMELAPSE and os.path.exists(output_path_full):
            timelaps(output_path_full)

        if DO_DELETE_IMAGES and os.path.exists(output_path_full):
            delete_captured_images(output_path_full)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc() # 詳細なエラー情報を表示
    finally:
        # 最後にまとめてリソースを解放する
        if cap and cap.isOpened():
            print("カメラを解放しています。")
            cap.release()
        cv2.destroyAllWindows()
        print("クリーンアップ処理が完了しました。")
        
    elapsed_time = time.time() - start_time
    print(f"全ての処理が完了しました。処理時間: {elapsed_time:.2f}秒")

