import cv2
import glob
import os
import shutil
import time
import datetime
from datetime import datetime
import queue
import threading
dt_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # 現在の日時を取得


# --- 定数定義 ---
# プレビュー用のウィンドウ名
WINDOW_NAME = "UMATracker Timelapse Creator"


### 初期設定
def setup_camera(camera_num, output_path):
    # 今日の日付を取得 (例: 20250831
    
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
            os.mkdir(folder_path)
            break
        
        # フォルダが存在すれば、次の番号を試す
        count += 1

    # 撮影前にカメラの動作確認
    print("--------------------------------------------------")
    print("デバッグ用のプレビューを表示します。")
    cap = cv2.VideoCapture(camera_num)
    if not cap.isOpened():
        print(f"エラー: カメラ {camera_num} を開けませんでした。")
        return None, None # 変更点: エラー時はNoneを返す
    
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
        return None, None # エラー時はNoneを返す

    cv2.imshow(WINDOW_NAME, frame)
    print("カメラ映像が表示されます。問題なければ、エンターキーを押してください。")
    print("（このキー入力で、タイムラプス撮影が開始されます）")
    while True:
        if cv2.waitKey(1) & 0xFF == 13: # 13はEnterキーのキーコード
            break
    # cv2.destroyWindow("Debug Preview")
    print("--------------------------------------------------")

    print(f"画像をフォルダ '{folder_path}' に保存します。")
    return folder_path, cap

    # # waiting_time秒待ってから撮影をスタートさせる
    # # capture_interval = 0.5 # 画像取得間隔（秒）
    # waiting_time = 0
    # print('Recording will be started in {0} seconds'.format(waiting_time))
    # time.sleep(waiting_time)
    # print('Start')

### 画像の撮影
def capture(output_path, cap, capture_interval, burst_num):
    q = queue.Queue()

    # ファイル保存をバックグラウンドで行う関数
    def saver():
        while True:
            path, frame = q.get() # キューからデータ取得
            if frame is None: # 終了シグナル
                break
            cv2.imwrite(path, frame)
            q.task_done()

    # 保存用スレッドを開始
    saver_thread = threading.Thread(target=saver, daemon=True)
    saver_thread.start()
    count = 1 # 撮影枚数のカウント、ファイル名に使用する。
    stop_requested = False # 変更点2: ループを抜けるためのフラグを追加
    print(f"撮影を開始します。約{capture_interval}秒ごとに{burst_num}枚撮影します。")
    print("撮影を終了する場合はエンターキーを押してください。")
    print("recording start",dt_now) # 撮影開始時の時刻を記録


    while True: # capture_interval秒ごとに画像の読み込みおよび保存を行う。
        for _ in range(burst_num):
            ret, frame = cap.read() # カメラからキャプチャされた画像をframeとして読み込む
            cv2.imshow(WINDOW_NAME, frame) # frameを画面に表示。なぜかこいつを残しておかないとenterで操作を止められない。
            k = cv2.waitKey(1)&0xff # キー入力を待つ。引数は入力待ち時間。
            print("撮影枚数:{0}".format(count)) # 撮影枚数の確認

            # ファイルへの保存
            path = os.path.join(output_path, f"{count:04d}.jpg")
            # cv2.imwrite(path, frame) # ← 元の処理をコメントアウト
            q.put((path, frame)) # ← 代わりにキューへ送る
            count += 1


            # エンターキーを押したら撮影終了
            if k == 13:
                stop_requested = True # フラグを立てる
                break 
        
        if stop_requested:
            break # 外側のwhileループを抜ける

        time.sleep(capture_interval) # capture_interval秒待つ

    print("撮影完了、残りの画像の保存を待っています...")
    q.put((None, None)) # スレッドに終了を通知
    # q.join() # キューのすべてのタスクが終わるまで待機
    print("全ての画像の保存が完了しました。")
    cap.release()
    cv2.destroyAllWindows()

### ファイル名をゼロ埋め連番にリネーム
def rename_files(output_path):
    # 撮影した画像を昇順で取得
    files = sorted(glob.glob('{0}/*.jpg'.format(output_path)))
    total_files = len(files)
    
    # 総枚数の桁数に合わせてゼロ埋めの桁数を決定（例: 120枚なら3桁）
    padding = len(str(total_files))
    
    print(f"{total_files}枚の画像を{padding}桁の連番にリネームします...")

    # 取得したファイルを1つずつリネーム
    folder_basename = os.path.basename(output_path)
    for i, file_path in enumerate(files):
        new_name = os.path.join(output_path, f"{folder_basename}_{i+1:0{padding}d}.jpg")
        os.rename(file_path, new_name)
    
    print("リネーム完了")

### 画像のタイムラプス化
def timelaps(output_path):
    images = sorted(glob.glob('{0}/*.jpg'.format(output_path)))
    print("画像の総枚数{0}".format(len(images)))

    if not images:
        print("画像がないため、動画を作成できません。")
        return

    # output_pathから日付_連番のフォルダ名を抽出
    date = os.path.basename(output_path)
    
    # フレームレートの設定、30fpsになるように調整
    # frame_rate = 2 if len(images) < 30 else len(images) / 30
    frame_rate = 66.66666
    if frame_rate*1000 > 65535:
        frame_rate = round(frame_rate)


    img_for_size = cv2.imread(images[0])
    height, width, layers = img_for_size.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 動画のコーデックをmp4に指定
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')

    # 動画の保存先を画像フォルダ内に指定
    video_path = os.path.join(output_path, f"{date}.mp4")
    video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    print(f"動画を '{video_path}' に変換中...")
    
    for image_path in images:
        img = cv2.imread(image_path)
        # サイズが異なる場合はリサイズ
        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height))
        video.write(img)
    
    video.release()
    print("動画変換完了")

### キャプチャした画像の削除
def delete_captured_images(output_path):
    print(f"フォルダ '{output_path}' 内の元画像を削除します。")
    image_files = glob.glob(f'{output_path}/*.jpg')
    for file_path in image_files:
        os.remove(file_path)
    print("画像ファイルの削除が完了しました。")

if __name__ == '__main__':
    start_time = time.time()

    # --- 撮影設定 ---
    CAMERA_NUM = 0  # PC内蔵カメラは0、USBカメラは1, 2...
    CAPTURE_INTERVAL = 1.0  # 画像取得間隔（秒）
    CAPTURE_NUM_OF_INTERVAL = 2  # 1間隔あたりの撮影枚数
    OUTPUT_PATH  = "/mnt/d/datas/capture_data/" #ファイルパス(Windows_SSD)
    # OUTPUT_PATH = "/mnt/d/datas/test_data" #ファイルパス(Windows_HDD),テスト用

    # --- 実行する処理の選択 --- # 必要に応じてTrue/Falseを切り替える
    DO_CAPTURE = True 
    DO_RENAME = False
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
            output_path_full = '/mnt/d/datas/test_data/20251008_02'# 例: "./capture_data/20251008_01"

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

# if __name__ == '__main__':
#     start = time.time()

#     # 撮影設定
#     CAMERA_NUM = 0 # カメラ番号,PC本体の場合は0を使用。
#     CAPTURE_INTERVAL = 1.0# 画像取得間隔（秒）
#     CAPTURE_NUM_OF_INTERVAL = 2 # 1間隔あたりの撮影枚数
#     # OUTPUT_PATH = "/Users/sonya/Library/CloudStorage/OneDrive-HiroshimaCityUniversity/2025/UMATracker/datas/capture_data" #ファイルパス(macOS)
#     # OUTPUT_PATH = "/Users/sonya/Library/CloudStorage/OneDrive-HiroshimaCityUniversity/2025/UMATracker/datas/test_data" #ファイルパス(macOS),テスト用
#     # OUTPUT_PATH = "/mnt/c/Users/Student/OneDrive - Hiroshima City University/2025/UMATracker/datas/capture_data" #ファイルパス(Linux)
#     # OUTPUT_PATH = "/mnt/c/Users/Student/OneDrive - Hiroshima City University/2025/UMATracker/datas/test_data" #ファイルパス(Linux),テスト用
#     # OUTPUT_PATH  = "/mnt/d/datas/capture_data/" #ファイルパス(Windows_SSD)
#     OUTPUT_PATH = "/mnt/d/datas/test_data" #ファイルパス(Windows_HDD),テスト用

#     cap = None
#     try:
#         # setup_cameraはフルパスを返すので、変数名をoutput_path_fullに変更
#         output_path_full, cap = setup_camera(CAMERA_NUM, OUTPUT_PATH)
#         if output_path_full and cap:
#             # 各関数には、setup_cameraが返したフルパスを渡す
#             capture(output_path_full, cap, CAPTURE_INTERVAL, CAPTURE_NUM_OF_INTERVAL)
#             rename_files(output_path_full)
#             # timelaps(output_path_full)
#             # delete_captured_images(output_path_full)
#     except Exception as e:
#         print(f"エラーが発生しました: {e}")
#         print("error",dt_now)# エラー発生時の時刻を記録
#     # finally:
#     #     # 最後にカメラを解放する
#     #     if cap:
#     #         cap.release()
#     #     cv2.destroyAllWindows()

#     elapsed_time = time.time() - start
    # print ("処理にかかった時間は:{0:.2f}".format(elapsed_time) + "[sec]")

# # 動画の作成のみを行う場合
# if __name__ == '__main__':
#     # 撮影済みの画像が保存されているフォルダのフルパスを指定します
#     target_folder = "/mnt/d/datas/capture_data/20251007_02"

#     # timelaps関数だけを呼び出して動画を再作成します
#     try:
#         print(f"フォルダ '{target_folder}' の画像から動画を再作成します。")
#         timelaps(target_folder)
#     except Exception as e:
#         print(f"エラーが発生しました: {e}")
