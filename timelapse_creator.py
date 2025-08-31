import cv2
import glob
import os
import shutil
import time
from datetime import datetime

### もろもろの初期設定
def setup_camera(camera_num):
    # 今日の日付を取得 (例: 20250831)
    today_str = datetime.now().strftime("%Y%m%d")
    count = 1

    # "日付_連番" のフォルダが存在しないかチェックするループ
    while True:
        # フォルダ名を生成 (例: 20250831_01)
        folder_name = f"{today_str}_{str(count).zfill(2)}"
        
        if not os.path.exists(folder_name):
            # フォルダが存在しなければ、その名前で作成してループを抜ける
            os.mkdir(folder_name)
            date = folder_name # これ以降の処理で使う変数'date'にフォルダ名を格納
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

    ret, frame = cap.read()
    if not ret:
        print("エラー: プレビュー用のフレームを取得できませんでした。")
        cap.release()
        return None, None # エラー時はNoneを返す

    cv2.imshow("Debug Preview", frame)
    print("カメラ映像が表示されます。問題なければ、エンターキーを押してください。")
    print("（このキー入力で、タイムラプス撮影が開始されます）")
    while True:
        if cv2.waitKey(1) & 0xFF == 13: # 13はEnterキーのキーコード
            break
    cv2.destroyWindow("Debug Preview")
    print("--------------------------------------------------")

    print(f"画像をフォルダ '{date}' に保存します。")
    return date,cap

    # # waiting_time秒待ってから撮影をスタートさせる
    # # capture_interval = 0.5 # 画像取得間隔（秒）
    # waiting_time = 0
    # print('Recording will be started in {0} seconds'.format(waiting_time))
    # time.sleep(waiting_time)
    # print('Start')

### 画像の撮影
def capture(date, cap, capture_interval):
    count = 1 # 撮影枚数のカウント、ファイル名に使用する。
    print("撮影を開始します。撮影を終了する場合はエンターキーを押してください")

    while True: # capture_interval秒ごとに画像の読み込みおよび保存を行う。
        ret, frame = cap.read() # カメラからキャプチャされた画像をframeとして読み込む
        cv2.imshow("camera", frame) # frameを画面に表示。なぜかこいつを残しておかないとenterで操作を止められない。
        k = cv2.waitKey(1)&0xff # キー入力を待つ。引数は入力待ち時間。
        print("撮影枚数:{0}".format(count)) # 撮影枚数の確認

        # ファイルへの保存
        path = "./{0}/".format(date) + "{0:04d}.jpg".format(count) # 連番で保存、後にリネーム
        cv2.imwrite(path, frame) # 画像をフォルダへ保存
        count += 1

        # エンターキーを押したら撮影終了
        if k == 13:
            break 
        time.sleep(capture_interval)
    print("撮影完了","撮影枚数:{0}".format(count-1))
    cap.release()
    cv2.destroyAllWindows()

### ファイル名をゼロ埋め連番にリネーム
def rename_files():
    # 撮影した画像を昇順で取得
    files = sorted(glob.glob('{0}/*.jpg'.format(date)))
    total_files = len(files)
    
    # 総枚数の桁数に合わせてゼロ埋めの桁数を決定（例: 120枚なら3桁）
    padding = len(str(total_files))
    
    print(f"{total_files}枚の画像を{padding}桁の連番にリネームします...")

    # 取得したファイルを1つずつリネーム
    for i, file_path in enumerate(files):
        # 新しいファイル名を生成 (例: ./日付/日付_001.jpg)
        new_name = "./{0}/{0}_{1}.jpg".format(date, str(i + 1).zfill(padding))
        os.rename(file_path, new_name)
    
    print("リネーム完了")

### 画像のタイムラプス化
def timelaps():
    images = sorted(glob.glob('{0}/*.jpg'.format(date)))
    print("画像の総枚数{0}".format(len(images)))

    if not images:
        print("画像がないため、動画を作成できません。")
        return

    if len(images) < 30:
        frame_rate = 2  
    else:
        frame_rate = len(images)/30

    img_for_size = cv2.imread(images[0])
    height, width, layers = img_for_size.shape

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v') # 動画のコーデックをmp4に指定

    # 動画の保存先を画像フォルダ内に指定
    video_path = f"./{date}/{date}.mp4"
    video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    print(f"動画を '{video_path}' に変換中...")
    
    for image_path in images:
        img = cv2.imread(image_path)
        video.write(img) 
    
    video.release()
    print("動画変換完了")

### キャプチャした画像の削除
def delete_captured_images():
    print(f"フォルダ '{date}' 内の元画像を削除します。")
    image_files = glob.glob(f'./{date}/*.jpg')
    for file_path in image_files:
        os.remove(file_path)
    print("画像ファイルの削除が完了しました。")

if __name__ == '__main__':
    start = time.time()
    CAMERA_NUM = 0 # カメラ番号,PC本体の場合は0を使用。
    CAPTURE_INTERVAL = 5.0 # 画像取得間隔（秒）
    OUTPUT_FILE = "/Users/sonya/Library/CloudStorage/OneDrive-HiroshimaCityUniversity/2025/UMATracker/datas"


    try:
        date, cap = setup_camera(CAMERA_NUM) # setup_cameraから変数の受け取り
        if date and cap: # date,capが正常に受け取れた場合のみ以降の動作を実行
            capture(date, cap, CAPTURE_INTERVAL)
            rename_files(date)
            timelaps(date)
            # delete_captured_images(date)
    except Exception as e:
        print(f"エラーが発生しました: {e}")

    elapsed_time = time.time() - start
    # print ("処理にかかった時間は:{0:.2f}".format(elapsed_time) + "[sec]")
