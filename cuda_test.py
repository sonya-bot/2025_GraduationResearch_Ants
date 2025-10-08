import cv2
import os
import torch  # PyTorchをインポート

def recognized_camera():
    """
    システムに接続されているカメラを0番から4番までスキャンし、利用可能なカメラを表示する
    """
    print("利用可能なカメラを検索しています...")
    # /dev/video* の数に合わせて試す範囲を調整 (例: 0から4まで)
    available_cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # WindowsではCAP_DSHOWを追加すると高速化することがある
        if cap.isOpened():
            print(f"カメラ {i} は利用可能です。")
            available_cameras.append(i)
            cap.release()
    if not available_cameras:
        print("利用可能なカメラが見つかりませんでした。")
    return available_cameras

def test_camera(CAMERA_INDEX):
    """
    指定された番号のカメラを起動し、映像をリアルタイムで処理・表示する。
    PyTorchでCUDAが利用可能な場合はGPUで、そうでない場合はCPUで処理を行う。
    """
    # --- 1. カメラの初期設定 ---
    # ウェブカメラのキャプチャを開始
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"エラー: カメラ {CAMERA_INDEX} を開けませんでした。")
        return

    # 解像度を640x480に設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # フォーマットをMJPGに設定 (対応しているカメラの場合、CPU負荷が下がる)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # ウィンドウを作成し、サイズと位置を調整
    WINDOW_NAME = f'Webcam Live (Index: {CAMERA_INDEX}) - PyTorch CUDA Test'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 100, 100)
    
    # --- 2. PyTorchでCUDAが利用可能かチェックし、ユーザーが使用を選択 ---
    is_pytorch_cuda_available = torch.cuda.is_available()
    use_gpu = False  # GPUを使用するかのフラグを初期化
    device = torch.device("cpu") # デフォルトのデバイスをCPUに設定

    if is_pytorch_cuda_available:
        # CUDAが利用可能な場合、ユーザーに使用するかどうかを尋ねる
        choice = input("PyTorchでCUDAが利用可能です。GPUを使用して処理しますか？ (y/n): ").lower()
        if choice == 'y':
            use_gpu = True
            device = torch.device("cuda") # デバイスをGPUに設定
            print(f"GPU ({torch.cuda.get_device_name(0)}) を使用して映像を処理します。")
        else:
            # 'y'以外が入力された場合はCPUを使用
            print("CPUを使用して映像を処理します。")
    else:
        print("PyTorchでCUDAが利用できません。CPUを使用して映像を処理します。")

    # --- 3. 映像の読み込みと処理のループ ---
    while(cap.isOpened()):
        # カメラから1フレーム読み込む
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームを読み込めませんでした。")
            break

        # ユーザーの選択に基づいて処理を分岐
        if use_gpu:
            # 【GPU処理 (PyTorch)】
            # 1. NumPy配列(BGR, uint8)をPyTorchテンソルに変換し、GPUに転送
            tensor = torch.from_numpy(frame).to(device, dtype=torch.float32)
            
            # 2. PyTorchを使い、GPU上でグレースケール変換を実行 (BGR -> Gray)
            # OpenCVのBGRの順序に合わせて重みを適用 (R:0.299, G:0.587, B:0.114)
            gray_tensor = (tensor[:, :, 2] * 0.299 + tensor[:, :, 1] * 0.587 + tensor[:, :, 0] * 0.114)
            
            # 3. 表示のために、テンソルをuint8に変換し、CPUに戻してからNumPy配列に変換
            processed_frame = gray_tensor.to(dtype=torch.uint8).cpu().numpy()
        else:
            # 【CPU処理 (OpenCV)】
            # CPU上でグレースケール変換を実行 (こちらのほうがシンプルで高速な場合が多い)
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 処理後のフレームをウィンドウに表示
        cv2.imshow(WINDOW_NAME, processed_frame)

        # 'enter'キー(キーコード13)が押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == 13:
            break

    # --- 4. 終了処理 ---
    # キャプチャを解放し、すべてのウィンドウを閉じる
    print("カメラを解放し、ウィンドウを閉じます。")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # まず利用可能なカメラをリストアップ
    recognized_camera()
    
    # ユーザーにテストしたいカメラの番号を入力してもらう
    try:
        index_str = input("テストしたいカメラの番号を入力してください (Enterキーのみで0番を選択): ")
        if not index_str:
            CAMERA_INDEX = 0
        else:
            CAMERA_INDEX = int(index_str)
        
        test_camera(CAMERA_INDEX)

    except ValueError:
        print("無効な入力です。数値を入力してください。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

