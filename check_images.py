import os
import glob
import imagehash
from PIL import Image

def check_image_hashes(folder_path):
    """
    指定されたフォルダ内の連続する画像のハッシュ値を比較し、重複を検出する
    """
    files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    
    if len(files) < 2:
        print("画像が2枚未満のため、比較できません。")
        return

    print("--- 連続する画像のハッシュ値を比較中 ---")
    
    # 最初の画像のハッシュ値を計算
    try:
        prev_hash = imagehash.average_hash(Image.open(files[0]))
    except IOError:
        print(f"警告: {files[0]} を読み込めませんでした。スキップします。")
        return
    
    duplicates_found = 0
    
    for i in range(1, len(files)):
        try:
            current_hash = imagehash.average_hash(Image.open(files[i]))
        except IOError:
            print(f"警告: {files[i]} を読み込めませんでした。スキップします。")
            continue
            
        # 前の画像とハッシュ値が同じかチェック
        if current_hash == prev_hash:
            print(f"❌ 重複を検出しました: {os.path.basename(files[i-1])} と {os.path.basename(files[i])} は同一です。")
            duplicates_found += 1
        
        # 次の比較のためにハッシュ値を更新
        prev_hash = current_hash

    if duplicates_found == 0:
        print("✅ 全ての画像はユニークです。上書きは発生していません。")
    else:
        print(f"--- 完了：合計 {duplicates_found} 件の重複が検出されました。")

# 使用例
# check_image_hashes("/mnt/d/datas/test_data/20251007_02")
#FOLDER_PATH = "D:\\datas\\test_data\\20251010_13"
#linuxのパス形式に書き換える
FOLDER_PATH = "/mnt/d/datas/test_data/20251010_13"
check_image_hashes(FOLDER_PATH)