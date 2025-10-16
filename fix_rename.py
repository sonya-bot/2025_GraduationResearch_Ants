import os
import glob

# --- ユーザーが設定する項目 ---
# 誤ってリネームしてしまった画像ファイルが入っているフォルダのパスを指定してください
# (例: 'D:/datas/capture_data/20251016_01' や '/Users/user/Desktop/images')
TARGET_FOLDER = "D:/capture_data/20251015_03"

def fix_incorrect_rename(folder_path):
    """
    ファイルの更新日時を元にファイルを正しい撮影順にソートし、
    改めて1から始まる連番にリネームする関数。
    """
    print(f"フォルダ '{folder_path}' 内のファイルを修正します。")

    # --- ステップ1: ファイル更新日時でソート ---
    print("ステップ1: ファイルの更新日時を取得し、撮影順に並べ替えます...")
    
    try:
        files = glob.glob(os.path.join(folder_path, '*.jpg'))
        if not files:
            print("エラー: 対象フォルダにJPGファイルが見つかりません。")
            return

        # os.path.getmtime(path)でファイルの最終更新日時を取得し、その値でソートする
        sorted_files = sorted(files, key=lambda p: os.path.getmtime(p))
        
        print(f"-> {len(sorted_files)}個のファイルを時系列順に並べ替えました。")

    except Exception as e:
        print(f"エラー: ファイルのソート中に問題が発生しました: {e}")
        return

    # --- ステップ2: ファイル名の衝突を避けるため、一時的な名前に変更 ---
    print("ステップ2: 安全に処理するため、一時的な名前に変更します...")
    temp_files = []
    try:
        for i, file_path in enumerate(sorted_files):
            # 誰とも被らないような一時ファイル名を作成
            temp_name = os.path.join(folder_path, f"__temp_{i+1:05d}.jpg")
            os.rename(file_path, temp_name)
            temp_files.append(temp_name)
        print("-> 一時ファイルへの変更が完了しました。")
    except Exception as e:
        print(f"エラー: 一時ファイルへのリネーム中に問題が発生しました: {e}")
        return

    # --- ステップ3: 1から始まる正しい連番にリネーム ---
    print("ステップ3: 1から始まる正しい連番にリネームします...")
    folder_basename = os.path.basename(folder_path)
    total_files = len(temp_files)
    padding = len(str(total_files))
    
    try:
        for i, temp_path in enumerate(temp_files):
            final_name = os.path.join(folder_path, f"{folder_basename}_{i+1:0{padding}d}.jpg")
            os.rename(temp_path, final_name)
        print(f"-> {total_files}個のファイルを正しくリネームしました。")

    except Exception as e:
        print(f"エラー: 最終的なリネーム中に問題が発生しました: {e}")
        return
            
    print("\n✅ 修正処理が正常に完了しました！")

if __name__ == '__main__':
    if not os.path.isdir(TARGET_FOLDER) or 'ここに修正したいフォルダのフルパスを記述してください' in TARGET_FOLDER:
        print(f"エラー: スクリプト内の 'TARGET_FOLDER' を正しいフォルダパスに設定してください。")
        print(f"現在の設定値: {TARGET_FOLDER}")
    else:
        # 実行前に最終確認
        answer = input(f"フォルダ '{TARGET_FOLDER}' のファイル名を修正します。本当によろしいですか？ (y/n): ").lower()
        if answer == 'y':
            fix_incorrect_rename(TARGET_FOLDER)
        else:
            print("処理を中止しました。")