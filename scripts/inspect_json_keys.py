import json
import os

def inspect_file_keys(file_path):
    """載入 JSON 檔案並遞迴地印出第一個物件的所有鍵"""
    
    def print_keys_recursively(obj, indent=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                print(f"{indent}- {key}")
                # 如果值是字典或列表，則遞迴深入
                if isinstance(value, (dict, list)):
                    print_keys_recursively(value, indent + "  ")
        elif isinstance(obj, list) and obj:
            # 如果是列表，則檢查第一個元素
            print(f"{indent}- [列表元素 0]")
            print_keys_recursively(obj[0], indent + "  ")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data or not isinstance(data, list):
            print("檔案為空或格式不正確 (不是一個 JSON 陣列)。")
            return

        first_item = data[0]
        print(f"檔案 '{os.path.basename(file_path)}' 的第一個查詢記錄的結構：")
        print_keys_recursively(first_item)

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")
    except json.JSONDecodeError:
        print(f"錯誤：解析 JSON 檔案 {file_path} 失敗。")
    except Exception as e:
        print(f"發生未知錯誤：{e}")

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")
    except json.JSONDecodeError:
        print(f"錯誤：解析 JSON 檔案 {file_path} 失敗。")
    except Exception as e:
        print(f"發生未知錯誤：{e}")

def main():
    """主函數"""
    # 使用絕對路徑以避免混淆
    base_path = 'C:/Users/USER/Desktop/code/Hybrid Reasoner/hybrid-reasoner/data/experiments/runs/batch_cl_ablation_all'
    file_name = 'gemma4_default_parser_bm25_on_cl20.json'
    file_path = os.path.join(base_path, file_name)
    
    inspect_file_keys(file_path)

if __name__ == "__main__":
    main()
