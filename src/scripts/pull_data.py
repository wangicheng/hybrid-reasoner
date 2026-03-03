import os
import shutil
import subprocess
import tempfile
from pathlib import Path

def run_cmd(cmd, cwd=None, ignore_errors=False):
    print(f"執行指令: {cmd}")
    result = subprocess.run(cmd, cwd=cwd, shell=True, text=True, capture_output=True)
    if result.returncode != 0 and not ignore_errors:
        print(f"錯誤 - 指令失敗: {cmd}")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def pull_data_from_orphan_branch():
    # 設定路徑
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    data_dir = project_root / 'data'
    
    # 取得遠端 URL
    res = run_cmd("git config --get remote.origin.url", cwd=project_root)
    remote_url = res.stdout.strip()
    
    if not remote_url:
        print("錯誤: 無法取得 remote.origin.url。")
        return

    # 使用暫存目錄，以免污染本地儲存庫
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"建立暫存工作目錄: {temp_dir}")
        
        # 檢查遠端是否存在 data 分支
        print("檢查遠端 data 分支是否存在...")
        res = run_cmd(f"git ls-remote --heads {remote_url} data", cwd=temp_dir)
        branch_exists = bool(res.stdout.strip())
        
        if not branch_exists:
            print("錯誤: 遠端不存在 data 分支，無法拉取資料。")
            return
            
        print("遠端存在 data 分支，正在 Clone 下載資料...")
        run_cmd(f"git clone --branch data --single-branch {remote_url} .", cwd=temp_dir)
        
        print("確保 Git LFS 的大檔案已經完整下載...")
        run_cmd("git lfs pull", cwd=temp_dir, ignore_errors=True)
            
        print(f"正在將最新資料同步複製到本地 {data_dir}...")
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            
        for item in temp_dir.iterdir():
            # 略過 git 專屬設定，不覆蓋本地版本控制環境
            if item.name in ['.git', '.gitattributes']:
                continue
            
            dest = data_dir / item.name
            if item.is_dir():
                # 若為資料夾則進行合併/覆蓋
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
                
        print("成功！已經從遠端的 data 分支把資料拉回來放到本地目前的 data 目錄裡了。")

if __name__ == '__main__':
    try:
        pull_data_from_orphan_branch()
    except Exception as e:
        print(f"執行失敗: {e}")
