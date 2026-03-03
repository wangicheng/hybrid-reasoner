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

def push_data_to_orphan_branch():
    # 設定路徑
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    data_dir = project_root / 'data'
    
    if not data_dir.exists():
        print(f"錯誤: 找不到 data 目錄 ({data_dir})")
        return

    # 取得遠端 URL
    res = run_cmd("git config --get remote.origin.url", cwd=project_root)
    remote_url = res.stdout.strip()
    
    if not remote_url:
        print("錯誤: 無法取得 remote.origin.url。")
        return

    # 使用暫存目錄來建立孤兒分支的操作環境，不會汙染主要的專案目錄
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"建立暫存工作目錄: {temp_dir}")
        
        # 檢查遠端是否存在 data 分支
        print("檢查遠端 data 分支是否存在...")
        res = run_cmd(f"git ls-remote --heads {remote_url} data", cwd=temp_dir)
        branch_exists = bool(res.stdout.strip())
        
        if branch_exists:
            print("遠端已有 data 分支，正在 Clone...")
            run_cmd(f"git clone --branch data --single-branch {remote_url} .", cwd=temp_dir)
            
            # 清空現有文件 (保留 .git)，以便處理刪除的檔案
            for item in temp_dir.iterdir():
                if item.name != '.git':
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        else:
            print("遠端不存在 data 分支，初始化新的 Repository...")
            run_cmd("git init", cwd=temp_dir)
            run_cmd("git checkout --orphan data", cwd=temp_dir)

        # 設定 Git LFS，避免超過 100MB 限制
        print("設定 Git LFS...")
        run_cmd("git lfs install", cwd=temp_dir, ignore_errors=True)
        # 直接寫入 .gitattributes 避免 cmd 執行 track 時出錯
        with open(temp_dir / ".gitattributes", "w") as f:
            f.write("*.sqlite filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
            f.write("*.pt filter=lfs diff=lfs merge=lfs -text\n")
            
        print(f"複製資料從 {data_dir} 到暫存工作目錄...")
        for item in data_dir.iterdir():
            if item.name == '.git':
                continue
            dest = temp_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
                
        # 檢查是否有變更
        res = run_cmd("git status --porcelain", cwd=temp_dir)
        if not res.stdout.strip():
            print("資料無任何變更，不需要 commit。")
            return
            
        print("Commit 變更...")
        run_cmd("git add -A", cwd=temp_dir)
        run_cmd('git commit -m "Auto-update data"', cwd=temp_dir)
        
        print("Push 至遠端...")
        if branch_exists:
            run_cmd("git push origin data", cwd=temp_dir)
        else:
            run_cmd(f"git push {remote_url} data", cwd=temp_dir)
            
        print("成功！已將 data 資料 push 到 data 分支。")

if __name__ == '__main__':
    try:
        push_data_to_orphan_branch()
    except Exception as e:
        print(f"執行失敗: {e}")
