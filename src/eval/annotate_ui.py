import os
import csv
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Annotation UI")

blind_csv_path = "data/experiments/pools/pilot_test_blind.csv"
annotated_csv_path = "data/experiments/pools/pilot_test_annotated.csv"

# In-memory storage of the tasks
tasks: List[Dict[str, Any]] = []

def load_tasks():
    global tasks
    tasks.clear()
    
    if not os.path.exists(blind_csv_path):
        return
        
    # Read the original blind test
    original_tasks = []
    with open(blind_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_tasks.append(row)
            
    # Read already annotated tasks (if any)
    annotated_scores = {}
    if os.path.exists(annotated_csv_path):
        with open(annotated_csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['Query ID']}_{row['Book ID']}"
                annotated_scores[key] = row.get("Score (0-3)", "")

    # Merge
    for task in original_tasks:
        key = f"{task['Query ID']}_{task['Book ID']}"
        task["Score (0-3)"] = annotated_scores.get(key, "")
        tasks.append(task)

def save_tasks():
    if not tasks:
        return
        
    with open(annotated_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tasks[0].keys())
        writer.writeheader()
        writer.writerows(tasks)

@app.on_event("startup")
def startup_event():
    load_tasks()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hybrid Reasoner - 評審盲標介面</title>
        <style>
            :root {
                --bg: #111827;
                --card-bg: #1F2937;
                --text: #F9FAFB;
                --text-muted: #9CA3AF;
                --accent: #3B82F6;
                --success: #10B981;
                --warning: #F59E0B;
                --danger: #EF4444;
                --border: #374151;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 0;
                display: flex;
                height: 100vh;
                overflow: hidden;
            }
            .sidebar {
                width: 250px;
                background-color: var(--card-bg);
                border-right: 1px solid var(--border);
                display: flex;
                flex-direction: column;
            }
            .sidebar-header {
                padding: 20px;
                border-bottom: 1px solid var(--border);
                font-weight: bold;
                font-size: 1.2rem;
            }
            .progress-container {
                padding: 20px;
                flex-grow: 1;
            }
            .progress-text {
                margin-bottom: 10px;
                font-size: 0.9rem;
                color: var(--text-muted);
            }
            .progress-bar {
                height: 8px;
                background-color: var(--border);
                border-radius: 4px;
                overflow: hidden;
            }
            .progress-fill {
                height: 100%;
                background-color: var(--success);
                width: 0%;
                transition: width 0.3s ease;
            }
            .main-content {
                flex-grow: 1;
                display: flex;
                flex-direction: column;
            }
            .topbar {
                padding: 20px 40px;
                background-color: var(--card-bg);
                border-bottom: 1px solid var(--border);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .query-box {
                font-size: 1.5rem;
                font-weight: bold;
                color: var(--accent);
            }
            .work-area {
                flex-grow: 1;
                padding: 40px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 20px;
                max-width: 900px;
                margin: 0 auto;
                width: 100%;
            }
            .book-card {
                background-color: var(--card-bg);
                border-radius: 12px;
                padding: 24px;
                border: 1px solid var(--border);
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }
            .book-title {
                font-size: 2rem;
                margin-top: 0;
                margin-bottom: 8px;
            }
            .book-meta {
                display: flex;
                gap: 15px;
                color: var(--text-muted);
                margin-bottom: 20px;
                font-size: 0.95rem;
            }
            .meta-tag {
                background-color: #374151;
                padding: 4px 10px;
                border-radius: 15px;
            }
            .book-intro {
                line-height: 1.6;
                color: #D1D5DB;
                white-space: pre-wrap;
                background-color: #1a222e;
                padding: 20px;
                border-radius: 8px;
                max-height: 400px;
                overflow-y: auto;
            }
            .action-area {
                margin-top: auto;
                background-color: var(--card-bg);
                border-top: 1px solid var(--border);
                padding: 20px 40px;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            .btn-group {
                display: flex;
                gap: 15px;
                justify-content: center;
            }
            button {
                flex: 1;
                padding: 15px;
                border: none;
                border-radius: 8px;
                font-size: 1.2rem;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.1s, opacity 0.2s;
                color: white;
            }
            button:hover { opacity: 0.9; transform: translateY(-2px); }
            button:active { transform: translateY(0); }
            
            .btn-0 { background-color: var(--danger); }
            .btn-1 { background-color: var(--warning); color: #000; }
            .btn-2 { background-color: #059669; }
            .btn-3 { background-color: var(--success); }
            .btn-nav { background-color: var(--border); color: var(--text); flex: 0.2;}
            
            #completion-msg {
                display: none;
                text-align: center;
                font-size: 2rem;
                padding: 50px;
                color: var(--success);
            }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <div class="sidebar-header">Hybrid Reasoner Eval</div>
            <div class="progress-container">
                <div class="progress-text">進度: <span id="progress-text">0 / 0</span></div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
                <div style="margin-top: 20px; font-size: 0.8rem; color: #9CA3AF;">
                    <b>評分標準：</b><br>
                    0分: 完全不相關或違反條件<br>
                    1分: 勉強相關<br>
                    2分: 部分相關<br>
                    3分: 高度相關 (極致符合)<br>
                    <br>
                    <i>提示: 可使用鍵盤 0, 1, 2, 3 快速打分。</i>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="topbar">
                <div>當前查詢條件 (Query)</div>
                <div class="query-box" id="query-text">載入中...</div>
            </div>
            
            <div id="completion-msg">
                🎉 太棒了！您已經完成所有評審標註！<br>
                <p style="font-size: 1rem; color: #9CA3AF; margin-top:20px;">請回到終端機執行 `metrics.py` 來計算實驗結果的 NDCG 吧。</p>
            </div>
            
            <div class="work-area" id="work-area">
                <div class="book-card">
                    <h2 class="book-title" id="book-title">書名</h2>
                    <div class="book-meta">
                        <span class="meta-tag" id="book-author">作者</span>
                        <span class="meta-tag" id="book-words">字數</span>
                        <span class="meta-tag" id="book-status">狀態</span>
                        <span id="book-id" style="display:none"></span>
                    </div>
                    <div class="book-intro" id="book-intro">簡介...</div>
                </div>
            </div>
            
            <div class="action-area" id="action-area">
                <div style="text-align: center; color: var(--text-muted); margin-bottom: 5px;">請為這本書符合 Query 的程度評分：</div>
                <div class="btn-group">
                    <button class="btn-nav" onclick="prevTask()">⬅️ 上一步</button>
                    <button class="btn-0" onclick="submitScore('0')">0 分<br><span style="font-size:0.8rem;font-weight:normal">不相關/違反條件</span></button>
                    <button class="btn-1" onclick="submitScore('1')">1 分<br><span style="font-size:0.8rem;font-weight:normal">勉強相關</span></button>
                    <button class="btn-2" onclick="submitScore('2')">2 分<br><span style="font-size:0.8rem;font-weight:normal">部分相關</span></button>
                    <button class="btn-3" onclick="submitScore('3')">3 分<br><span style="font-size:0.8rem;font-weight:normal">極致符合</span></button>
                </div>
            </div>
        </div>

        <script>
            let tasks = [];
            let currentIndex = 0;

            async function fetchTasks() {
                const res = await fetch('/api/tasks');
                tasks = await res.json();
                
                // 尋找第一個還沒評分的題目
                let firstUnscored = tasks.findIndex(t => t["Score (0-3)"] === "" || t["Score (0-3)"] === null);
                if(firstUnscored !== -1) {
                    currentIndex = firstUnscored;
                } else {
                    currentIndex = tasks.length > 0 ? tasks.length - 1 : 0;
                }
                
                renderCurrentTask();
            }

            function renderCurrentTask() {
                if(tasks.length === 0) {
                    document.getElementById('query-text').innerText = "找不到任務資料（請確認 pilot_test_blind.csv 存在）";
                    return;
                }
                
                // 檢查是否全完成 (或是已經跳出範圍)
                let completedCount = tasks.filter(t => t["Score (0-3)"] !== "").length;
                
                // Update Progress
                document.getElementById('progress-text').innerText = `${completedCount} / ${tasks.length}`;
                document.getElementById('progress-fill').style.width = `${(completedCount / tasks.length) * 100}%`;

                if (currentIndex >= tasks.length) {
                    document.getElementById('work-area').style.display = 'none';
                    document.getElementById('action-area').style.display = 'none';
                    document.getElementById('completion-msg').style.display = 'block';
                    document.getElementById('query-text').innerText = "評分完畢";
                    return;
                } else {
                    document.getElementById('work-area').style.display = 'flex';
                    document.getElementById('action-area').style.display = 'flex';
                    document.getElementById('completion-msg').style.display = 'none';
                }

                const task = tasks[currentIndex];
                document.getElementById('query-text').innerText = task["Query"];
                document.getElementById('book-title').innerText = task["Title"];
                document.getElementById('book-author').innerText = "✍️ " + task["Author"];
                document.getElementById('book-words').innerText = "📚 " + task["Words (萬)"] + " 萬字";
                document.getElementById('book-status').innerText = "⏳ " + task["Status"];
                document.getElementById('book-intro').innerText = task["Intro"].replace(/\[標籤:.*?\]\\n/, "");
                document.getElementById('book-id').innerText = task["Book ID"];
                
                // 視覺提示目前這題是否已打分
                const existingScore = task["Score (0-3)"];
                document.querySelectorAll('.btn-group button[class^="btn-"]').forEach(btn => {
                    btn.style.boxShadow = "none";
                    btn.style.border = "none";
                });
                if(existingScore !== "") {
                    let activeBtn = document.querySelector(`.btn-${existingScore}`);
                    if(activeBtn) {
                        activeBtn.style.boxShadow = "0 0 0 4px white";
                    }
                }
            }

            async function submitScore(score) {
                if (currentIndex >= tasks.length) return;
                
                const task = tasks[currentIndex];
                task["Score (0-3)"] = score;
                
                // Send to backend
                await fetch('/api/submit', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        query_id: task["Query ID"],
                        book_id: task["Book ID"],
                        score: score
                    })
                });

                // Next task
                currentIndex++;
                renderCurrentTask();
            }
            
            function prevTask() {
                if(currentIndex > 0) {
                    currentIndex--;
                    renderCurrentTask();
                }
            }

            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                if (['0', '1', '2', '3'].includes(e.key)) {
                    submitScore(e.key);
                } else if (e.key === 'ArrowLeft') {
                    prevTask();
                } else if (e.key === 'ArrowRight' && tasks[currentIndex]["Score (0-3)"] !== "") {
                    // 如果已打分，允許按右鍵下一題
                    currentIndex++;
                    renderCurrentTask();
                }
            });

            // Start
            fetchTasks();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/tasks")
async def get_tasks():
    return JSONResponse(content=tasks)

class ScoreSubmission(BaseModel):
    query_id: str
    book_id: str
    score: str

@app.post("/api/submit")
async def submit_score(data: ScoreSubmission):
    # Update memory
    for t in tasks:
        if t["Query ID"] == data.query_id and t["Book ID"] == data.book_id:
            t["Score (0-3)"] = data.score
            break
            
    # Save to CSV
    save_tasks()
    return {"message": "Saved"}

if __name__ == "__main__":
    print("啟動評審專用網頁介面...")
    print("請開啟瀏覽器並前往： http://127.0.0.1:8080")
    uvicorn.run(app, host="127.0.0.1", port=8080)
