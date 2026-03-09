import asyncio
import json
import os
import sys
import random

# Fix stdout encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.engine import HybridEngine

async def run_blind_test_generator():
    engine = HybridEngine()
    
    queries_path = "data/experiments/queries.json"
    if not os.path.exists(queries_path):
        print(f"Error: {queries_path} not found.")
        return
        
    with open(queries_path, "r", encoding="utf-8") as f:
        queries_data = json.load(f)
        
    strategies = ["score_only", "hybrid_fusion", "original_llm_reranker_top10"]
    limit_per_query = 3  # 盲測只看前三名
    
    # 使用所有的題目進行盲測
    test_queries = queries_data
    
    output_dir = "data/experiments/results"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "blind_test_report.md")
    answer_path = os.path.join(output_dir, "blind_test_answers.json")
    
    print("\n" + "=" * 60)
    print(f"🎧 [第二階段] 盲測考卷生成 (Blind Test Generator)")
    print(f"📝 正在為 {len(test_queries)} 個魔王題生成考卷...")
    print("=" * 60 + "\n")
    
    report_lines = [
        "# 📚 混合推理引擎：盲測大會 (Blind Test)",
        "> **測試說明**：以下有幾個極具挑戰性的情境與要求。請以真實使用者的角度，閱讀每種「神秘候選人」推薦的前三本書。",
        "> 如果您覺得這組推薦 **「有打中您的心（精準/有趣/想看）」**，請在前面的 `[ ]` 打勾 `[x]`。",
        "",
        "---"
    ]
    
    answers = {}
    
    for q_data in test_queries:
        q_id = q_data["id"]
        query_text = q_data["query"]
        
        print(f"處理題目 [{q_id}]: {query_text[:30].replace(chr(10), ' ')}...")
        
        report_lines.append(f"## 題目 {q_id}")
        report_lines.append(f"**使用者的心聲：**\n> {query_text.replace(chr(10), ' ')}\n")
        
        # 打亂策略順序，真正盲測
        shuffled_strategies = strategies.copy()
        random.shuffle(shuffled_strategies)
        
        answers[q_id] = {}
        
        for idx, strategy in enumerate(shuffled_strategies, 1):
            candidate_alias = f"Candidate_{chr(64+idx)}" # A, B, C
            answers[q_id][candidate_alias] = strategy
            
            report_lines.append(f"### [ ] 候選人 {candidate_alias} (您覺得這組推薦好嗎？打勾)")
            
            try:
                response = await engine.search(
                    user_query=query_text,
                    limit=limit_per_query,
                    rerank_strategy=strategy,
                    explain=False # 關閉解釋加速
                )
                
                results = response.get("results", [])
                
                if not results:
                    report_lines.append(f"* (無推薦結果)\n")
                    continue
                    
                for rank, res in enumerate(results, 1):
                    book = res["item"]
                    title = book.get("name", "未知書名")
                    author = book.get("author", "未知作者")
                    intro = book.get("intro", "無簡介") # 顯示完整簡介
                    
                    report_lines.append(f"**{rank}.《{title}》** (作者: {author})")
                    report_lines.append(f"*簡介摘要*: {intro}...\n")
                    
            except Exception as e:
                report_lines.append(f"*(檢索失敗: {e})*\n")
                
        report_lines.append("---\n")
        
    # 寫入考卷
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    # 寫入答案本(防作弊用)
    with open(answer_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=4)
        
    print(f"\n✅ 考卷已生成！請前往查看並進行盲測：")
    print(f"👉 考卷: {report_path}")
    print(f"👉 答案: {answer_path} (測完再看！)")

if __name__ == "__main__":
    asyncio.run(run_blind_test_generator())
