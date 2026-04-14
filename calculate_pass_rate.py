import json

def calculate():
    with open("failed_queries_retest.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total_queries = len(data)
    passed_queries = 0
    violations = []
    
    for q in data:
        qid = q.get("id", "unknown")
        golden = q.get("golden_rules", {})
        results = q.get("results", [])
        
        req_tags = set(golden.get("required_tags", []))
        blocked_tags = set(golden.get("blocked_tags", []))
        
        # 1. 檢查 Blocked Tags (極度嚴格：Top 5 都不准有)
        found_blocked = False
        blocked_detail = ""
        for res in results:
            item_tags = set(res.get("tags", []))
            intersect = item_tags.intersection(blocked_tags)
            if intersect:
                found_blocked = True
                blocked_detail = f"命中禁止標籤 {list(intersect)} (作品: {res.get('title')})"
                break
        
        # 2. 檢查 Required Tags (Top 5 至少要有一本全中，或覆蓋所有標籤)
        # 這裡定義為：Top 5 中是否有任何作品包含了黃金標籤
        found_required = True
        if req_tags:
            # 檢查是否有任何一本書滿足了必要的標籤
            match_any = False
            for res in results:
                item_tags = set(res.get("tags", []))
                if req_tags.issubset(item_tags):
                    match_any = True
                    break
            if not match_any:
                found_required = False
        
        if not found_blocked and found_required:
            passed_queries += 1
        else:
            reason = []
            if found_blocked: reason.append(blocked_detail)
            if not found_required: reason.append(f"未召回必要標籤 {list(req_tags)}")
            violations.append(f"[{qid}] {', '.join(reason)}")

    pass_rate = (passed_queries / total_queries) * 100 if total_queries > 0 else 0
    
    print("-" * 30)
    print(f"[REPORT] Search Quality Evaluation")
    print("-" * 30)
    print(f"Total Queries: {total_queries}")
    print(f"Passed:        {passed_queries}")
    print(f"Pass Rate:     {pass_rate:.2f}%")
    print("-" * 30)
    if violations:
        print("[FAIL] Violation Details:")
        for v in violations:
            print(f"  {v}")


if __name__ == "__main__":
    calculate()
