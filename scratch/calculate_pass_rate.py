import json
import os

def normalize_tags(tags):
    if not tags: return []
    return [t.strip().lower() for t in tags]

def tag_matches(rule_tag, book_tag):
    return rule_tag.lower() in book_tag.lower()

def check_rules(golden_rules, book):
    violations = []
    
    # Status
    req_status = golden_rules.get("required_status")
    if req_status:
        status = str(book.get("publish_status", "")).lower()
        if req_status == "completed":
            if status not in ["completed", "completed_status", "finished", "done", "已完結", "完結"]:
                violations.append(f"Status mismatch: expected completed, got {status}")
        elif req_status == "ongoing":
            if status not in ["ongoing", "in_progress", "running", "連載中", "連載"]:
                violations.append(f"Status mismatch: expected ongoing, got {status}")

    # Required tags
    req_tags = golden_rules.get("required_tags") or []
    book_tags = normalize_tags(book.get("tags", []))
    for rt in req_tags:
        if not any(tag_matches(rt, bt) for bt in book_tags):
            violations.append(f"Missing required tag: {rt}")

    # Blocked tags
    blocked_tags = golden_rules.get("blocked_tags") or []
    for bt in blocked_tags:
        if any(tag_matches(bt, b_tag) for b_tag in book_tags):
            violations.append(f"Contains blocked tag: {bt}")

    return violations

def main():
    with open("data/experiments/queries.json", "r", encoding="utf-8") as f:
        queries_config = {q["id"]: q for q in json.load(f)}
    
    with open("data/books_crawled.json", "r", encoding="utf-8") as f:
        books_db = {}
        for b in json.load(f):
            key = (b.get("name"), b.get("author"))
            books_db[key] = b

    with open("bm25_evaluation_results.json", "r", encoding="utf-8") as f:
        results_data = json.load(f)

    total_queries = 0
    full_pass_queries = 0 # All top 5 are valid
    partial_pass_queries = 0 # At least one in top 5 is valid
    
    for query_run in results_data:
        q_id = query_run.get("query_id")
        q_conf = queries_config.get(q_id)
        if not q_conf: 
            continue
            
        total_queries += 1
        golden_rules = q_conf.get("golden_rules", {})
        results = query_run.get("results", [])
        
        valid_count = 0
        for res in results:
            key = (res.get("name"), res.get("author"))
            book_meta = books_db.get(key)
            if book_meta and not check_rules(golden_rules, book_meta):
                valid_count += 1
        
        if valid_count == len(results) and len(results) > 0:
            full_pass_queries += 1
        if valid_count > 0:
            partial_pass_queries += 1

    print(f"Total Queries: {total_queries}")
    print(f"Full Pass (All Top-5 valid): {full_pass_queries} ({full_pass_queries/total_queries:.2%})")
    print(f"Success@5 (At least one valid): {partial_pass_queries} ({partial_pass_queries/total_queries:.2%})")

if __name__ == "__main__":
    main()
