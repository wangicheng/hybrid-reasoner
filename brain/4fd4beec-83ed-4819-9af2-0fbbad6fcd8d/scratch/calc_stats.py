import json
import os
import glob
from collections import defaultdict

dir_path = r'data\experiments\subset_runs\batch_20260512_115112_gemma_4_31b_it'
sizes = [100, 500, 1000, 2000, 5000]

stats = defaultdict(lambda: {"success_count": 0, "empty_count": 0, "total_queries": 0})

for size in sizes:
    pattern = os.path.join(dir_path, f'hybrid_size{size:04d}_run*.json')
    files = glob.glob(pattern)
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                stats[size]["total_queries"] += 1
                # In these files, if it's in the JSON, it's a successful run (at least it executed)
                stats[size]["success_count"] += 1 
                if not item.get("results"):
                    stats[size]["empty_count"] += 1

print("Size | Success Rate | Empty Rate")
for size in sizes:
    s = stats[size]
    sr = (s["success_count"] / s["total_queries"]) * 100 if s["total_queries"] > 0 else 0
    er = (s["empty_count"] / s["total_queries"]) * 100 if s["total_queries"] > 0 else 0
    print(f"{size:4d} | {sr:12.1f}% | {er:10.1f}%")
