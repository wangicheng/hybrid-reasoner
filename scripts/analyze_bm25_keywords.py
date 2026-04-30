import json
import re
from collections import Counter, defaultdict

path = r"c:\dev\hybrid-reasoner\data\experiments\runs\batch_20260429_005905\gemma4_default_parser_bm25_on.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

token_counter = Counter()
per_query = {}

for entry in data:
    qid = entry.get("query_id")
    texts = []
    for crit in entry.get("parsed_criteria", []):
        if crit.get("name") == "semantic_similarity":
            qt = crit.get("parameters", {}).get("query_text")
            if qt:
                texts.append(qt)
    combined = " ".join(texts)
    # split on whitespace and punctuation
    tokens = [t.strip() for t in re.split(r"[\s,;。、，\n]+", combined) if t.strip()]
    per_query[qid] = tokens
    token_counter.update(tokens)

# print per-query tokens (first line only)
print("Per-query extracted keywords (first semantic query_text tokens):")
for qid, toks in per_query.items():
    print(f"{qid}: {' | '.join(toks[:12])}")

print('\nTop 40 tokens overall:')
for tok, cnt in token_counter.most_common(40):
    print(f"{tok}: {cnt}")

# identify duplicated tokens within queries
print('\nQueries with duplicated tokens (showing token and count):')
for qid, toks in per_query.items():
    c = Counter(toks)
    dups = {t:n for t,n in c.items() if n>1}
    if dups:
        print(f"{qid}: {dups}")
