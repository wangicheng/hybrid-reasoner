"""
Verification script for search improvement fixes:
1. Negation trap fix in scoring_functions.py
2. Minimum semantic threshold in engine.py (logic check only)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.logic.scoring_functions import score_keyword_match

print("=" * 60)
print("Test 1: Negation Trap Fix")
print("=" * 60)

# Mock item: intro says "This is NOT an online game novel"
mock_item_negation = {
    "id": 1,
    "name": "Test Book A",
    "classification": "Fantasy",
    "tags": ["Adventure"],
    "intro": "\u9019\u4e0d\u662f\u7db2\u904a\u6587\u3002\u4e3b\u89d2\u7a7f\u8d8a\u5230\u7570\u4e16\u754c\uff0c\u5c55\u958b\u5192\u96aa\u65c5\u7a0b\u3002"
}

score, reason = score_keyword_match(mock_item_negation, {"field": "classification", "keyword": "\u7db2\u904a"})
print(f"  Score: {score}, Reason: {reason}")
assert score == 0.0, f"FAIL: Expected 0.0 but got {score}"
print("  [PASS] Negation correctly detected, score is 0.0")

# Mock item: intro says positive mention
mock_item_positive = {
    "id": 2,
    "name": "Test Book B",
    "classification": "Urban",
    "tags": ["Romance"],
    "intro": "\u9019\u662f\u4e00\u6b3e\u7db2\u904a\u6539\u7de8\u7684\u6545\u4e8b\uff0c\u4e3b\u89d2\u5728\u865b\u64ec\u4e16\u754c\u4e2d\u63a2\u7d22\u3002"
}

score2, reason2 = score_keyword_match(mock_item_positive, {"field": "classification", "keyword": "\u7db2\u904a"})
print(f"  Score: {score2}, Reason: {reason2}")
assert score2 == 0.6, f"FAIL: Expected 0.6 but got {score2}"
print("  [PASS] Positive mention correctly scored 0.6")

# Edge case: negation with different wording
mock_item_negation2 = {
    "id": 3,
    "name": "Test Book C",
    "classification": "Adventure",
    "tags": [],
    "intro": "\u672c\u4f5c\u4e26\u975e\u7db2\u904a\u985e\u578b\uff0c\u800c\u662f\u7d14\u6b63\u7684\u7570\u4e16\u754c\u5192\u96aa\u3002"
}

score3, reason3 = score_keyword_match(mock_item_negation2, {"field": "classification", "keyword": "\u7db2\u904a"})
print(f"  Score: {score3}, Reason: {reason3}")
assert score3 == 0.0, f"FAIL: Expected 0.0 but got {score3}"
print("  [PASS] Negation with different wording correctly detected")

print()
print("=" * 60)
print("Test 2: Minimum Semantic Threshold Logic Check")
print("=" * 60)

# Simulate the filtering logic from engine.py
mock_scored_items = [
    {"item": {"name": "Book A"}, "score": 0.76, "vector_score": 0.28},
    {"item": {"name": "Book B"}, "score": 0.62, "vector_score": 0.55},
    {"item": {"name": "Book C"}, "score": 0.50, "vector_score": 0.65},
]

filtered = [r for r in mock_scored_items if r['vector_score'] > 0.6]
print(f"  Before filter: {len(mock_scored_items)} items")
print(f"  After filter (vector_score > 0.6): {len(filtered)} items")
assert len(filtered) == 1, f"FAIL: Expected 1 item but got {len(filtered)}"
assert filtered[0]["item"]["name"] == "Book C"
print("  [PASS] Only high-vector items survive the threshold")

# Edge case: all filtered out
mock_all_low = [
    {"item": {"name": "Book X"}, "score": 0.5, "vector_score": 0.3},
    {"item": {"name": "Book Y"}, "score": 0.4, "vector_score": 0.2},
]
filtered_empty = [r for r in mock_all_low if r['vector_score'] > 0.6]
assert len(filtered_empty) == 0
print("  [PASS] All-low-vector case correctly returns empty")

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
