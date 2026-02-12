import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.logic.scoring_functions import score_keyword_match

def test_fallback():
    print("Testing score_keyword_match fallback logic...")
    
    # Case 1: Primary Match (Classification matches)
    item1 = {"classification": "奇幻", "tags": [], "name": "Test Book"}
    score1 = score_keyword_match(item1, {"field": "classification", "keyword": "奇幻"})
    print(f"Case 1 (Primary Match): Expected 1.0, Got {score1}")
    assert score1 == 1.0

    # Case 2: Tag Fallback (Classification mismatch, Tag matches)
    item2 = {"classification": "Publisher", "tags": ["奇幻", "Adventure"], "name": "Test Book"}
    score2 = score_keyword_match(item2, {"field": "classification", "keyword": "奇幻"})
    print(f"Case 2 (Tag Fallback): Expected 0.8, Got {score2}")
    assert score2 == 0.8

    # Case 3: Title Fallback (Classification/Tag mismatch, Title matches)
    item3 = {"classification": "Publisher", "tags": [], "name": "奇幻旅程"}
    score3 = score_keyword_match(item3, {"field": "classification", "keyword": "奇幻"})
    print(f"Case 3 (Title Fallback): Expected 0.8, Got {score3}")
    assert score3 == 0.8
    
    # Case 4: Intro Fallback (All mismatch, Intro matches)
    item4 = {"classification": "Publisher", "tags": [], "name": "Test Book", "intro": "這是一個關於奇幻世界的冒險故事。"}
    score4 = score_keyword_match(item4, {"field": "classification", "keyword": "奇幻"})
    print(f"Case 4 (Intro Fallback): Expected 0.6, Got {score4}")
    assert score4 == 0.6

    print("All tests passed!")

if __name__ == "__main__":
    test_fallback()
