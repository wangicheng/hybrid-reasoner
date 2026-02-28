import unittest
from src.logic.scoring_functions import score_keyword_match, score_numeric_range
from src.logic.registry import ScoringRegistry

class TestScoring(unittest.TestCase):
    def test_registry(self):
        # Verify functions are registered
        self.assertIsNotNone(ScoringRegistry.get("keyword_match"))
        self.assertIsNotNone(ScoringRegistry.get("numeric_range"))

    def test_keyword_match(self):
        item = {"category": "Laptop", "desc": "Good battery"}
        
        # Exact match (case insensitive logic check)
        score = score_keyword_match(item, {"field": "category", "keyword": "laptop"})
        self.assertEqual(score[0], 1.0)
        
        # Partial match
        score = score_keyword_match(item, {"field": "desc", "keyword": "battery"})
        self.assertEqual(score[0], 1.0)
        
        # No match
        score = score_keyword_match(item, {"field": "category", "keyword": "phone"})
        self.assertEqual(score[0], 0.0)
        
        # Missing field
        score = score_keyword_match(item, {"field": "missing", "keyword": "laptop"})
        self.assertEqual(score[0], 0.0)

    def test_numeric_range(self):
        item = {"price": 1000, "weight": 2.5}
        
        # Within range
        score = score_numeric_range(item, {"field": "price", "max_val": 1500})
        self.assertEqual(score[0], 1.0)
        
        # Above max matches 0
        score = score_numeric_range(item, {"field": "price", "max_val": 900})
        self.assertEqual(score[0], 0.0)
        
        # Below min
        score = score_numeric_range(item, {"field": "price", "min_val": 1200})
        self.assertEqual(score[0], 0.0)
        
        # Range
        score = score_numeric_range(item, {"field": "price", "min_val": 800, "max_val": 1200})
        self.assertEqual(score[0], 1.0)

if __name__ == '__main__':
    unittest.main()
