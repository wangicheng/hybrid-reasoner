import unittest
from unittest.mock import patch, MagicMock
from src.core.llm import parse_query
from src.models.schemas import QueryParseResult, ScoringCriteria

class TestParser(unittest.TestCase):
    @patch("src.core.llm.genai.GenerativeModel")
    def test_parse_query(self, mock_model_class):
        # Setup mock
        mock_model_instance = mock_model_class.return_value
        mock_response = MagicMock()
        
        # Mocking the text response to be valid JSON matching our schema
        mock_json_response = """
        {
            "original_query": "cheap laptop",
            "criteria": [
                {
                    "name": "numeric_range", 
                    "weight": 0.8, 
                    "parameters": {"field": "price", "max_val": 1000},
                    "description": "Price constraint"
                }
            ],
            "search_terms": ["laptop"]
        }
        """
        mock_response.text = mock_json_response
        mock_model_instance.generate_content.return_value = mock_response
        
        # Run function
        result = parse_query("cheap laptop")
        
        # Assertions
        self.assertIsInstance(result, QueryParseResult)
        self.assertEqual(result.original_query, "cheap laptop")
        self.assertEqual(len(result.criteria), 1)
        self.assertEqual(result.criteria[0].name, "numeric_range")
        self.assertEqual(result.criteria[0].parameters["max_val"], 1000)

if __name__ == '__main__':
    unittest.main()
