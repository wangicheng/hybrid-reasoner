import unittest
from unittest.mock import patch, MagicMock
from src.core.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    @patch("src.core.vector_store.QdrantClient")
    @patch("src.core.vector_store.SentenceTransformer")
    def test_search(self, mock_transformer_cls, mock_qdrant_cls):
        # Setup mocks
        mock_client = mock_qdrant_cls.return_value
        mock_model = mock_transformer_cls.return_value
        
        # Mock encoding
        mock_model.encode.return_value = [0.1, 0.2, 0.3] # Dummy vector
        
        # Mock search result
        mock_hit = MagicMock()
        mock_hit.id = "1"
        mock_hit.score = 0.9
        mock_hit.payload = {"name": "Test Item"}
        
        mock_client.search.return_value = [mock_hit]
        
        # Init store
        vs = VectorStore()
        
        # Run search
        results = vs.search("test query")
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "1")
        self.assertEqual(results[0]["score"], 0.9)
        mock_model.encode.assert_called_with("test query")
        mock_client.search.assert_called()

if __name__ == '__main__':
    unittest.main()
