import unittest
from unittest.mock import patch, MagicMock
from src.core.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    @patch("src.core.vector_store.os.environ.get")
    @patch("src.core.vector_store.QdrantClient")
    @patch("src.core.vector_store.genai.Client")
    def test_search(self, mock_genai_cls, mock_qdrant_cls, mock_env_get):
        # Setup mocks
        mock_env_get.return_value = "dummy-api-key"
        mock_client = mock_qdrant_cls.return_value
        mock_genai_client = mock_genai_cls.return_value
        
        # Mock encoding
        mock_embed_response = MagicMock()
        mock_embed_response.embeddings = [MagicMock(values=[0.1, 0.2, 0.3])]
        mock_genai_client.models.embed_content.return_value = mock_embed_response
        
        # Mock search result
        mock_hit = MagicMock()
        mock_hit.id = "1"
        mock_hit.score = 0.9
        mock_hit.payload = {"name": "Test Item"}
        
        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_client.query_points.return_value = mock_response
        
        # Init store
        vs = VectorStore()
        
        # Run search
        results, query_vector = vs.search("test query")
        
        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "1")
        self.assertEqual(results[0]["score"], 0.9)
        mock_genai_client.models.embed_content.assert_called()
        mock_client.query_points.assert_called()

if __name__ == '__main__':
    unittest.main()
