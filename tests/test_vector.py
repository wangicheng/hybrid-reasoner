import unittest
from unittest.mock import patch, MagicMock
from src.core.vector_store import VectorStore

class TestVectorStore(unittest.TestCase):
    @patch("src.core.vector_store.os.environ.get")
    @patch("src.core.vector_store.QdrantClient")
    @patch("src.core.vector_store.genai.Client")
    def test_search_content_vector(self, mock_genai_cls, mock_qdrant_cls, mock_env_get):
        """Test search using the content named vector."""
        mock_env_get.return_value = "dummy-api-key"
        mock_client = mock_qdrant_cls.return_value
        mock_genai_client = mock_genai_cls.return_value
        
        # Mock embedding
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
        
        vs = VectorStore()
        
        # Search with default (content vector)
        results, query_vector = vs.search("test query")
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "1")
        self.assertEqual(results[0]["score"], 0.9)
        
        # Verify query_points was called with using="content"
        call_kwargs = mock_client.query_points.call_args
        self.assertEqual(call_kwargs.kwargs.get("using") or call_kwargs[1].get("using"), "content")

    @patch("src.core.vector_store.os.environ.get")
    @patch("src.core.vector_store.QdrantClient")
    @patch("src.core.vector_store.genai.Client")
    def test_search_tags_vector(self, mock_genai_cls, mock_qdrant_cls, mock_env_get):
        """Test search using the tags named vector."""
        mock_env_get.return_value = "dummy-api-key"
        mock_client = mock_qdrant_cls.return_value
        mock_genai_client = mock_genai_cls.return_value
        
        mock_embed_response = MagicMock()
        mock_embed_response.embeddings = [MagicMock(values=[0.4, 0.5, 0.6])]
        mock_genai_client.models.embed_content.return_value = mock_embed_response
        
        mock_hit = MagicMock()
        mock_hit.id = "2"
        mock_hit.score = 0.85
        mock_hit.payload = {"name": "Tag Match Item", "tags": ["奇幻", "冒險"]}
        
        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_client.query_points.return_value = mock_response
        
        vs = VectorStore()
        
        # Search with tags vector
        results, query_vector = vs.search(
            "奇幻",
            vector_name="tags",
            task_type="SEMANTIC_SIMILARITY"
        )
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "2")
        self.assertEqual(results[0]["score"], 0.85)
        
        # Verify query_points was called with using="tags"
        call_kwargs = mock_client.query_points.call_args
        self.assertEqual(call_kwargs.kwargs.get("using") or call_kwargs[1].get("using"), "tags")

    @patch("src.core.vector_store.os.environ.get")
    @patch("src.core.vector_store.QdrantClient")
    @patch("src.core.vector_store.genai.Client")
    def test_build_texts(self, mock_genai_cls, mock_qdrant_cls, mock_env_get):
        """Test the text builders for content and tags."""
        mock_env_get.return_value = "dummy-api-key"
        
        item = {
            "name": "鬥破蒼穹",
            "intro": "三十年河東三十年河西",
            "classification": "玄幻",
            "tags": ["熱血", "升級流", "鬥氣"],
        }
        
        content_text = VectorStore._build_content_text(item)
        self.assertIn("鬥破蒼穹", content_text)
        self.assertIn("三十年河東", content_text)
        # Content should NOT contain tags
        self.assertNotIn("熱血", content_text)
        
        tags_text = VectorStore._build_tags_text(item)
        self.assertIn("玄幻", tags_text)
        self.assertIn("熱血", tags_text)
        self.assertIn("升級流", tags_text)
        # Tags should NOT contain intro
        self.assertNotIn("三十年河東", tags_text)

if __name__ == '__main__':
    unittest.main()
