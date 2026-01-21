import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.engine import HybridEngine
from src.models.schemas import QueryParseResult, ScoringCriteria, ScoringParameters

def test_explainability_flow():
    print("=== Explainability Flow Verification (Google GenAI) ===")
    
    engine = HybridEngine()
    
    # Mock parse_query to return a result that includes a semantic_similarity criteria
    # This ensures calculate_score returns > 0 (using vector score) so items aren't filtered out.
    with patch('src.core.engine.parse_query') as mock_parse:
        mock_parse.return_value = QueryParseResult(
            original_query="fantasy",
            search_terms=["fantasy"],
            criteria=[
                ScoringCriteria(
                    name="semantic_similarity",
                    weight=1.0,
                    parameters=ScoringParameters(query_text="fantasy"),
                    description="Match semantic meaning"
                )
            ] 
        )
        
        with patch.object(engine.vs, 'search') as mock_vs_search, \
             patch.object(engine.db, 'get_item') as mock_db_get:
             
            # These scores will be multiplied by weight 1.0 -> total score > 0
            mock_vs_search.return_value = [
                {"id": 1, "score": 0.9},
                {"id": 2, "score": 0.8},
                {"id": 3, "score": 0.7},
                {"id": 4, "score": 0.6}
            ]
            
            def side_effect_get_item(eid):
                return {
                    "id": eid, 
                    "name": f"Book {eid}", 
                    "intro": f"This is the intro for Book {eid}. It is a fantasy novel.",
                    "author": "Author A",
                    "tags": ["fantasy", "magic"]
                }
            mock_db_get.side_effect = side_effect_get_item
            
            # --- MOCK GOOGLE GENAI CLIENT ---
            with patch('src.core.explainer.genai.Client') as MockGenAIClient:
                mock_client_instance = MockGenAIClient.return_value
                mock_response = MagicMock()
                mock_response.text = "Mocked Explanation: This book matches your query using Gemini."
                mock_client_instance.models.generate_content.return_value = mock_response
                
                print("Running search with query 'fantasy novels'...")
                try:
                    result = engine.search("fantasy novels", limit=5)
                except Exception as e:
                    print(f"CRITICAL ERROR during search: {e}")
                    import traceback
                    traceback.print_exc()
                    return

                print("\nResult Structure Keys:", result.keys())
                items = result.get("results", [])
                print(f"Number of results: {len(items)}")
                
                explanation_count = 0
                for i, item in enumerate(items):
                    # item structure in result is {'item': ..., 'score': ..., 'explanation': ...}
                    bk = item['item']
                    print(f"\n[Rank {i+1}] Book: {bk['name']}")
                    print(f"Score: {item['score']}")
                    explanation = item.get('explanation')
                    print(f"Explanation: {explanation}")
                    
                    if explanation:
                        explanation_count += 1
                        if "Mocked Explanation" in explanation:
                             print("   -> OK: Explanation generated via Mock.")
                    else:
                        print("   -> OK: No explanation expected (Rank > 3).")

                if explanation_count == 3:
                     print("\nSUCCESS: Exactly top 3 items have explanations.")
                else:
                     # If items < 3, then it's fine if count == len(items)
                     # But here we mocked 4 items w/ good scores, so we expect 3.
                     if len(items) >= 3 and explanation_count == 3:
                         print("\nSUCCESS: Exactly top 3 items have explanations.")
                     elif len(items) < 3 and explanation_count == len(items):
                         print("\nSUCCESS: All returned items ( < 3) have explanations.")
                     else:
                         print(f"\nFAILURE: Expected 3 explanations, got {explanation_count}. Total items: {len(items)}")

if __name__ == "__main__":
    test_explainability_flow()
