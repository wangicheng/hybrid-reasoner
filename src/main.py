"""Hybrid Reasoner - Main Search Interface."""

import asyncio
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

from src.core.api_utils import get_api_key_rotator
from src.core.engine import HybridEngine


def print_separator(char: str = "=", length: int = 60) -> None:
    print(char * length)


def display_results(result: dict) -> None:
    """Display search results in a readable format."""
    print_separator()
    print(f"Query: {result['query']}")
    print_separator("-")

    print("\nParsed Criteria:")
    if result.get("parsed_criteria"):
        for i, criteria in enumerate(result["parsed_criteria"], 1):
            name = criteria.get("name", "N/A")
            weight = criteria.get("weight", 0)
            params = criteria.get("parameters", {})
            params_clean = {k: v for k, v in params.items() if v is not None}
            print(f"  {i}. {name} (weight: {weight:.2f})")
            if params_clean:
                print(f"     params: {params_clean}")
    else:
        print("  (no parsed criteria)")

    print_separator("-")
    print(f"\nResults ({len(result.get('results', []))}):\n")

    if not result.get("results"):
        print("  No results found.")
        return

    for i, res in enumerate(result["results"], 1):
        item = res["item"]
        score = res["score"]
        explanation = res.get("explanation")

        print(f"[{i}] {item.get('name', 'N/A')}")
        print(f"  Author: {item.get('author', 'N/A')}")
        print(f"  Classification: {item.get('classification', 'N/A')}")
        print(f"  Tags: {', '.join(item.get('tags', [])) if item.get('tags') else 'N/A'}")
        print(f"  Score: {score:.4f}")

        if explanation:
            print("\n  Explanation:")
            for line in explanation.split("\n"):
                print(f"     {line}")
        print()


async def run_search(engine: HybridEngine, query: str, limit: int = 5) -> None:
    result = await engine.search(query, limit=limit)
    display_results(result)


def main() -> None:
    print_separator("=")
    print("Hybrid Reasoner")
    print("Type 'q', 'exit', or 'quit' to leave.")
    print_separator("=")

    try:
        api_keys = get_api_key_rotator().api_keys
    except Exception as exc:
        print(f"Error: no Google API key found. {exc}")
        print("Please set GOOGLE_API_KEY or GOOGLE_API_KEYS in your .env file.")
        return

    print(f"Google API keys found: {len(api_keys)}")
    print("\nInitializing engine...")
    engine = HybridEngine()
    print("Engine ready.\n")

    while True:
        try:
            query = input("Query: ").strip()
            if not query:
                continue
            if query.lower() in {"q", "exit", "quit"}:
                print("\nBye.")
                break

            print(f"\nSearching for: {query}\n")
            asyncio.run(run_search(engine, query, limit=5))

        except KeyboardInterrupt:
            print("\n\nBye.")
            break
        except Exception as exc:
            print(f"\nError: {exc}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
