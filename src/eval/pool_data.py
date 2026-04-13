import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_QUERIES_PATH = Path("data/experiments/queries.json")


def load_queries(queries_path: Path = DEFAULT_QUERIES_PATH) -> List[Dict[str, Any]]:
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    with queries_path.open("r", encoding="utf-8") as f:
        queries = json.load(f)

    if not isinstance(queries, list):
        raise ValueError(f"Queries file must contain a list: {queries_path}")

    return queries


def _looks_like_run_data(data: Any) -> bool:
    if not isinstance(data, list):
        return False

    for item in data:
        if not isinstance(item, dict):
            return False
        if "query_id" not in item or "results" not in item:
            return False

    return True


def load_runs(experiment_dir: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    base_dir = Path(experiment_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Missing experiment directory: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Experiment path is not a directory: {base_dir}")

    runs: Dict[str, List[Dict[str, Any]]] = {}
    for file_path in sorted(base_dir.glob("*.json")):
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if _looks_like_run_data(data):
            runs[file_path.stem] = data

    if not runs:
        raise FileNotFoundError(f"No run JSON files found in {base_dir}")

    return runs


def build_pooled_queries(
    queries_config: List[Dict[str, Any]],
    runs: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    pooled_queries: List[Dict[str, Any]] = []

    for query_conf in queries_config:
        query_id = str(query_conf["id"])
        query_text = str(query_conf["query"])
        candidates: Dict[str, Dict[str, Any]] = {}

        for engine_name, engine_run in runs.items():
            query_run = next(
                (item for item in engine_run if str(item.get("query_id")) == query_id),
                None,
            )
            if not query_run:
                continue

            for result in query_run.get("results", []):
                book_id = str(result.get("book_id", "")).strip()
                if not book_id:
                    continue

                candidate = candidates.setdefault(
                    book_id,
                    {
                        "book_id": book_id,
                        "title": result.get("title", ""),
                        "author": result.get("author", ""),
                        "intro": result.get("intro", ""),
                        "words_total": result.get("words_total", 0),
                        "publish_status": result.get("publish_status", ""),
                        "tags": result.get("tags", []),
                        "source_engines": [],
                        "original_ranks": {},
                    },
                )

                if engine_name not in candidate["source_engines"]:
                    candidate["source_engines"].append(engine_name)
                    candidate["original_ranks"][engine_name] = result.get("rank")

        pooled_queries.append(
            {
                "query_id": query_id,
                "query": query_text,
                "golden_rules": query_conf.get("golden_rules", {}),
                "candidates": list(candidates.values()),
            }
        )

    return pooled_queries


def load_experiment_pool(
    experiment_dir: str | Path,
    queries_path: Path = DEFAULT_QUERIES_PATH,
) -> List[Dict[str, Any]]:
    queries = load_queries(queries_path)
    runs = load_runs(experiment_dir)
    return build_pooled_queries(queries, runs)


def build_annotation_rows(pooled_queries: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for pooled_query in pooled_queries:
        query_id = pooled_query["query_id"]
        query_text = pooled_query["query"]

        for candidate in pooled_query["candidates"]:
            tags = candidate.get("tags", [])
            if isinstance(tags, str):
                tags_text = tags
            else:
                tags_text = ", ".join(str(tag) for tag in tags if str(tag).strip())

            words_total = candidate.get("words_total") or 0
            words_in_10k = round(float(words_total) / 10000, 1) if words_total else 0

            rows.append(
                {
                    "Query ID": str(query_id),
                    "Query": str(query_text),
                    "Book ID": str(candidate.get("book_id", "")),
                    "Title": str(candidate.get("title", "")),
                    "Author": str(candidate.get("author", "")),
                    "Words (萬)": str(words_in_10k),
                    "Status": str(candidate.get("publish_status", "")),
                    "Tags": tags_text,
                    "Intro": str(candidate.get("intro", "")),
                    "Score (0-3)": "",
                    "Comment": "",
                }
            )

    return rows
