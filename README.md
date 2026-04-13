# Hybrid Reasoner

Novel search app with a CLI and a FastAPI web API.

## Requirements

- Python 3.11+
- Google Gemini API key in `.env`
- `data/all_tags.json`
- Qdrant collection `novel_tags`

## Setup

1. Create and activate a virtual environment.

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env` from the example file.

   ```bash
   Copy-Item .env.example .env
   ```

4. Set one of these variables in `.env`:

   ```ini
   GOOGLE_API_KEY=your_gemini_api_key
   # or
   GOOGLE_API_KEYS=key1,key2,key3
   ```

## Run

### CLI

```bash
python -m src.main
```

### Web API

```bash
python -m src.web_api
```

The API serves the web UI at `http://localhost:8000`.

## Data Import

1. Crawl data.

   ```bash
   python -m src.scripts.crawler_linovelib
   ```

2. Ingest crawled data into SQLite and Qdrant.

   ```bash
   python -m src.scripts.ingest_linovelib
   ```

The ingestion script reads `data/books_crawled.json`.

## SLM Training Data Generation

Generate fixed, strict JSONL datasets for multi-label tag extraction (without deleting any source records):

```bash
python -m src.scripts.generate_fixed_tag_training_data --overwrite
```

Outputs are written to `data/experiments/slm_tag_dataset/`:

- `schema_v1.json`: strict training schema
- `textbook_v1.jsonl`: stage-1 concept lessons (phi-1 style)
- `exercises_v1.jsonl`: stage-2 full exercises (intro + long thinking + final tags)
- `manifest_v1.json`: generation summary and counts

### Independent Intro-Only (Gemini)

Generate a second dataset where Gemini sees only the intro text (plus label taxonomy), and predicts tags independently:

```bash
python -m src.scripts.generate_gemini_intro_only_dataset --overwrite
```

By default, the script performs an API preflight check on startup and then immediately starts/resumes generation.
Use `--skip-preflight` only if you explicitly want to bypass this check.

Default behavior without `--overwrite`:

- Keep existing records in `exercises_v2_gemini_intro_only.jsonl`
- Append only new books that are not already generated
- Write a new timestamped manifest file if a manifest already exists
- Write each generated row immediately so interruption does not lose completed records
- Persist run checkpoint in `run_state_v2_gemini_intro_only.json` for progress visibility
- Persist per-book failures in `failed_v2_gemini_intro_only.jsonl`

For a local dry-run (no API call):

```bash
python -m src.scripts.generate_gemini_intro_only_dataset --dry-run --max-books 10 --overwrite
```

Outputs are written to `data/experiments/slm_tag_dataset_gemini_intro_only/`:

- `schema_v2_gemini_intro_only.json`
- `exercises_v2_gemini_intro_only.jsonl`
- `failed_v2_gemini_intro_only.jsonl`
- `run_state_v2_gemini_intro_only.json`
- `manifest_v2_gemini_intro_only.json`

## Notes

- If `data/all_tags.json` is missing, startup fails.
- If Qdrant does not contain `novel_tags`, startup fails.
- The CLI is interactive; there is no `--seed` or `--query` mode.
