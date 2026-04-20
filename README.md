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

## Gemini Tag Description A/B

Use this experiment to compare Gemini tag extraction quality with and without tag definitions.

1. Prepare tag descriptions in `data/tag_descriptions.json`.
2. Run one command:

   ```bash
   python scripts/run_gemini_tag_description_ab.py --model-id gemini-2.5-flash-lite
   ```

3. Check outputs in `data/experiments/tag_description_ab/`:

   - `gemini_no_tag_desc_report.json`
   - `gemini_with_tag_desc_report.json`
   - `gemini_tag_desc_comparison.json`
   - `summary.md`

To enable tag definitions in production parse flow, set:

```ini
GEMINI_TAG_DESCRIPTION_ENABLED=true
TAG_DESCRIPTIONS_PATH=data/tag_descriptions.json
```

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

## Notes

- If `data/all_tags.json` is missing, startup fails.
- If Qdrant does not contain `novel_tags`, startup fails.
- The CLI is interactive; there is no `--seed` or `--query` mode.
