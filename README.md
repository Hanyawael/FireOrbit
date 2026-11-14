# FireOrbit

## Layout

```
api/index.py      Vercel serverless entrypoint (wraps backend.main:app)
backend/          FastAPI app
llm/              Gemini answer generation
embedding/        index building + hybrid retrieval (local only)
chunking/, data_processed/   data used at runtime
public/           static dashboard (served by Vercel)
DATA/             raw NASA data (not deployed)
```

## Deploy on Vercel (no Docker)

1. Import the GitHub repo in Vercel (framework preset: **Other**, no build command).
2. Add the environment variable `GEMINI_API_KEY` (optional, enables written AI answers).
3. Deploy. The API is served at `/api/*` and `/health`, the dashboard at `/`.

On Vercel the app runs in keyword-search mode: `chromadb` and `sentence-transformers`
are too large for a serverless function, so they are not in `requirements.txt`.

## Local setup

```bash
pip install -r requirements-dev.txt
python embedding/build_index.py
uvicorn backend.main:app
```
