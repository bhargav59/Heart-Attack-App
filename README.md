# Heart Attack Risk Predictor — Full‑Stack

End-to-end heart attack risk prediction app with:

- Streamlit frontend
- FastAPI backend with prediction and training endpoints
- ML training pipeline to retrain on Indian datasets
- SQLite persistence for prediction logs (Postgres ready)
- Dockerized local deployment and CI tests

## Features

- 13 clinical inputs (age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall)
- Probability-based risk with clear risk levels (low/moderate/high)
- File upload to retrain model on Indian cohorts (CSV)
- API endpoints: `/health`, `/predict`, `/train`

## Project structure

```
backend/           # FastAPI service (prediction, training, logging)
ml/                # Training scripts and utilities
models/            # Saved model + scaler (artifacts)
data/              # Datasets (ignored), includes sample_indian_heart.csv
app.py             # Streamlit frontend
requirements.txt   # Python dependencies
docker-compose.yml # One command to run frontend + backend
```

## Quick start (local)

1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start backend API (in a new terminal)

```bash
uvicorn backend.main:app --reload --port 8000
```

3. Start Streamlit frontend

```bash
export BACKEND_URL=http://localhost:8000
streamlit run app.py
```

Open http://localhost:8501

## Train on an Indian dataset

Option A (UI): Use the Streamlit sidebar to upload a CSV matching the schema:

```
age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall,target
```

See `data/sample_indian_heart.csv` for an example. After upload, the app will call the backend `/train` endpoint and refresh the model.

Option B (CLI):

```bash
python -m ml.train data/your_indian_dataset.csv
```

> Note: Label orientation in legacy models was inverted (class 0 = high risk). The API accounts for this when computing risk.

### Acquiring an Indian dataset

Because fully open Indian cardiac datasets are limited, you have three approaches:

1. Direct download: If you have a URL to a CSV, run:
   ```bash
   python data/fetch_indian_dataset.py --url https://example.com/indian_heart.csv --out data/indian_heart.csv
   ```
2. Kaggle: Provide a Kaggle dataset slug (and set `KAGGLE_USERNAME`/`KAGGLE_KEY`):
   ```bash
   python data/fetch_indian_dataset.py --kaggle ankur6u/heart-disease-dataset --out data/indian_heart.csv
   ```
3. Synthetic fallback (for development only):
   ```bash
   python data/generate_indian_synthetic.py --rows 1500 --out data/indian_heart_synthetic.csv
   ```

Then train:

```bash
uvicorn backend.main:app --port 8000 &  # if not already running
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "data/indian_heart_synthetic.csv", "target_column": "target"}'
```

Review provenance & licensing before any non-demo usage. Synthetic data is NOT clinically valid.

### Web scraping (last resort)

If a site publishes a heart-related table (and permits scraping per its Terms of Service), you can try extracting it:

```bash
python data/scrape_indian_dataset.py --url https://example.com/heart_table.html --out data/indian_scraped.csv
```

Notes:

- Works best for pages with HTML tables. JavaScript-rendered tables may not load without a headless browser.
- Column names are mapped heuristically to the required schema. Review and clean the output before training.
- Respect robots.txt and licensing; don’t scrape protected content.

## Docker (recommended for quick demo)

```bash
docker compose up --build
```

Then visit http://localhost:8501 (frontend) and http://localhost:8000/health (backend).

## API examples

Predict:

```bash
curl -X POST http://localhost:8000/predict \
	-H "Content-Type: application/json" \
	-d '{
		"data": [{
			"age": 55, "sex": 1, "cp": 0, "trtbps": 140, "chol": 260,
			"fbs": 0, "restecg": 1, "thalachh": 150, "exng": 0,
			"oldpeak": 1.2, "slp": 1, "caa": 0, "thall": 2
		}]
	}'
```

Train:

```bash
curl -X POST http://localhost:8000/train \
	-H "Content-Type: application/json" \
	-d '{"dataset_path": "data/sample_indian_heart.csv", "target_column": "target"}'
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

- `BACKEND_URL` — where Streamlit calls the API
- `DATABASE_URL` — defaults to SQLite (switch to Postgres if desired)
- `CORS_ORIGINS` — allowed origins for the API

## Development notes

- Dependencies are pinned in `requirements.txt`.
- Tests run via `pytest`. See `.github/workflows/ci.yml` for CI.
- Models are stored under `models/` (ignored by git). The app falls back to root-level artifacts for backward compatibility.

## Disclaimer

This app is for educational purposes only and is not a medical device. Always consult healthcare professionals for medical advice.
