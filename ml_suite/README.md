# ML Suite

Upload any dataset, pick X and y, and get a tuned, compared, reusable model —
with a visual report and a prediction API.

Runs entirely on your machine. No accounts, no cloud, no credentials.

## Start it

```bash
cd ml_suite
./run.sh
```

Then open **http://127.0.0.1:8501**.

| Command | What it does |
|---|---|
| `./run.sh` | API + dashboard |
| `./run.sh api` | API only — http://127.0.0.1:8000/docs |
| `./run.sh ui` | Dashboard only |
| `./run.sh stop` | Stop both |

Logs go to `logs/api.log` and `logs/ui.log`.

## First run

1. **Data Source** → *File upload* → **Load customers_classification.csv**
2. **Configure** → target `churned` → **Start training**
3. Watch the leaderboard fill in, then open **Report**
4. **Predict** → type a row or upload a file

## What it does

**Ingestion** — CSV, TSV, Excel, JSON, Parquet; any SQLAlchemy database
(Neon/Postgres, MySQL, SQLite); MongoDB collections (nested documents are
flattened). Column names are de-duplicated, numeric and date columns are
recovered from strings, and empty or constant columns are dropped.

**Profiling** — every column is classified as numeric, categorical,
high-cardinality, datetime, boolean, or identifier, and the UI shows exactly
how each will be treated before you commit.

**Preprocessing** — built as a `ColumnTransformer` and fitted *inside* the
model pipeline:

| Column kind | Treatment |
|---|---|
| numeric | median impute → standardise |
| categorical (≤ 20 distinct) | mode impute → one-hot (unknown categories ignored) |
| high cardinality (> 20) | mode impute → ordinal encode → scale |
| datetime | expand to year/month/day/weekday/hour/epoch |
| identifier | dropped — no signal |

**Task detection** — classification vs regression is inferred from the target
and can be overridden.

**Training** — every suitable model is tuned with `RandomizedSearchCV` over
cross-validation, then scored on a held-out test split it never saw:

- *Classification*: LogisticRegression, SVC, DecisionTree, RandomForest,
  GradientBoosting, KNeighbors, GaussianNB, XGBoost
- *Regression*: Linear, Ridge, Lasso, ElasticNet, SVR, DecisionTree,
  RandomForest, GradientBoosting, KNeighbors, XGBoost

A model that fails is recorded and the comparison continues.

**Reporting** — ranked leaderboard, per-metric comparison, confusion matrix or
predicted-vs-actual and residual plots, feature importance, and the winning
hyperparameters.

**Reuse** — the winner is saved as one pipeline containing its own imputers,
scalers, and encoders. Prediction takes raw rows in the original schema; no
preprocessing is repeated by hand, which is what keeps predictions consistent
with training.

## Use a saved model from code

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/api/predict/mdl_xxxxxxxx",
    json={"rows": [{"age": 31, "income": None, "plan": "basic", ...}]},
)
print(response.json()["predictions"])
```

Missing values, unseen categories, and column order are all handled.

## Layout

```
backend/
  core/          config, logging, metrics, exceptions
  ingestion/     file / SQL / MongoDB loaders + sanitiser
  preprocessing/ profiler, ColumnTransformer builder, datetime transformer
  training/      task detection, model zoo, trainer, jobs, evaluation, predict
  registry/      dataset / run / model persistence
  api/routes/    datasets, training, models, predict, health
  main.py        app, middleware, error handlers
frontend/
  app.py         home
  pages/         1 Data Source · 2 Configure · 3 Train · 4 Report · 5 Predict
  lib/           API client, chart builders
storage/         datasets, models, runs   (created at runtime)
```

## Operations

| Endpoint | Purpose |
|---|---|
| `/health` | liveness |
| `/ready` | readiness + storage check |
| `/metrics` | Prometheus text format |
| `/system` | limits, available models, counters |
| `/docs` | interactive API reference |

Every request gets an `X-Request-ID` that appears in both logs. Console logs are
human-readable; `logs/mlsuite.log` is one JSON object per line.

## Configuration

All optional, all environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MLSUITE_PORT` | 8000 | API port |
| `MLSUITE_UI_PORT` | 8501 | Dashboard port |
| `MLSUITE_MAX_UPLOAD_MB` | 200 | Upload size cap |
| `MLSUITE_MAX_TRAIN_ROWS` | 200000 | Rows above this are sampled |
| `MLSUITE_SEARCH_ITER` | 12 | Hyperparameter combinations per model |
| `MLSUITE_CV_FOLDS` | 3 | Cross-validation folds |
| `MLSUITE_TRAIN_WORKERS` | 2 | Concurrent training runs |
| `MLSUITE_LOG_JSON` | 0 | Set to 1 for JSON console logs |

## Known limits

- Training runs in-process on a thread pool. It is sized for one workstation,
  not for many concurrent users.
- Storage is the local filesystem. The `backend/registry/store.py` interface is
  the only thing that needs reimplementing to move to MongoDB or Postgres.
- CORS is open and there is no authentication — this is a localhost tool. Both
  need closing before it is exposed to a network.
- Very wide datasets (> 500 features) are rejected by default; one-hot encoding
  a high-cardinality column is the usual cause.
