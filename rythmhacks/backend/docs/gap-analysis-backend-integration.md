# Gap Analysis — Backend and Integration

This document lists backend and integration gaps inferred from the current frontend and desired features. It complements the UI/UX gap analysis document.

## Backend/API Gaps

- No service to upload/select datasets (CSV/Parquet) and return schema + sample rows.
- No training/inference endpoints for algorithms (Regression, Decision Tree, Neural Networks).
- No metrics endpoints to compute/serve: R², MAE, MSE, RMSE, ROC curve points, confusion matrix, SHAP values.
- No plotting data endpoints (e.g., decision boundary grids, tree structures, UMAP embeddings).
- No job management for long-running tasks (start job, poll status, stream logs, retrieve artifacts).
- No model registry or run tracking (parameters, metrics, artifacts).
- No authentication/authorization or multi-user workspace separation.

## Integration Gaps

- Frontend lacks typed client/SDK for backend routes; no shared schemas.
- No websocket/SSE for live progress (epochs, metrics over time).
- No storage strategy described for datasets and artifacts (local disk, cloud bucket, database).
- No error handling contract (standard error shape, retry/backoff, user-facing messages).
- No versioning of APIs or compatibility notes across pages.

## Recommended First Endpoints (MVP)

- `POST /datasets/upload` → dataset id, inferred schema, preview rows.
- `POST /models/regression/train` → job id; `GET /jobs/{id}` for status; `GET /models/{id}/metrics` for summary; `GET /models/{id}/series` for metric trends.
- `POST /models/tree/train` → returns tree structure (nodes, edges) + metrics.
- `GET /plots/decision-boundary` with model id/params → grid data for heatmap/contours.
- `GET /models/{id}/confusion-matrix` and `/roc-curve`.
- WebSocket `/ws/progress` for streaming metric updates.

## Data Contracts (sketch)

- Metrics summary:
  ```json
  {"r2": 0.91, "mae": 0.43, "mse": 0.31, "rmse": 0.56}
  ```
- Trend series:
  ```json
  {"steps": [1,2,3], "r2": [0.7,0.8,0.9], "mae": [0.9,0.6,0.4]}
  ```
- Tree structure:
  ```json
  {"nodes":[{"id":1,"label":"root","class":"A"}],"edges":[{"from":1,"to":2,"label":"x<3"}]}
  ```

## Infrastructure Notes

- Start with a lightweight FastAPI/Flask service; persist runs to SQLite; store artifacts to local `models/` directory.
- Add CORS config for the frontend dev server; rate-limit uploads; validate CSV size.

