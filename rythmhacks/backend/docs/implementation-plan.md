# Backend Implementation Plan (MVP)

Scope
- Provide minimal endpoints for upload, regression fit, decision tree, and metrics (summary + trends).
- Host a WebSocket for two-user collaboration (editor + chat) with rooms per `{route}:{datasetId}`.
- Allow mock-first development; switch via env.

Deliverables
- REST endpoints:
  - POST `/datasets/upload` → { datasetId, schema, preview, rows }
  - GET `/models/regression/fit?datasetId=` → { points, line, metrics }
  - POST `/models/tree/train` (or GET example) → { nodes, edges, metrics }
  - GET `/metrics/summary?modelId=` → { r2, mae, mse, rmse }
  - GET `/metrics/trends?modelId=` → { steps, r2, mae, mse, rmse }
- WebSocket endpoint `/ws/room/{roomId}` multiplexing `chat` and `editor` messages.
- In-memory room store with periodic persistence to disk under `models/rooms/`.

Message Protocol (WS)
- join {roomId, user} → ack {version, content, users}
- presence {cursor, selection}
- editor_change {version, range:{from,to}, text} → ack {version}
- save {version, content} → ack {version}
- chat {id, text, createdAt}
- leave {roomId}

Data Contracts (sketch)
- Regression fit: { points:[{x,y}], line:{m,b}, metrics:{ r2, mae, mse, rmse } }
- Tree: { nodes:[{id,label,class}], edges:[{from,to,label}] }
- Metrics trends: { steps:[...], r2:[...], mae:[...], mse:[...], rmse:[...] }

Mock Strategy
- If `USE_MOCKS=true`, serve static JSON from `models/mock/` directories for all endpoints.
- Provide simple generators for trends and scatter points.

Infra & Structure
- FastAPI/Starlette app layout per techstack.md; add `websockets/room.py`.
- CORS enabled for dev; file uploads limited (25MB), CSV parsed with pandas.
- Persist uploads under `datasets/{datasetId}`; sanitize filenames.

Acceptance Criteria
- Endpoints respond with the documented JSON shapes.
- WS supports 2 concurrent users editing the same buffer with presence visible.
- Server runs locally with USE_MOCKS on or off; frontend can switch via env.
