# MediLytica Backend Documentation Index

## Overview
Complete backend integration documentation for all MediLytica features, detailing API endpoints, WebSocket protocols, database schemas, and frontend-backend interactions.

---

## Architecture Documents

### 1. [Tech Stack & Architecture](./techstack.md)
**Complete backend infrastructure specification**
- FastAPI web framework setup
- WebSocket server configuration
- Database schema (SQLite/PostgreSQL)
- API endpoint structure
- Authentication & authorization
- Deployment strategy
- Environment configuration

**Key Technologies:**
- FastAPI (REST API & WebSocket)
- PyTorch (Deep Learning)
- scikit-learn (Traditional ML)
- Pandas/NumPy (Data Processing)
- SQLAlchemy (ORM)
- Redis (Caching)

---

### 2. [Graph Visualization Strategy](./graph-visualization-strategy.md)
**Network graph visualization for ML model architectures**
- React Flow integration
- PyTorch model to graph conversion
- Real-time training visualization
- Interactive node exploration
- WebSocket streaming for updates

**API Endpoints:**
- `GET /api/viz/network/{model_id}` - Get network graph
- `WS /ws/viz/{session_id}` - Real-time updates

---

## Feature Documentation

### 3. [Hyperparameter Controls](./feature-hyperparameter-controls.md)
**Dynamic hyperparameter configuration UI**
- Parameter validation & constraints
- Real-time sync across users
- Preset configurations
- Training orchestration
- L1/L2 regularization

**API Endpoints:**
- `POST /api/train/configure` - Save config
- `GET /api/train/presets` - Get presets
- `POST /api/train/validate` - Validate params

**Frontend Integration:**
- Sliders, dropdowns, number inputs
- Default values from algorithm registry
- Real-time validation feedback

---

### 4. [Neural Network Visualization](./feature-neural-network-visualization.md)
**Interactive layer-by-layer architecture display**
- Dynamic PyTorch model building
- Layer addition/removal
- Neuron count adjustment
- Parameter counting
- Architecture export

**API Endpoints:**
- `POST /api/model/build` - Build from layers
- `GET /api/model/{id}/architecture` - Get structure
- `GET /api/model/{id}/params` - Parameter count

**Frontend Integration:**
- Layer boxes with +/- controls
- Real-time parameter count
- Architecture validation

---

### 5. [Output Visualization](./feature-output-visualization.md)
**Training output and decision boundary plots**
- Spiral dataset generation
- Decision boundary rendering
- Multiple dataset types (moons, circles, blobs)
- Real-time training updates
- Plotly integration

**API Endpoints:**
- `POST /api/datasets/generate` - Create dataset
- `GET /api/training/{session_id}/predictions` - Get predictions
- `WS /ws/training/{session_id}` - Live updates

**Visualization Types:**
- 2D scatter plots (classification)
- Decision boundaries (contour)
- Regression lines
- 3D surfaces

---

### 6. [Chat Collaboration](./feature-chat-collaboration.md)
**Real-time team messaging and collaboration**
- WebSocket-based chat rooms
- Rich text formatting (Markdown)
- Message persistence
- User presence tracking
- @mentions and notifications
- File attachments

**API Endpoints:**
- `POST /api/chat/rooms` - Create room
- `GET /api/chat/rooms/{id}/messages` - Message history
- `POST /api/chat/rooms/{id}/messages` - Send message
- `WS /ws/chat/{room_id}` - Real-time chat

**WebSocket Events:**
- `new_message` - New chat message
- `user_joined` - User entered room
- `user_left` - User exited room
- `user_typing` - Typing indicator

---

### 7. [Scrollable Data Table](./feature-data-table.md)
**Kaggle-style dataset preview and exploration**
- TanStack Table (React Table v8)
- Server-side pagination
- Sorting & filtering
- Column statistics
- Virtual scrolling for large datasets
- CSV/Parquet export

**API Endpoints:**
- `POST /api/data/upload` - Upload dataset
- `GET /api/data/preview/{id}` - Paginated preview
- `GET /api/data/statistics/{id}` - Column stats
- `GET /api/data/filter/{id}` - Filter rows
- `GET /api/data/download/{id}` - Export data

**Features:**
- 50/100/200 rows per page
- Click column headers to sort
- Search across all columns
- Null value highlighting
- Data type badges

---

### 8. [Sidebar Algorithm Navigation](./feature-sidebar-navigation.md)
**Algorithm registry and selection system**
- Hierarchical category structure
- Dynamic algorithm loading
- Hyperparameter schema management
- Algorithm instantiation
- Metadata and documentation

**API Endpoints:**
- `GET /api/algorithms` - List all algorithms
- `GET /api/algorithms/categories` - Get categories
- `GET /api/algorithms/{id}` - Algorithm details
- `GET /api/algorithms/{id}/hyperparameters` - Get params
- `POST /api/algorithms/{id}/validate-params` - Validate

**Algorithm Registry:**
- Neural Networks (5+ algorithms)
- Traditional ML (5+ algorithms)
- Ensemble Methods (5+ algorithms)
- Preprocessing (planned)
- Dimensionality Reduction (planned)

---

### 9. [Code Editor Integration](./feature-code-editor.md)
**Monaco Editor (VS Code engine) for Python code**
- Syntax highlighting
- IntelliSense autocomplete
- Code execution sandbox
- Collaborative editing
- Snippet library (PyTorch, sklearn)

**API Endpoints:**
- `POST /api/code/execute` - Run Python code
- `POST /api/code/save/{session_id}` - Save code
- `GET /api/code/load/{session_id}` - Load code
- `WS /ws/code/{session_id}` - Collaborative editing

**Security:**
- AST validation (no dangerous imports)
- Timeout protection (30s default)
- Restricted builtins
- Sandboxed execution

**Keybindings:**
- `Ctrl+Enter` - Execute code
- `Ctrl+S` - Save code

---

### 10. [Model Performance Charts](./feature-performance-charts.md)
**4-grid comprehensive metrics dashboard**
- Training/validation curves (loss & accuracy)
- Feature importance (bar chart)
- ROC curves (multi-class)
- Confusion matrix (heatmap)

**API Endpoints:**
- `GET /api/training/{session_id}/metrics` - All metrics
- `GET /api/training/{session_id}/export-metrics` - Export JSON/CSV
- `WS /ws/training/{session_id}` - Real-time updates

**Metrics Calculated:**
- Classification: accuracy, precision, recall, F1, AUC
- Regression: MSE, RMSE, MAE, R²
- Feature importance: Tree-based, linear, neural network
- Confusion matrix: All classes

**Chart Library:**
- Plotly.js for interactive visualizations
- Hover tooltips
- Zoom/pan
- Export as PNG

---

### 11. [Data Upload & Import](./feature-data-upload.md)
**Multi-format data ingestion pipeline**
- Drag-and-drop interface
- CSV, Excel, JSON, Parquet support
- Upload progress tracking
- Automatic type detection
- Data validation
- Preview with statistics

**API Endpoints:**
- `POST /api/data/upload` - Upload file
- `GET /api/data/preview/{id}` - Preview data
- `GET /api/data/statistics/{id}` - Get stats
- `POST /api/data/preprocess` - Configure preprocessing

**Validation Checks:**
- File size limit (500MB)
- Empty dataset detection
- Column name validation
- Null column detection
- Duplicate column names
- Memory usage warnings

**Type Detection:**
- Numeric (continuous)
- Categorical (low cardinality)
- Binary
- Datetime
- Text (high cardinality)
- Identifier (unique IDs)

---

### 12. [Multi-User Real-Time Synchronization](./feature-multi-user-sync.md)
**Collaborative workspace with WebSocket infrastructure**
- User presence tracking
- Workspace state synchronization
- Cursor position broadcasting
- Conflict resolution (Operational Transformation)
- Role-based permissions (Owner/Editor/Viewer)

**API Endpoints:**
- `GET /api/workspaces/{id}/users` - Active users
- `GET /api/workspaces/{id}/state` - Current state
- `POST /api/workspaces/{id}/invite` - Invite user
- `POST /api/workspaces/{id}/join` - Join via token
- `WS /ws/workspace/{id}` - Real-time sync

**WebSocket Events:**
- `workspace_state` - Initial state sync
- `user_joined` - New collaborator
- `user_left` - User disconnected
- `state_update` - Parameter/config change
- `presence_update` - View/cursor change
- `cursor_position` - Collaborative editing

**Permissions:**
- **Owner**: All permissions
- **Editor**: Edit params, train, code, upload
- **Viewer**: Read-only access

---

## Integration Flow Diagrams

### Training Workflow
```
User (Frontend)
    ↓
1. Select Algorithm (Sidebar)
    ↓
2. Configure Hyperparameters (Controls)
    ↓
3. Upload/Select Dataset (Data Upload)
    ↓
4. Review Data (Data Table)
    ↓
5. Start Training (API POST /api/train/start)
    ↓
Backend (FastAPI)
    ↓
6. Build PyTorch Model (Dynamic NN)
    ↓
7. Train with Progress Updates (WebSocket)
    ↓
8. Calculate Metrics (Metrics Engine)
    ↓
Frontend (Real-time Updates)
    ↓
9. Display Training Curves (Performance Charts)
    ↓
10. Show Decision Boundaries (Output Viz)
    ↓
11. Generate Network Graph (Graph Viz)
```

### Collaboration Workflow
```
User A                    Backend                    User B
  |                          |                          |
  |---- Connect WS --------->|<------ Connect WS -------|
  |<--- workspace_state -----|----> workspace_state --->|
  |                          |                          |
  |-- state_update: lr=0.01->|                          |
  |                          |-- broadcast: lr=0.01 --->|
  |                          |                          |
  |                          |<-- state_update: epoch=200|
  |<-- broadcast: epoch=200--|                          |
  |                          |                          |
  |-- cursor_position ------>|                          |
  |                          |-- broadcast: cursor ---->|
  |                          |                          |
```

---

## Database Schema Summary

### Core Tables

**experiments**
- id, name, description, created_at, updated_at
- algorithm_id, dataset_id, status

**training_sessions**
- id, experiment_id, hyperparameters (JSON)
- training_history (JSON), model_path
- status, started_at, completed_at

**datasets**
- id, filename, total_rows, total_columns
- dtypes (JSON), upload_date, file_path

**chat_rooms**
- id, name, experiment_id, created_at

**chat_messages**
- id, room_id, user_id, user_name
- content, content_type, formatting (JSON)
- timestamp, edited_at

**code_sessions**
- id, user_id, code (TEXT), language
- created_at, updated_at

**workspace_permissions**
- id, workspace_id, user_id, role
- granted_at, granted_by

---

## API Summary

### REST Endpoints (Total: 40+)

**Training**: 5 endpoints
**Algorithms**: 5 endpoints
**Data**: 7 endpoints
**Chat**: 4 endpoints
**Code**: 3 endpoints
**Metrics**: 2 endpoints
**Visualization**: 3 endpoints
**Workspace**: 4 endpoints
**Authentication**: 4 endpoints (planned)
**Models**: 5 endpoints

### WebSocket Endpoints (Total: 6)

- `/ws/training/{session_id}` - Training progress
- `/ws/chat/{room_id}` - Team chat
- `/ws/code/{session_id}` - Code collaboration
- `/ws/viz/{session_id}` - Visualization updates
- `/ws/workspace/{workspace_id}` - State sync
- `/ws/data/{dataset_id}` - Data streaming

---

## Tech Stack Summary

### Backend
- **Web Framework**: FastAPI 0.104+
- **ML Libraries**: PyTorch 2.0+, scikit-learn 1.3+
- **Data Processing**: Pandas 2.0+, NumPy 1.24+
- **Database**: SQLAlchemy 2.0+ (SQLite dev, PostgreSQL prod)
- **Caching**: Redis 7.0+
- **Task Queue**: Celery 5.3+ (for long training jobs)
- **Validation**: Pydantic 2.0+

### Frontend
- **Framework**: React 19.1.1
- **Build Tool**: Vite 7.1.7
- **Router**: React Router 7.9.4
- **Visualization**: Plotly.js 3.1.2, React Flow 11.0+
- **Editor**: Monaco Editor (@monaco-editor/react)
- **Tables**: TanStack Table v8
- **State**: React Context API + WebSocket

### Infrastructure
- **WebSocket**: FastAPI native WebSocket support
- **File Storage**: Local filesystem (dev), S3 (prod)
- **Authentication**: JWT tokens (planned)
- **Monitoring**: Prometheus + Grafana (planned)

---

## Development Roadmap

### Phase 1: Core Features ✅
- ✅ Hyperparameter controls
- ✅ Neural network visualization
- ✅ Output visualization
- ✅ Basic sidebar navigation

### Phase 2: Collaboration 🚧
- ✅ Chat system (documented)
- ✅ Multi-user sync (documented)
- ⏳ WebSocket implementation
- ⏳ User authentication

### Phase 3: Data Management 🚧
- ✅ Data upload (documented)
- ✅ Data table (documented)
- ⏳ Preprocessing pipeline
- ⏳ Feature engineering

### Phase 4: Advanced Features 📋
- ⏳ Code editor implementation
- ⏳ Performance charts implementation
- ⏳ Algorithm registry implementation
- ⏳ AutoML integration

### Phase 5: Production 📋
- ⏳ Authentication & authorization
- ⏳ Deployment configuration
- ⏳ Performance optimization
- ⏳ Testing suite

---

## Quick Start Guide

### 1. Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn sqlalchemy pandas numpy torch scikit-learn

# Run server
uvicorn backend.app.main:app --reload --port 8000
```

### 2. Database Setup
```bash
# Initialize database
python backend/app/db/init_db.py

# Run migrations
alembic upgrade head
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 4. Access Application
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- WebSocket Test: http://localhost:8000/ws-test

---

## Documentation Standards

Each feature document includes:

1. **Overview**: High-level description
2. **Current Implementation**: Existing frontend code
3. **Backend Integration**: API endpoints, WebSocket protocols
4. **Database Schema**: Tables and relationships
5. **Frontend Implementation**: React components with WebSocket
6. **Styling**: CSS for the feature
7. **Future Enhancements**: Planned improvements

---

## Contributing

When adding new features:

1. Create feature documentation in `backend/docs/`
2. Define API endpoints with Pydantic models
3. Implement WebSocket protocol if real-time needed
4. Add database models if persistence required
5. Update this index document
6. Add integration tests

---

## Support & Resources

- **Backend API Documentation**: `/docs` (Swagger UI)
- **WebSocket Testing Tool**: `/ws-test`
- **Database Schema**: Generated with `sqlalchemy-schemadisplay`
- **Architecture Diagrams**: See `graph-visualization-strategy.md`

---

**Last Updated**: October 25, 2025
**Version**: 1.0.0
**Status**: Documentation Complete - Implementation In Progress
