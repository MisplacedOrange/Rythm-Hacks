# Feature Documentation: Scrollable Data Table

## Overview
Kaggle-style data preview table with sorting, filtering, pagination, and efficient handling of large datasets. Enables users to explore uploaded CSV/Parquet files, view feature statistics, and select columns for ML training.

---

## Frontend Implementation

### Component Library Selection

**Recommended: TanStack Table (React Table v8)**
- Lightweight, headless UI
- Virtual scrolling for large datasets
- Built-in sorting, filtering, pagination
- TypeScript support
- Framework agnostic

**Alternative: AG Grid Community**
- Feature-rich enterprise grid
- Excel-like interface
- Advanced filtering
- Cell editing
- Higher bundle size

---

### TanStack Table Implementation

#### Installation

```bash
npm install @tanstack/react-table
```

#### Component Structure

```javascript
// frontend/src/components/DataTable.jsx
import React, { useMemo, useState, useEffect } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
} from '@tanstack/react-table'
import './DataTable.css'

export default function DataTable({ datasetId }) {
  const [data, setData] = useState([])
  const [columns, setColumns] = useState([])
  const [loading, setLoading] = useState(true)
  const [sorting, setSorting] = useState([])
  const [globalFilter, setGlobalFilter] = useState('')
  const [pagination, setPagination] = useState({
    pageIndex: 0,
    pageSize: 50,
  })
  const [totalRows, setTotalRows] = useState(0)
  const [statistics, setStatistics] = useState(null)

  // Fetch dataset preview
  useEffect(() => {
    fetchDataPreview()
    fetchStatistics()
  }, [datasetId, pagination.pageIndex, pagination.pageSize, sorting])

  const fetchDataPreview = async () => {
    setLoading(true)
    try {
      const response = await fetch(
        `http://localhost:8000/api/data/preview/${datasetId}?` +
        `page=${pagination.pageIndex}&` +
        `page_size=${pagination.pageSize}&` +
        `sort_by=${sorting[0]?.id || ''}&` +
        `sort_desc=${sorting[0]?.desc || false}`
      )
      const result = await response.json()
      
      setData(result.rows)
      setTotalRows(result.total_rows)
      
      // Generate columns from data
      if (result.columns) {
        const cols = result.columns.map(col => ({
          accessorKey: col.name,
          header: col.name,
          cell: info => formatCell(info.getValue(), col.dtype),
          meta: {
            dtype: col.dtype,
            nullable: col.nullable
          }
        }))
        setColumns(cols)
      }
    } catch (error) {
      console.error('Failed to fetch data:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchStatistics = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/data/statistics/${datasetId}`
      )
      const stats = await response.json()
      setStatistics(stats)
    } catch (error) {
      console.error('Failed to fetch statistics:', error)
    }
  }

  const formatCell = (value, dtype) => {
    if (value === null || value === undefined) {
      return <span className="null-value">NULL</span>
    }
    
    if (dtype === 'float64') {
      return typeof value === 'number' ? value.toFixed(4) : value
    }
    
    if (dtype === 'datetime64') {
      return new Date(value).toLocaleString()
    }
    
    return value
  }

  const table = useReactTable({
    data,
    columns,
    pageCount: Math.ceil(totalRows / pagination.pageSize),
    state: {
      sorting,
      globalFilter,
      pagination,
    },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    manualPagination: true, // Server-side pagination
    manualSorting: true,    // Server-side sorting
  })

  return (
    <div className="data-table-container">
      {/* Header with stats */}
      <div className="table-header">
        <h2>Dataset Preview</h2>
        <div className="dataset-stats">
          <span><strong>Rows:</strong> {totalRows.toLocaleString()}</span>
          <span><strong>Columns:</strong> {columns.length}</span>
          <span><strong>Memory:</strong> {formatBytes(statistics?.memory_usage)}</span>
        </div>
      </div>

      {/* Search bar */}
      <div className="table-controls">
        <input
          type="text"
          value={globalFilter ?? ''}
          onChange={e => setGlobalFilter(e.target.value)}
          placeholder="Search all columns..."
          className="search-input"
        />
        
        <button onClick={() => downloadDataset(datasetId)}>
          Download CSV
        </button>
      </div>

      {/* Column statistics panel */}
      {statistics && (
        <ColumnStatistics 
          stats={statistics.column_stats}
          onSelectColumn={(col) => console.log('Selected:', col)}
        />
      )}

      {/* Table */}
      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            {table.getHeaderGroups().map(headerGroup => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map(header => (
                  <th 
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    className={header.column.getIsSorted() ? 'sorted' : ''}
                  >
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext()
                    )}
                    <span className="sort-indicator">
                      {{
                        asc: ' ↑',
                        desc: ' ↓',
                      }[header.column.getIsSorted()] ?? ''}
                    </span>
                    <div className="column-meta">
                      {header.column.columnDef.meta?.dtype}
                    </div>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={columns.length} className="loading-cell">
                  Loading data...
                </td>
              </tr>
            ) : (
              table.getRowModel().rows.map(row => (
                <tr key={row.id}>
                  {row.getVisibleCells().map(cell => (
                    <td key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination controls */}
      <div className="pagination-controls">
        <button
          onClick={() => table.setPageIndex(0)}
          disabled={!table.getCanPreviousPage()}
        >
          {'<<'}
        </button>
        <button
          onClick={() => table.previousPage()}
          disabled={!table.getCanPreviousPage()}
        >
          {'<'}
        </button>
        
        <span className="page-info">
          Page {pagination.pageIndex + 1} of {table.getPageCount()}
          {' | '}
          Showing {pagination.pageIndex * pagination.pageSize + 1}-
          {Math.min((pagination.pageIndex + 1) * pagination.pageSize, totalRows)}
          {' of '} {totalRows.toLocaleString()} rows
        </span>
        
        <button
          onClick={() => table.nextPage()}
          disabled={!table.getCanNextPage()}
        >
          {'>'}
        </button>
        <button
          onClick={() => table.setPageIndex(table.getPageCount() - 1)}
          disabled={!table.getCanNextPage()}
        >
          {'>>'}
        </button>
        
        <select
          value={pagination.pageSize}
          onChange={e => table.setPageSize(Number(e.target.value))}
          className="page-size-select"
        >
          {[25, 50, 100, 200].map(pageSize => (
            <option key={pageSize} value={pageSize}>
              Show {pageSize}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}

// Column statistics component
function ColumnStatistics({ stats, onSelectColumn }) {
  return (
    <div className="column-stats-panel">
      <h3>Column Statistics</h3>
      <div className="stats-grid">
        {Object.entries(stats).map(([colName, colStats]) => (
          <div key={colName} className="stat-card" onClick={() => onSelectColumn(colName)}>
            <h4>{colName}</h4>
            <div className="stat-details">
              <span>Type: {colStats.dtype}</span>
              <span>Non-null: {colStats.non_null_count}</span>
              <span>Null: {colStats.null_count}</span>
              {colStats.unique_count && (
                <span>Unique: {colStats.unique_count}</span>
              )}
              {colStats.mean !== undefined && (
                <>
                  <span>Mean: {colStats.mean.toFixed(2)}</span>
                  <span>Std: {colStats.std.toFixed(2)}</span>
                  <span>Min: {colStats.min}</span>
                  <span>Max: {colStats.max}</span>
                </>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
```

---

## Backend Implementation

### API Endpoints

```python
# backend/app/api/routes/data.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json

router = APIRouter()

# In-memory cache for uploaded datasets (use Redis in production)
datasets_cache: Dict[str, pd.DataFrame] = {}

class DatasetMetadata(BaseModel):
    id: str
    filename: str
    total_rows: int
    total_columns: int
    column_names: List[str]
    dtypes: Dict[str, str]
    memory_usage: int
    upload_date: str

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    nullable: bool

class DataPreviewResponse(BaseModel):
    rows: List[Dict[str, Any]]
    columns: List[ColumnInfo]
    total_rows: int
    page: int
    page_size: int

@router.post("/api/data/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload CSV/Parquet dataset
    
    Returns dataset ID for subsequent operations
    """
    try:
        # Read file based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(file.file)
        else:
            raise HTTPException(400, "Unsupported file format. Use CSV or Parquet")
        
        # Generate unique ID
        dataset_id = generate_unique_id()
        
        # Cache dataset
        datasets_cache[dataset_id] = df
        
        # Save metadata
        metadata = DatasetMetadata(
            id=dataset_id,
            filename=file.filename,
            total_rows=len(df),
            total_columns=len(df.columns),
            column_names=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            memory_usage=df.memory_usage(deep=True).sum(),
            upload_date=datetime.utcnow().isoformat()
        )
        
        # Optionally persist to disk for large datasets
        save_path = Path(f"data/uploads/{dataset_id}.parquet")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        
        return {
            'dataset_id': dataset_id,
            'metadata': metadata.dict()
        }
    
    except Exception as e:
        raise HTTPException(500, f"Failed to upload dataset: {str(e)}")

@router.get("/api/data/preview/{dataset_id}", response_model=DataPreviewResponse)
async def get_data_preview(
    dataset_id: str,
    page: int = Query(0, ge=0),
    page_size: int = Query(50, ge=1, le=1000),
    sort_by: Optional[str] = None,
    sort_desc: bool = False
):
    """
    Get paginated dataset preview
    
    Args:
        dataset_id: Dataset identifier
        page: Page number (0-indexed)
        page_size: Rows per page
        sort_by: Column to sort by
        sort_desc: Sort descending if True
    """
    # Load from cache or disk
    if dataset_id not in datasets_cache:
        parquet_path = Path(f"data/uploads/{dataset_id}.parquet")
        if not parquet_path.exists():
            raise HTTPException(404, "Dataset not found")
        datasets_cache[dataset_id] = pd.read_parquet(parquet_path)
    
    df = datasets_cache[dataset_id]
    
    # Apply sorting
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=not sort_desc)
    
    # Pagination
    start_idx = page * page_size
    end_idx = start_idx + page_size
    page_df = df.iloc[start_idx:end_idx]
    
    # Convert to records
    rows = page_df.replace({np.nan: None}).to_dict('records')
    
    # Column info
    columns = [
        ColumnInfo(
            name=col,
            dtype=str(df[col].dtype),
            nullable=df[col].isnull().any()
        )
        for col in df.columns
    ]
    
    return DataPreviewResponse(
        rows=rows,
        columns=columns,
        total_rows=len(df),
        page=page,
        page_size=page_size
    )

@router.get("/api/data/statistics/{dataset_id}")
async def get_dataset_statistics(dataset_id: str):
    """
    Get comprehensive dataset statistics
    
    Returns:
        - Column-wise statistics (mean, std, min, max for numeric)
        - Null counts
        - Unique value counts
        - Data type distribution
        - Memory usage
    """
    if dataset_id not in datasets_cache:
        parquet_path = Path(f"data/uploads/{dataset_id}.parquet")
        if not parquet_path.exists():
            raise HTTPException(404, "Dataset not found")
        datasets_cache[dataset_id] = pd.read_parquet(parquet_path)
    
    df = datasets_cache[dataset_id]
    
    # Overall statistics
    overall_stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'duplicate_rows': int(df.duplicated().sum()),
    }
    
    # Column statistics
    column_stats = {}
    for col in df.columns:
        col_data = df[col]
        
        base_stats = {
            'dtype': str(col_data.dtype),
            'non_null_count': int(col_data.notna().sum()),
            'null_count': int(col_data.isnull().sum()),
            'null_percentage': float(col_data.isnull().mean() * 100),
            'unique_count': int(col_data.nunique()),
        }
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            base_stats.update({
                'mean': float(col_data.mean()) if not col_data.isna().all() else None,
                'std': float(col_data.std()) if not col_data.isna().all() else None,
                'min': float(col_data.min()) if not col_data.isna().all() else None,
                'max': float(col_data.max()) if not col_data.isna().all() else None,
                'q25': float(col_data.quantile(0.25)) if not col_data.isna().all() else None,
                'q50': float(col_data.quantile(0.50)) if not col_data.isna().all() else None,
                'q75': float(col_data.quantile(0.75)) if not col_data.isna().all() else None,
            })
        
        # Categorical columns
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
            top_values = col_data.value_counts().head(10)
            base_stats.update({
                'top_values': {str(k): int(v) for k, v in top_values.items()},
                'most_common': str(col_data.mode()[0]) if len(col_data.mode()) > 0 else None,
            })
        
        column_stats[col] = base_stats
    
    return {
        'overall': overall_stats,
        'column_stats': column_stats
    }

@router.get("/api/data/filter/{dataset_id}")
async def filter_dataset(
    dataset_id: str,
    column: str,
    operator: str = Query(..., regex="^(eq|ne|gt|lt|ge|le|contains|in)$"),
    value: str = None,
    values: List[str] = Query(None)
):
    """
    Filter dataset by column conditions
    
    Operators:
        - eq: equals
        - ne: not equals
        - gt: greater than
        - lt: less than
        - ge: greater or equal
        - le: less or equal
        - contains: string contains
        - in: value in list
    """
    if dataset_id not in datasets_cache:
        raise HTTPException(404, "Dataset not found")
    
    df = datasets_cache[dataset_id]
    
    if column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not found")
    
    # Apply filter
    if operator == 'eq':
        filtered_df = df[df[column] == value]
    elif operator == 'ne':
        filtered_df = df[df[column] != value]
    elif operator == 'gt':
        filtered_df = df[df[column] > float(value)]
    elif operator == 'lt':
        filtered_df = df[df[column] < float(value)]
    elif operator == 'ge':
        filtered_df = df[df[column] >= float(value)]
    elif operator == 'le':
        filtered_df = df[df[column] <= float(value)]
    elif operator == 'contains':
        filtered_df = df[df[column].str.contains(value, na=False)]
    elif operator == 'in':
        filtered_df = df[df[column].isin(values)]
    else:
        raise HTTPException(400, "Invalid operator")
    
    return {
        'total_rows': len(filtered_df),
        'filtered_rows': filtered_df.replace({np.nan: None}).to_dict('records')
    }

@router.get("/api/data/download/{dataset_id}")
async def download_dataset(dataset_id: str, format: str = 'csv'):
    """Download dataset in specified format"""
    from fastapi.responses import StreamingResponse
    import io
    
    if dataset_id not in datasets_cache:
        raise HTTPException(404, "Dataset not found")
    
    df = datasets_cache[dataset_id]
    
    if format == 'csv':
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=dataset_{dataset_id}.csv"}
        )
    elif format == 'parquet':
        output = io.BytesIO()
        df.to_parquet(output, index=False)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=dataset_{dataset_id}.parquet"}
        )
    else:
        raise HTTPException(400, "Unsupported format")
```

---

## Advanced Features

### Virtual Scrolling for Large Datasets

```javascript
// Use react-window for efficient rendering
import { FixedSizeList as List } from 'react-window'

function VirtualizedTable({ data, columns }) {
  const Row = ({ index, style }) => (
    <div style={style} className="table-row">
      {columns.map(col => (
        <div key={col.id} className="table-cell">
          {data[index][col.id]}
        </div>
      ))}
    </div>
  )

  return (
    <List
      height={600}
      itemCount={data.length}
      itemSize={35}
      width="100%"
    >
      {Row}
    </List>
  )
}
```

### Column Selection for ML Training

```javascript
const [selectedColumns, setSelectedColumns] = useState([])

const toggleColumnSelection = (columnName) => {
  setSelectedColumns(prev => 
    prev.includes(columnName)
      ? prev.filter(c => c !== columnName)
      : [...prev, columnName]
  )
}

// Send to training endpoint
const startTraining = async () => {
  await fetch('http://localhost:8000/api/train/start', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: datasetId,
      feature_columns: selectedColumns.filter(c => c !== targetColumn),
      target_column: targetColumn
    })
  })
}
```

---

## Styling (Kaggle-inspired)

```css
/* frontend/src/components/DataTable.css */
.data-table-container {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  overflow: hidden;
}

.table-header {
  padding: 16px 20px;
  border-bottom: 2px solid #e0e0e0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.dataset-stats {
  display: flex;
  gap: 24px;
  font-size: 14px;
  color: #666;
}

.table-controls {
  padding: 12px 20px;
  display: flex;
  gap: 12px;
  border-bottom: 1px solid #e0e0e0;
}

.search-input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.table-wrapper {
  flex: 1;
  overflow: auto;
  -ms-overflow-style: none;
  scrollbar-width: none;
}

.table-wrapper::-webkit-scrollbar {
  display: none;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.data-table thead {
  position: sticky;
  top: 0;
  background: #f8f9fa;
  z-index: 10;
}

.data-table th {
  padding: 12px 16px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #dee2e6;
  cursor: pointer;
  user-select: none;
}

.data-table th:hover {
  background: #e9ecef;
}

.data-table th.sorted {
  background: #e3f2fd;
  color: #1976d2;
}

.column-meta {
  font-size: 11px;
  color: #999;
  font-weight: normal;
  margin-top: 2px;
}

.data-table td {
  padding: 10px 16px;
  border-bottom: 1px solid #f0f0f0;
}

.data-table tbody tr:hover {
  background: #f8f9fa;
}

.null-value {
  color: #999;
  font-style: italic;
}

.pagination-controls {
  padding: 12px 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  border-top: 1px solid #e0e0e0;
  background: #f8f9fa;
}

.pagination-controls button {
  padding: 6px 12px;
  border: 1px solid #ddd;
  background: white;
  border-radius: 4px;
  cursor: pointer;
}

.pagination-controls button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.pagination-controls button:hover:not(:disabled) {
  background: #e9ecef;
}

.page-info {
  font-size: 13px;
  color: #666;
}

.column-stats-panel {
  padding: 16px;
  background: #f8f9fa;
  border-bottom: 1px solid #e0e0e0;
  max-height: 200px;
  overflow-y: auto;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
  margin-top: 8px;
}

.stat-card {
  background: white;
  padding: 12px;
  border-radius: 6px;
  border: 1px solid #e0e0e0;
  cursor: pointer;
  transition: all 0.2s;
}

.stat-card:hover {
  border-color: #4682B4;
  box-shadow: 0 2px 8px rgba(70, 130, 180, 0.15);
}

.stat-card h4 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #333;
}

.stat-details {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: #666;
}
```

---

## Performance Optimization

### Server-side Processing

```python
# Use chunked reading for very large files
@router.get("/api/data/preview-chunked/{dataset_id}")
async def get_chunked_preview(dataset_id: str, chunk_size: int = 10000):
    """Stream data in chunks for very large datasets"""
    parquet_path = Path(f"data/uploads/{dataset_id}.parquet")
    
    async def generate_chunks():
        for chunk in pd.read_parquet(parquet_path, chunksize=chunk_size):
            yield chunk.to_json(orient='records')
    
    return StreamingResponse(generate_chunks(), media_type="application/json")
```

### Caching Strategy

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def get_cached_statistics(dataset_id: str, cache_key: str):
    """Cache expensive statistics calculations"""
    df = datasets_cache[dataset_id]
    return calculate_statistics(df)
```

---

## Future Enhancements

1. **Column Operations**: Rename, delete, reorder columns
2. **Data Cleaning**: Handle missing values, outliers
3. **Feature Engineering**: Create derived columns, transformations
4. **Export Selection**: Export filtered/selected rows
5. **Visualization**: Inline histograms, correlation heatmaps
6. **SQL Interface**: Query data with SQL
7. **Collaboration**: Shared views, annotations
8. **Version Control**: Track dataset changes
9. **Excel Integration**: Import/export Excel files
10. **AutoML Integration**: Auto-detect feature types, suggest models
## MVP Additions (Missing)

- Bind to dataset preview from upload response.
- Pagination or virtualized rows for >1k rows preview.
- Column types detection badges; basic sorting by column.
- Copy CSV snippet and download sample actions.
