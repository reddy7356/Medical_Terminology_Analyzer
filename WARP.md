# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Medical Terminology Analyzer with Neon PostgreSQL + pgvector for analyzing medical terminology patterns across 1000 medical cases using vector embeddings.

## Setup Guide

### Prerequisites

1. **Python 3.8+** (recommended: 3.12+)
2. **Neon PostgreSQL account** (optional but recommended)
3. **OpenAI API key** (optional - system works with fallback embeddings)

### Step-by-Step Setup

#### 1. Clone and Install Dependencies

```bash
# Navigate to the project
cd /Users/saiofocalallc/Medical_Terminology_Analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Or use the automated setup script:
```bash
./setup.sh
```

#### 2. Configure Neon PostgreSQL (Recommended)

**Get your Neon connection string:**

1. Sign up for free at [neon.tech](https://neon.tech)
2. Create a new project in the Neon console
3. Navigate to **Connection Details** in your project dashboard
4. Copy the connection string (format: `postgresql://user:password@host/database?sslmode=require`)
5. Enable pgvector extension by running in Neon SQL Editor:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

**Note:** Without Neon, the system runs in analysis-only mode (no database persistence).

#### 3. Configure OpenAI API (Optional)

**Get your OpenAI API key:**

1. Sign up at [platform.openai.com](https://platform.openai.com/signup)
2. Navigate to [API Keys](https://platform.openai.com/api-keys)
3. Click **Create new secret key** and copy it (starts with `sk-`)
4. The system uses `text-embedding-3-small` model (~$0.02 per 1M tokens)

**Note:** Without OpenAI, the system uses deterministic SHA-256 embeddings automatically.

#### 4. Create Environment File

```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your credentials
# Add your NEON_CONNECTION_STRING (if using database)
# Add your OPENAI_API_KEY (if using OpenAI embeddings)
```

The `env.example` file contains detailed comments explaining each variable.

#### 5. Verify Installation

```bash
# Run the main analyzer
python medical_terminology_analyzer.py

# In another terminal, check the health endpoint
curl http://localhost:5556/api/health
```

You should see a JSON response with system status.

## Essential Commands

### Running the System

**Main analyzer with Flask API:**
```bash
python medical_terminology_analyzer.py
# Access at: http://localhost:5556
```

**MCP Server (Model Context Protocol):**
```bash
python terminology_mcp_server.py
```

**Cardio-Respiratory Classifier:**
```bash
python cardio_respiratory_classifier.py
```

**Analysis Scripts:**
```bash
# Analyze classification results
python analyze_results.py

# Comprehensive efficiency analysis
python efficiency_analysis.py

# Compact analysis
python compact_analysis.py

# Simple analysis
python simple_analysis.py
```

### API Endpoints

The Flask API (port 5556) provides:
- `GET /api/health` - Health check and system status
- `POST /api/analyze_terminology` - Analyze medical terminology in text
- `GET /api/terminology_stats` - Get terminology statistics from database
- `POST /api/similar_terms` - Find similar medical terms via vector search

### MCP Server Tools

The MCP server exposes these tools for programmatic access:
- `terminology_analyze` - Analyze medical terminology in text
- `terminology_search` - Search for similar medical terms using embeddings
- `terminology_stats` - Get analysis statistics
- `terminology_categories` - Get medical terminology categories
- `terminology_embedding` - Get vector embedding for medical text

## Architecture

### System Overview

**3-Tier Architecture:**
1. **Neon PostgreSQL with pgvector** - Serverless PostgreSQL with vector similarity search
2. **Python Analyzers** - Medical terminology extraction and classification
3. **Flask API / MCP Server** - HTTP endpoints and Model Context Protocol interface

### Database Schema

**Tables:**
- `medical_terms` - Stores medical terms with vector embeddings
  - `term` (VARCHAR) - Medical term
  - `category` (VARCHAR) - One of 13 medical categories
  - `frequency` (INTEGER) - Occurrence count
  - `cases` (TEXT[]) - Array of case IDs
  - `embedding` (vector(1536)) - OpenAI embedding for semantic search
  - `synonyms` (TEXT[]) - Related terms
  - Indexed with `ivfflat` for vector similarity search using cosine distance

- `terminology_analysis` - Case-level analysis results
  - `case_id` - Case identifier
  - `total_terms`, `unique_terms` - Term counts
  - `categories` (JSONB) - Category distribution
  - `top_terms` (JSONB) - Most frequent terms
  - `medical_score`, `complexity_score` - Computed metrics

### Medical Terminology Categories

The system categorizes terms into 13 medical specialties:
1. cardiovascular
2. respiratory
3. neurological
4. gastrointestinal
5. endocrine
6. infectious
7. orthopedic
8. oncology
9. psychiatric
10. renal
11. hematology
12. dermatology
13. ophthalmology

### Embedding Strategy

**Dual-mode embeddings** with graceful degradation:
- **Primary:** OpenAI `text-embedding-3-small` (1536 dimensions) via API
- **Fallback:** Deterministic SHA-256-based embeddings when OpenAI unavailable
  - Uses normalized hash vectors to maintain cosine similarity semantics
  - System remains fully functional without OpenAI API key

### Case Processing Workflow

1. Read medical cases from `data/` directory (1000 cases from MIMIC dataset)
2. Extract medical terms using regex patterns
3. Categorize terms into 13 specialties
4. Generate embeddings (OpenAI or fallback)
5. Batch insert to Neon database (50 terms per batch)
6. Calculate analysis metrics (medical_score, complexity_score)
7. Store analysis results

### ML Classification Pipeline

**Cardio-Respiratory Classifier** (`cardio_respiratory_classifier.py`):
- Classifies cases as "cardio_respiratory_only" vs "mixed_or_other"
- Uses scikit-learn models: Logistic Regression, Random Forest, SVM, Naive Bayes
- TF-IDF vectorization of case text
- Outputs to `cardio_resp_results/`:
  - `classification_results.json` - Per-case classifications
  - `performance_report.json` - Model performance metrics
  - `cardio_respiratory_cases.json` - Filtered cardio/resp cases

## Environment Configuration

### Required Variables

Create `.env` file with:

```bash
# Neon PostgreSQL Connection (required for database features)
NEON_CONNECTION_STRING=postgresql://username:password@your-neon-host/dbname?sslmode=require

# OpenAI API Key (optional - uses fallback embeddings if missing)
OPENAI_API_KEY=your-openai-api-key-here

# Flask Configuration
FLASK_HOST=localhost
FLASK_PORT=5556
DEBUG=True

# Vector Database Settings
VECTOR_DIMENSION=1536
SIMILARITY_THRESHOLD=0.7
MAX_RESULTS=100

# Medical Terminology Settings
MIN_TERM_FREQUENCY=2
MAX_TERMS_PER_CASE=50
```

### Graceful Degradation

The system handles missing configurations:
- **No NEON_CONNECTION_STRING:** Runs in analysis-only mode (no database persistence)
- **No OPENAI_API_KEY:** Uses deterministic SHA-256 embeddings automatically
- **Port conflict:** Automatically finds available port if 5556 is occupied

## Key Implementation Details

### Database Connection

- **Connection pooling** with keepalive (600s idle, 30s interval, 3 count)
- **Retry logic** with exponential backoff (3 attempts, 2^n second delays)
- **Health checks** monitor connection state and reconnect if closed
- Uses `psycopg2` with `RealDictCursor` for dict-based results

### Batch Processing

- Processes medical terms in **batches of 50** to balance memory and performance
- Intermediate batches written every 50 cases during processing
- Embeddings cached per term to minimize API calls

### Vector Similarity Search

- Uses pgvector's **cosine distance operator** (`<=>`)
- Query: `SELECT ... ORDER BY embedding <=> query_vector LIMIT n`
- Returns similarity score as `1 - distance`
- IVFFlat index on embeddings for fast approximate nearest neighbor search

### Medical Term Extraction

Uses regex patterns to identify:
- Capitalized medical terms
- Known medical conditions (diabetes, hypertension, pneumonia, etc.)
- Anatomical terms (heart, lung, brain, etc.)
- Symptom phrases (chest pain, shortness of breath, etc.)

### Web Server

- Flask app with CORS enabled
- Runs on `0.0.0.0` to accept external connections
- Automatic port selection if preferred port unavailable
- Multiprocessing uses 'spawn' method to avoid semaphore leaks on macOS

## Data Location

Medical cases stored in `data/` directory:
- 1000 case files: `case_0001_mimic.txt` through `case_1000_mimic.txt`
- Cases sourced from MIMIC (Medical Information Mart for Intensive Care) dataset

## Troubleshooting

**Database connection fails:**
- Check NEON_CONNECTION_STRING is valid (not placeholder)
- Verify DNS resolution to Neon host
- System continues in analysis-only mode if database unavailable

**OpenAI embedding errors:**
- System automatically falls back to deterministic embeddings
- Check api.openai.com is reachable
- Verify OPENAI_API_KEY is valid

**Port already in use:**
- System automatically finds next available port
- Check logs for actual port used
