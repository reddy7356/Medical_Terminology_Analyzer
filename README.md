# Medical Terminology Analyzer with Neon Vector Database

## Overview
A specialized system for analyzing medical terminology patterns across 1000 medical cases using Neon PostgreSQL with vector embeddings.

## Features
- **Neon Vector Database**: Serverless PostgreSQL with pgvector for similarity search
- **Medical Terminology Analysis**: Extract and analyze medical terms from case files
- **Vector Embeddings**: Convert medical text to embeddings for semantic search
- **MCP Server**: Model Context Protocol server for integration
- **1000 Medical Cases**: Processed dataset from MIMIC and patient data

## Quick Start

1. **Setup Environment**:
   ```bash
   cd /Users/saiofocalallc/Medical_Terminology_Analyzer
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Neon**:
   ```bash
   cp .env.example .env
   # Edit .env with your Neon connection string
   ```

3. **Run the System**:
   ```bash
   python medical_terminology_analyzer.py
   ```

## Project Structure
```
Medical_Terminology_Analyzer/
├── medical_terminology_analyzer.py    # Main analyzer
├── neon_terminology_database.py      # Neon database operations
├── terminology_mcp_server.py         # MCP server
├── requirements.txt                  # Dependencies
├── .env.example                     # Environment template
└── data/                           # Medical cases data
    └── combined_1000_cases/
```

## API Endpoints
- `POST /api/analyze_terminology` - Analyze medical terminology
- `GET /api/terminology_stats` - Get terminology statistics
- `POST /api/similar_terms` - Find similar medical terms
- `GET /api/health` - Health check

## MCP Server
The MCP server provides programmatic access to terminology analysis:
- `terminology.analyze` - Analyze medical terminology
- `terminology.search` - Search for similar terms
- `terminology.stats` - Get analysis statistics





