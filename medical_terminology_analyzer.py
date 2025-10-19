#!/usr/bin/env python3
"""
Medical Terminology Analyzer with Neon Vector Database
Analyzes medical terminology patterns across 1000 cases using vector embeddings
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import re
from collections import defaultdict, Counter
import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS

# Neon database integration
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import pgvector
    NEON_AVAILABLE = True
except ImportError:
    NEON_AVAILABLE = False
    print("⚠️  Neon dependencies not available. Install with: pip install psycopg2-binary pgvector")

# OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not available. Install with: pip install openai")

# Vector operations
try:
    from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MedicalTerm:
    """Medical term structure"""
    term: str
    category: str
    frequency: int
    cases: List[str]
    embedding: Optional[List[float]] = None
    synonyms: List[str] = None

@dataclass
class TerminologyAnalysis:
    """Terminology analysis results"""
    case_id: str
    total_terms: int
    unique_terms: int
    categories: Dict[str, int]
    top_terms: List[Tuple[str, int]]
    medical_score: float
    complexity_score: float

class NeonTerminologyDatabase:
    """Neon PostgreSQL database operations for medical terminology"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Connect to Neon database"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Close existing connection if any
                if self.conn and not self.conn.closed:
                    self.conn.close()
                
                self.conn = psycopg2.connect(
                    self.connection_string,
                    connect_timeout=30,
                    keepalives_idle=600,
                    keepalives_interval=30,
                    keepalives_count=3
                )
                self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
                
                # Enable pgvector extension
                self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.conn.commit()
                
                logger.info("✅ Connected to Neon database")
                return True
            except Exception as e:
                logger.warning(f"🔄 Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"❌ Failed to connect to Neon after {max_retries} attempts: {e}")
                    return False
    
    def create_tables(self):
        """Create tables for medical terminology analysis"""
        try:
            # Medical terms table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS medical_terms (
                    id SERIAL PRIMARY KEY,
                    term VARCHAR(255) NOT NULL UNIQUE,
                    category VARCHAR(100),
                    frequency INTEGER DEFAULT 0,
                    cases TEXT[],
                    embedding vector(1536),
                    synonyms TEXT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Terminology analysis table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS terminology_analysis (
                    id SERIAL PRIMARY KEY,
                    case_id VARCHAR(255) NOT NULL,
                    total_terms INTEGER,
                    unique_terms INTEGER,
                    categories JSONB,
                    top_terms JSONB,
                    medical_score FLOAT,
                    complexity_score FLOAT,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_terms_category ON medical_terms(category)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_terms_frequency ON medical_terms(frequency)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_case ON terminology_analysis(case_id)")
            
            # Create vector similarity index
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_terms_embedding ON medical_terms USING ivfflat (embedding vector_cosine_ops)")
            
            self.conn.commit()
            logger.info("✅ Created terminology analysis tables")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            return False
    
    def drop_and_recreate_tables(self):
        """Drop and recreate tables (useful for testing)"""
        try:
            # Drop tables if they exist
            self.cursor.execute("DROP TABLE IF EXISTS terminology_analysis CASCADE")
            self.cursor.execute("DROP TABLE IF EXISTS medical_terms CASCADE")
            self.conn.commit()
            
            # Recreate tables
            return self.create_tables()
        except Exception as e:
            logger.error(f"❌ Failed to drop and recreate tables: {e}")
            return False
    
    def insert_medical_term(self, term: MedicalTerm):
        """Insert a medical term with embedding"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Check if connection is still valid
                if self.conn.closed != 0:
                    logger.warning("🔄 Connection closed, reconnecting...")
                    if not self.connect():
                        return False
                
                # Start a new transaction
                self.cursor.execute("BEGIN")
                
                self.cursor.execute("""
                    INSERT INTO medical_terms (term, category, frequency, cases, embedding, synonyms)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (term) DO UPDATE SET
                        frequency = EXCLUDED.frequency,
                        cases = EXCLUDED.cases,
                        embedding = EXCLUDED.embedding,
                        synonyms = EXCLUDED.synonyms
                """, (
                    term.term,
                    term.category,
                    term.frequency,
                    term.cases,
                    term.embedding,
                    term.synonyms or []
                ))
                
                self.cursor.execute("COMMIT")
                return True
                
            except psycopg2.OperationalError as e:
                logger.warning(f"🔄 Database operational error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    try:
                        self.conn.rollback()
                        self.connect()  # Reconnect
                    except:
                        pass
                    continue
                else:
                    logger.error(f"❌ Failed to insert term {term.term} after {max_retries} attempts: {e}")
                    return False
                    
            except psycopg2.Error as e:
                logger.error(f"❌ Database error for term {term.term}: {e}")
                try:
                    self.conn.rollback()
                except:
                    pass
                return False
                
            except Exception as e:
                logger.error(f"❌ Failed to insert term {term.term}: {e}")
                try:
                    self.conn.rollback()
                except:
                    pass
                return False
        
        return False
    
    def insert_medical_terms_batch(self, terms: List[MedicalTerm], batch_size: int = 100):
        """Insert multiple medical terms in batches for better performance"""
        if not terms:
            return True
            
        try:
            # Check if connection is still valid
            if self.conn.closed != 0:
                logger.warning("🔄 Connection closed, reconnecting...")
                if not self.connect():
                    return False
            
            # Process in batches
            for i in range(0, len(terms), batch_size):
                batch = terms[i:i + batch_size]
                
                try:
                    # Start transaction for batch
                    self.cursor.execute("BEGIN")
                    
                    for term in batch:
                        self.cursor.execute("""
                            INSERT INTO medical_terms (term, category, frequency, cases, embedding, synonyms)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (term) DO UPDATE SET
                                frequency = EXCLUDED.frequency,
                                cases = EXCLUDED.cases,
                                embedding = EXCLUDED.embedding,
                                synonyms = EXCLUDED.synonyms
                        """, (
                            term.term,
                            term.category,
                            term.frequency,
                            term.cases,
                            term.embedding,
                            term.synonyms or []
                        ))
                    
                    # Commit batch
                    self.cursor.execute("COMMIT")
                    logger.info(f"✅ Inserted batch {i//batch_size + 1} ({len(batch)} terms)")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to insert batch {i//batch_size + 1}: {e}")
                    try:
                        self.cursor.execute("ROLLBACK")
                    except:
                        pass
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert terms batch: {e}")
            try:
                self.cursor.execute("ROLLBACK")
                
            except:
                pass
            return False
    
    def find_similar_terms(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Find similar medical terms using vector similarity"""
        try:
            self.cursor.execute("""
                SELECT term, category, frequency, 
                       1 - (embedding <=> %s) as similarity
                FROM medical_terms 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_embedding, query_embedding, limit))
            
            results = self.cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Failed to find similar terms: {e}")
            return []
    
    def get_terminology_stats(self) -> Dict:
        """Get terminology analysis statistics"""
        try:
            # Total terms
            self.cursor.execute("SELECT COUNT(*) as total_terms FROM medical_terms")
            total_terms = self.cursor.fetchone()['total_terms']
            
            # Category distribution
            self.cursor.execute("""
                SELECT category, COUNT(*) as count 
                FROM medical_terms 
                GROUP BY category 
                ORDER BY count DESC
            """)
            categories = {row['category']: row['count'] for row in self.cursor.fetchall()}
            
            # Most frequent terms
            self.cursor.execute("""
                SELECT term, frequency 
                FROM medical_terms 
                ORDER BY frequency DESC 
                LIMIT 20
            """)
            top_terms = [(row['term'], row['frequency']) for row in self.cursor.fetchall()]
            
            return {
                'total_terms': total_terms,
                'categories': categories,
                'top_terms': top_terms
            }
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}

class MedicalTerminologyAnalyzer:
    """Main analyzer for medical terminology"""
    
    def __init__(self, neon_connection_string: str, openai_api_key: str):
        self.neon_db = NeonTerminologyDatabase(neon_connection_string)
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if OPENAI_AVAILABLE else None
        
        # Medical terminology categories
        self.categories = {
            'cardiovascular': ['heart', 'cardiac', 'hypertension', 'blood pressure', 'chest pain', 'myocardial'],
            'respiratory': ['pneumonia', 'copd', 'asthma', 'breathing', 'lung', 'respiratory'],
            'neurological': ['stroke', 'seizure', 'headache', 'neurological', 'brain', 'cognitive'],
            'gastrointestinal': ['abdominal', 'stomach', 'liver', 'bowel', 'digestive', 'gastro'],
            'endocrine': ['diabetes', 'glucose', 'insulin', 'thyroid', 'hormone', 'endocrine'],
            'infectious': ['infection', 'fever', 'sepsis', 'bacterial', 'viral', 'antibiotic'],
            'orthopedic': ['fracture', 'bone', 'joint', 'arthritis', 'orthopedic', 'skeletal'],
            'oncology': ['cancer', 'tumor', 'oncology', 'malignancy', 'chemotherapy', 'radiation'],
            'psychiatric': ['depression', 'anxiety', 'mental', 'psychiatric', 'behavioral', 'cognitive'],
            'renal': ['kidney', 'renal', 'dialysis', 'urinary', 'nephrology', 'creatinine'],
            'hematology': ['blood', 'anemia', 'hemoglobin', 'platelet', 'coagulation', 'hematology'],
            'dermatology': ['skin', 'rash', 'dermatology', 'lesion', 'wound', 'cutaneous'],
            'ophthalmology': ['eye', 'vision', 'retinal', 'ophthalmology', 'visual', 'ocular']
        }
    
    def extract_medical_terms(self, text: str) -> List[str]:
        """Extract medical terms from text"""
        # Medical term patterns
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized terms
            r'\b(?:diabetes|hypertension|pneumonia|copd|asthma|stroke|seizure|fracture|cancer|depression|anxiety)\b',
            r'\b(?:heart|lung|brain|liver|kidney|bone|blood|skin|eye)\b',
            r'\b(?:chest pain|shortness of breath|abdominal pain|headache|fever|fatigue)\b'
        ]
        
        terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.update(matches)
        
        return list(terms)
    
    def categorize_term(self, term: str) -> str:
        """Categorize a medical term"""
        term_lower = term.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword in term_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get OpenAI embedding for text"""
        if not self.openai_client:
            return None
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Failed to get embedding: {e}")
            return None
    
    def analyze_case_terminology(self, case_id: str, case_text: str) -> TerminologyAnalysis:
        """Analyze medical terminology in a case"""
        # Extract terms
        terms = self.extract_medical_terms(case_text)
        unique_terms = list(set(terms))
        
        # Categorize terms
        categories = defaultdict(int)
        for term in unique_terms:
            category = self.categorize_term(term)
            categories[category] += 1
        
        # Calculate scores
        medical_score = len(unique_terms) / max(len(case_text.split()), 1) * 100
        complexity_score = len(unique_terms) / 10.0  # Normalize to 0-1
        
        # Top terms by frequency
        term_counts = Counter(terms)
        top_terms = term_counts.most_common(10)
        
        return TerminologyAnalysis(
            case_id=case_id,
            total_terms=len(terms),
            unique_terms=len(unique_terms),
            categories=dict(categories),
            top_terms=top_terms,
            medical_score=medical_score,
            complexity_score=complexity_score
        )
    
    def process_cases(self, cases_directory: str):
        """Process all medical cases and store in Neon"""
        if not self.neon_db.connect():
            return False
        
        if not self.neon_db.create_tables():
            return False
        
        cases_path = Path(cases_directory)
        if not cases_path.exists():
            logger.error(f"❌ Cases directory not found: {cases_directory}")
            return False
        
        case_files = list(cases_path.glob("*.txt"))
        logger.info(f"📁 Processing {len(case_files)} medical cases...")
        
        processed = 0
        all_terms = []  # Collect all terms for batch processing
        
        for case_file in case_files:
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_text = f.read()
                
                # Analyze terminology
                analysis = self.analyze_case_terminology(case_file.stem, case_text)
                
                # Extract terms for this case
                terms = self.extract_medical_terms(case_text)
                term_counts = Counter(terms)
                
                # Collect terms for batch processing
                for term, frequency in term_counts.items():
                    category = self.categorize_term(term)
                    embedding = self.get_embedding(term)
                    
                    medical_term = MedicalTerm(
                        term=term,
                        category=category,
                        frequency=frequency,
                        cases=[case_file.stem],
                        embedding=embedding
                    )
                    
                    all_terms.append(medical_term)
                
                processed += 1
                if processed % 100 == 0:
                    logger.info(f"📊 Processed {processed}/{len(case_files)} cases")
                
                # Process in batches every 50 cases to avoid memory issues
                if processed % 50 == 0 and all_terms:
                    logger.info(f"🔄 Processing batch of {len(all_terms)} terms...")
                    self.neon_db.insert_medical_terms_batch(all_terms)
                    all_terms = []  # Clear the batch
                
            except Exception as e:
                logger.error(f"❌ Failed to process {case_file}: {e}")
        
        # Process remaining terms
        if all_terms:
            logger.info(f"🔄 Processing final batch of {len(all_terms)} terms...")
            self.neon_db.insert_medical_terms_batch(all_terms)
        
        logger.info(f"✅ Processed {processed} medical cases")
        return True

# Flask web interface
app = Flask(__name__)
CORS(app)

# Global analyzer instance
analyzer = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'neon_available': NEON_AVAILABLE,
        'openai_available': OPENAI_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze_terminology', methods=['POST'])
def analyze_terminology():
    """Analyze medical terminology in text"""
    try:
        data = request.json
        text = data.get('text', '')
        case_id = data.get('case_id', 'unknown')
        
        if not analyzer:
            return jsonify({'error': 'Analyzer not initialized'}), 500
        
        analysis = analyzer.analyze_case_terminology(case_id, text)
        
        return jsonify({
            'success': True,
            'analysis': {
                'case_id': analysis.case_id,
                'total_terms': analysis.total_terms,
                'unique_terms': analysis.unique_terms,
                'categories': analysis.categories,
                'top_terms': analysis.top_terms,
                'medical_score': analysis.medical_score,
                'complexity_score': analysis.complexity_score
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/terminology_stats', methods=['GET'])
def terminology_stats():
    """Get terminology analysis statistics"""
    try:
        if not analyzer or not analyzer.neon_db.conn:
            return jsonify({'error': 'Database not connected'}), 500
        
        stats = analyzer.neon_db.get_terminology_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/similar_terms', methods=['POST'])
def similar_terms():
    """Find similar medical terms"""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not analyzer or not analyzer.openai_client:
            return jsonify({'error': 'OpenAI not available'}), 500
        
        # Get embedding for query
        embedding = analyzer.get_embedding(query)
        if not embedding:
            return jsonify({'error': 'Failed to get embedding'}), 500
        
        # Find similar terms
        similar = analyzer.neon_db.find_similar_terms(embedding, limit=10)
        
        return jsonify({'success': True, 'similar_terms': similar})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    """Main function to run the terminology analyzer"""
    global analyzer
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    neon_connection = os.getenv('NEON_CONNECTION_STRING')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not neon_connection:
        logger.error("❌ NEON_CONNECTION_STRING not found in environment")
        return
    
    if not openai_key:
        logger.error("❌ OPENAI_API_KEY not found in environment")
        return
    
    # Initialize analyzer
    analyzer = MedicalTerminologyAnalyzer(neon_connection, openai_key)
    
    # Process cases if directory exists
    cases_dir = "/Users/saiofocalallc/Enhanced_Robust_Medical_RAG_Clean_CLEAN/processed_datasets/combined_1000_cases"
    if os.path.exists(cases_dir):
        logger.info("🔄 Processing medical cases...")
        analyzer.process_cases(cases_dir)
    
    # Start web server
    port = int(os.getenv('FLASK_PORT', 5556))
    logger.info(f"🌐 Starting Medical Terminology Analyzer on port {port}")
    logger.info(f"📱 Access at: http://localhost:{port}")
    
    app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == "__main__":
    main()
