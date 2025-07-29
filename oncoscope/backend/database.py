"""
Database module for OncoScope
"""
import aiosqlite
from datetime import datetime
from typing import List, Dict, Optional
import json
import os
from pathlib import Path

from .config import settings

# Database path
DB_PATH = Path(settings.database_url.replace("sqlite:///", ""))

async def init_db():
    """Initialize the database with required tables"""
    DB_PATH.parent.mkdir(exist_ok=True, parents=True)
    
    async with aiosqlite.connect(str(DB_PATH)) as db:
        # Create analysis history table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT UNIQUE NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                mutations TEXT NOT NULL,
                overall_risk_score REAL,
                risk_classification TEXT,
                recommendations TEXT,
                full_results TEXT,
                confidence_score REAL,
                user_session TEXT
            )
        """)
        
        # Create mutations cache table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS mutation_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mutation_id TEXT UNIQUE NOT NULL,
                gene TEXT NOT NULL,
                variant TEXT NOT NULL,
                analysis_data TEXT NOT NULL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create user sessions table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                analysis_count INTEGER DEFAULT 0
            )
        """)
        
        await db.commit()

async def save_analysis(analysis_data: Dict):
    """Save analysis results to database"""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        try:
            mutations_str = json.dumps(analysis_data.get("mutations", []))
            recommendations_str = json.dumps(analysis_data.get("clinical_recommendations", []))
            full_results_str = json.dumps(analysis_data)
            
            await db.execute("""
                INSERT INTO analysis_history 
                (analysis_id, mutations, overall_risk_score, risk_classification, 
                 recommendations, full_results, confidence_score, user_session)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_data["analysis_id"],
                mutations_str,
                analysis_data.get("overall_risk_score", 0.0),
                analysis_data.get("risk_classification", "UNKNOWN"),
                recommendations_str,
                full_results_str,
                analysis_data.get("confidence_metrics", {}).get("overall_confidence", 0.0),
                analysis_data.get("user_session", "anonymous")
            ))
            
            await db.commit()
            
        except Exception as e:
            import logging
            logging.error(f"Failed to save analysis: {e}")

async def get_analysis_history(limit: int = 10, offset: int = 0) -> List[Dict]:
    """Retrieve analysis history"""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        
        cursor = await db.execute("""
            SELECT analysis_id, timestamp, mutations, overall_risk_score, 
                   risk_classification, confidence_score
            FROM analysis_history
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        rows = await cursor.fetchall()
        
        history = []
        for row in rows:
            history.append({
                "analysis_id": row["analysis_id"],
                "timestamp": row["timestamp"],
                "mutations": json.loads(row["mutations"]),
                "overall_risk_score": row["overall_risk_score"],
                "risk_classification": row["risk_classification"],
                "confidence_score": row["confidence_score"]
            })
        
        return history

async def get_analysis_by_id(analysis_id: str) -> Optional[Dict]:
    """Retrieve specific analysis by ID"""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        
        cursor = await db.execute("""
            SELECT * FROM analysis_history
            WHERE analysis_id = ?
        """, (analysis_id,))
        
        row = await cursor.fetchone()
        
        if row:
            return {
                "analysis_id": row["analysis_id"],
                "timestamp": row["timestamp"],
                "full_results": json.loads(row["full_results"])
            }
        
        return None

async def cache_mutation_analysis(mutation_id: str, gene: str, variant: str, analysis_data: Dict):
    """Cache mutation analysis for faster lookups"""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        try:
            await db.execute("""
                INSERT OR REPLACE INTO mutation_cache
                (mutation_id, gene, variant, analysis_data, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                mutation_id,
                gene,
                variant,
                json.dumps(analysis_data)
            ))
            
            await db.commit()
            
        except Exception as e:
            import logging
            logging.error(f"Failed to cache mutation: {e}")

async def get_cached_mutation(mutation_id: str) -> Optional[Dict]:
    """Retrieve cached mutation analysis"""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        
        cursor = await db.execute("""
            SELECT analysis_data FROM mutation_cache
            WHERE mutation_id = ?
            AND datetime(last_updated) > datetime('now', '-30 days')
        """, (mutation_id,))
        
        row = await cursor.fetchone()
        
        if row:
            return json.loads(row["analysis_data"])
        
        return None

async def cleanup_old_data(days_to_keep: int = 90):
    """Clean up old analysis data"""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        # Delete old analysis history
        await db.execute("""
            DELETE FROM analysis_history
            WHERE datetime(timestamp) < datetime('now', ? || ' days')
        """, (-days_to_keep,))
        
        # Delete old mutation cache
        await db.execute("""
            DELETE FROM mutation_cache
            WHERE datetime(last_updated) < datetime('now', '-30 days')
        """)
        
        await db.commit()