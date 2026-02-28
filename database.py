"""
Database module for storing and retrieving face signatures
"""
import sqlite3
import numpy as np
import json
import pickle
import os
from datetime import datetime
from typing import List, Tuple, Optional

from config import DATABASE_PATH
from utils import normalize_vector, euclidean_distance

class FaceDatabase:
    """Database for storing face signatures and identities"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_PATH
        self.conn = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connect to SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Face signatures table (page 3: S(I) = (R, F, E, A))
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signatures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                resolution TEXT NOT NULL,  -- R: (height, width)
                f_vector BLOB NOT NULL,    -- F: identification vector (2062 dim)
                e_vector BLOB NOT NULL,    -- E: verification vector (128 dim)
                a_vector TEXT NOT NULL,    -- A: authentication vector [a1, a2]
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        # Remove unique constraint if exists
        try:
            cursor.execute("DROP INDEX IF EXISTS idx_signatures_unique")
        except:
            pass
        
        # Access logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_type TEXT,  -- 'granted', 'denied', 'suspicious'
                confidence REAL,
                liveness_check BOOLEAN,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        self.conn.commit()
    
    def add_person(self, name: str) -> int:
        """Add a new person to the database"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO persons (name) VALUES (?)",
            (name,)
        )
        person_id = cursor.lastrowid
        self.conn.commit()
        return person_id
    
    def add_signature(self, person_id: int, signature: dict):
        """
        Add a face signature for a person
        signature format: {'R': (h, w), 'F': np.array, 'E': np.array, 'A': [a1, a2]}
        """
        cursor = self.conn.cursor()
        
        # Convert numpy arrays to bytes
        f_bytes = signature['F'].tobytes()
        e_bytes = signature['E'].tobytes()
        
        # Convert A vector to JSON string
        a_json = json.dumps(signature['A'])
        
        # Convert resolution to string
        resolution_str = f"{signature['R'][0]},{signature['R'][1]}"
        
        cursor.execute('''
            INSERT INTO signatures 
            (person_id, resolution, f_vector, e_vector, a_vector)
            VALUES (?, ?, ?, ?, ?)
        ''', (person_id, resolution_str, f_bytes, e_bytes, a_json))
        
        self.conn.commit()
    
    def get_person_signatures(self, person_id: int) -> List[dict]:
        """Get all signatures for a person"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM signatures WHERE person_id = ? ORDER BY created_at DESC",
            (person_id,)
        )
        
        signatures = []
        for row in cursor.fetchall():
            # Convert bytes back to numpy arrays
            f_vector = np.frombuffer(row['f_vector'], dtype=np.float32)
            e_vector = np.frombuffer(row['e_vector'], dtype=np.float32)
            a_vector = json.loads(row['a_vector'])
            
            # Parse resolution
            h, w = map(int, row['resolution'].split(','))
            
            signatures.append({
                'id': row['id'],
                'R': (h, w),
                'F': f_vector,
                'E': e_vector,
                'A': a_vector,
                'created_at': row['created_at']
            })
        
        return signatures
    
    def find_person_by_f_vector(self, f_vector: np.ndarray, threshold: float = 0.7) -> Tuple[Optional[int], float]:
        """
        Find person by identification vector F (page 4, step 2)
        Returns (person_id, distance) if found, (None, inf) otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT person_id FROM signatures")
        person_ids = [row[0] for row in cursor.fetchall()]
        
        min_distance = float('inf')
        best_person_id = None
        
        for person_id in person_ids:
            # Get all signatures for this person
            signatures = self.get_person_signatures(person_id)
            if not signatures:
                continue
            
            # Calculate minimum distance to any signature of this person
            for sig in signatures:
                distance = euclidean_distance(f_vector, sig['F'])
                if distance < min_distance:
                    min_distance = distance
                    best_person_id = person_id
        
        # Check if the minimum distance is below threshold
        if min_distance <= threshold:
            return best_person_id, min_distance
        else:
            return None, min_distance
    
    def verify_person(self, person_id: int, e_vector: np.ndarray, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Verify person using verification vector E (page 4, step 3)
        Returns (verified, min_distance)
        """
        signatures = self.get_person_signatures(person_id)
        if not signatures:
            return False, float('inf')
        
        # Find minimum distance to any stored E vector
        min_distance = float('inf')
        for sig in signatures:
            distance = euclidean_distance(e_vector, sig['E'])
            if distance < min_distance:
                min_distance = distance
        
        return min_distance <= threshold, min_distance
    
    def log_access(self, person_id: Optional[int], access_type: str, 
                   confidence: float, liveness_check: bool):
        """Log access attempt"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO access_logs (person_id, access_type, confidence, liveness_check)
            VALUES (?, ?, ?, ?)
        ''', (person_id, access_type, confidence, liveness_check))
        self.conn.commit()
    
    def get_statistics(self) -> dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Count persons
        cursor.execute("SELECT COUNT(*) FROM persons")
        stats['total_persons'] = cursor.fetchone()[0]
        
        # Count signatures
        cursor.execute("SELECT COUNT(*) FROM signatures")
        stats['total_signatures'] = cursor.fetchone()[0]
        
        # Count access logs
        cursor.execute("SELECT COUNT(*) FROM access_logs")
        stats['total_access_logs'] = cursor.fetchone()[0]
        
        # Count successful vs failed accesses
        cursor.execute("SELECT access_type, COUNT(*) FROM access_logs GROUP BY access_type")
        stats['access_breakdown'] = dict(cursor.fetchall())
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()