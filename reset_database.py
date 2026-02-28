#!/usr/bin/env python3
"""
Reset the FaSIVA database
"""
import os
import sqlite3

def reset_database():
    """Reset the database completely"""
    db_path = "faces_database.db"
    
    if os.path.exists(db_path):
        print(f"Removing existing database: {db_path}")
        os.remove(db_path)
    
    # Create new database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signatures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            resolution TEXT NOT NULL,
            f_vector BLOB NOT NULL,
            e_vector BLOB NOT NULL,
            a_vector TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_type TEXT,
            confidence REAL,
            liveness_check BOOLEAN,
            FOREIGN KEY (person_id) REFERENCES persons (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print("Database reset successfully!")
    print("Tables created: persons, signatures, access_logs")

if __name__ == "__main__":
    reset_database()
