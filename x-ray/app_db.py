"""
app_db.py
SQLite database layer for PneumoVision.
- Creates tables on first run.
- Provides helper functions for predictions, metrics, history, admin.
"""
import os
import sqlite3
from datetime import datetime

DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "pneumovision.db")
os.makedirs(DB_DIR, exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Predictions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            prediction TEXT,
            confidence REAL,
            preprocessing TEXT,
            model TEXT,
            inference_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Experiments / metrics table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            preprocessing TEXT,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            roc_auc REAL,
            training_time REAL,
            inference_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Users table (simple admin)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Logs table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(image_path, prediction, confidence, preprocessing, model, inference_time):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (image_path, prediction, confidence, preprocessing, model, inference_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (image_path, prediction, confidence, preprocessing, model, inference_time))
    conn.commit()
    conn.close()


def get_history(limit=100):
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_history():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


def save_experiment(model, preprocessing, accuracy, precision, recall, f1, roc_auc, training_time, inference_time):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO experiments (model, preprocessing, accuracy, precision, recall, f1_score, roc_auc, training_time, inference_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (model, preprocessing, accuracy, precision, recall, f1, roc_auc, training_time, inference_time))
    conn.commit()
    conn.close()


def get_experiments():
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_action(action, details=""):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO logs (action, details) VALUES (?, ?)", (action, details))
    conn.commit()
    conn.close()


def get_logs(limit=200):
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]