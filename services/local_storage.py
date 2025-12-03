import sqlite3
import time
import os

class StorageService:
    def __init__(self, db_path="data/attendance.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                direction TEXT,
                similarity REAL,
                timestamp REAL,
                synced INTEGER DEFAULT 0
            )
        ''')
        self.conn.commit()

    def add_log(self, name, direction, similarity):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO attendance_logs (name, direction, similarity, timestamp) VALUES (?, ?, ?, ?)",
            (name, direction, similarity, time.time())
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_unsynced_logs(self, limit=50):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM attendance_logs WHERE synced = 0 LIMIT ?", (limit,))
        return cursor.fetchall()

    def mark_synced(self, log_ids):
        cursor = self.conn.cursor()
        cursor.executemany("UPDATE attendance_logs SET synced = 1 WHERE id = ?", [(i,) for i in log_ids])
        self.conn.commit()