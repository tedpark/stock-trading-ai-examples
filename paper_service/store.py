"""Atomic request claims and persisted run results, one SQLite database per service."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

class Conflict(ValueError):pass

class Store:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, status TEXT NOT NULL, result TEXT, error TEXT)')
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=30)
        db.row_factory=sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()
    def claim(self,identifier,fingerprint):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM runs WHERE id=?',(identifier,)).fetchone()
            if row:
                if row['fingerprint']!=fingerprint:raise Conflict('request ID already used for different input')
                if row['status']!='completed':raise Conflict('request already claimed; inspect its status before retrying')
                return json.loads(row['result'])
            db.execute('INSERT INTO runs(id,fingerprint,status) VALUES (?,?,?)',(identifier,fingerprint,'running'))
        return None
    def finish(self,identifier,result=None,error=None):
        with self.connect() as db:
            db.execute('UPDATE runs SET status=?,result=?,error=? WHERE id=?',
                       ('failed' if error else 'completed',json.dumps(result,allow_nan=False) if result else None,error,identifier))
    def get(self,identifier):
        with self.connect() as db:row=db.execute('SELECT * FROM runs WHERE id=?',(identifier,)).fetchone()
        if row is None:return None
        result=dict(row);result.pop('fingerprint')
        result['result']=json.loads(result['result']) if result['result'] else None
        return result
    def list(self):
        with self.connect() as db:return [dict(r) for r in db.execute('SELECT id,status FROM runs ORDER BY rowid DESC LIMIT 100')]
