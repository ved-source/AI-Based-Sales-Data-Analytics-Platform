import sqlite3
import json
import os
from typing import Dict, Any, Optional, List


class DatabaseManager:
    _instance = None

    @classmethod
    def get_instance(cls, db_path: str = "sales_analytics.db"):
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance

    def __init__(self, db_path: str):
        if hasattr(self, "_initialized"):
            return
        self.db_path = db_path
        self._initialized = True
        self.initialize_tables()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_tables(self):
        with self.get_connection() as conn:
            c = conn.cursor()
            # 1. Pipeline Runs Table
            c.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    file_name TEXT,
                    status TEXT
                );
            """)

            # 2. Analytics Results Table (stores compiled data in JSON formats for simplicity)
            c.execute("""
                CREATE TABLE IF NOT EXISTS run_data (
                    run_id TEXT PRIMARY KEY,
                    kpi_sales REAL,
                    kpi_orders INTEGER,
                    kpi_profit REAL,
                    kpi_margin REAL,
                    schema_json TEXT,
                    eda_json TEXT,
                    forecast_json TEXT,
                    anomaly_json TEXT,
                    shap_json TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
            """)

            # 3. Chat Messages Table
            c.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    sender TEXT,
                    message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
            """)
            conn.commit()

    def create_run(self, run_id: str, file_name: str) -> None:
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, file_name, status) VALUES (?, ?, 'RUNNING')",
                (run_id, file_name)
            )
            conn.commit()

    def update_run_status(self, run_id: str, status: str) -> None:
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ?",
                (status, run_id)
            )
            conn.commit()

    def save_run_results(self, run_id: str, kpi_sales: float, kpi_orders: int, kpi_profit: float, kpi_margin: float,
                         schema: Dict, eda: Dict, forecast: Dict, anomaly: Dict, shap: Dict) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO run_data (
                    run_id, kpi_sales, kpi_orders, kpi_profit, kpi_margin,
                    schema_json, eda_json, forecast_json, anomaly_json, shap_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, kpi_sales, kpi_orders, kpi_profit, kpi_margin,
                json.dumps(schema), json.dumps(eda), json.dumps(forecast), json.dumps(anomaly), json.dumps(shap)
            ))
            conn.commit()

    def get_run_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM run_data WHERE run_id = ?", (run_id,)).fetchone()
            if not row:
                return None
            return {
                "run_id": row["run_id"],
                "kpis": {
                    "total_sales": row["kpi_sales"],
                    "total_orders": row["kpi_orders"],
                    "total_profit": row["kpi_profit"],
                    "profit_margin_pct": row["kpi_margin"]
                },
                "schema": json.loads(row["schema_json"]),
                "eda": json.loads(row["eda_json"]),
                "forecast": json.loads(row["forecast_json"]),
                "anomaly": json.loads(row["anomaly_json"]),
                "shap": json.loads(row["shap_json"])
            }

    def save_chat_message(self, run_id: str, sender: str, message: str) -> None:
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO chat_history (run_id, sender, message) VALUES (?, ?, ?)",
                (run_id, sender, message)
            )
            conn.commit()

    def get_chat_history(self, run_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT sender, message, timestamp FROM chat_history WHERE run_id = ? ORDER BY id ASC LIMIT ?",
                (run_id, limit)
            )
            return [{"sender": r["sender"], "message": r["message"], "timestamp": r["timestamp"]} for r in rows]

    def get_all_runs(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT run_id, timestamp, file_name, status FROM runs ORDER BY timestamp DESC")
            return [{"run_id": r["run_id"], "timestamp": r["timestamp"], "file_name": r["file_name"], "status": r["status"]} for r in rows]
