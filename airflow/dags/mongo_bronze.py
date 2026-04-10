"""
Utilidades compartidas para lectura de MongoDB y escritura en datalake_bronze.
"""

import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import Any

from pymongo import MongoClient


# ──────────────────────────────────────────────
# Configuración (puede sobreescribirse con Airflow Variables)
# ──────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "MONGO_URI_REMOVED")
MONGO_DB  = os.getenv("MONGO_DB",  "dataProgrammingAnalysis")
BRONZE_BASE_PATH = os.getenv("BRONZE_BASE_PATH", "/opt/airflow/datalake_bronze")

# ──────────────────────────────────────────────
# Serialización
# ──────────────────────────────────────────────
class MongoEncoder(json.JSONEncoder):
    """Serializa tipos de MongoDB que json nativo no entiende."""

    def default(self, obj: Any) -> Any:
        from bson import ObjectId
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.hex()
        return super().default(obj)


# ──────────────────────────────────────────────
# Lectura desde MongoDB
# ──────────────────────────────────────────────
def get_mongo_client() -> MongoClient:
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=5_000)


def fetch_collection(collection_name: str, query: dict | None = None) -> list[dict]:
    """
    Lee todos los documentos de una colección.

    Args:
        collection_name: Nombre de la colección en MongoDB.
        query: Filtro opcional (default: todos los documentos).

    Returns:
        Lista de documentos como dicts.
    """
    query = query or {}
    with get_mongo_client() as client:
        db   = client[MONGO_DB]
        docs = list(db[collection_name].find(query))
    return docs


# ──────────────────────────────────────────────
# Escritura en datalake_bronze
# ──────────────────────────────────────────────
def write_to_bronze(
    data: list[dict],
    layer: str,
    execution_date: str | None = None,
) -> str:
    """
    Persiste datos en datalake_bronze/<layer>/YYYY-MM-DD.json.

    Args:
        data:           Lista de documentos a escribir.
        layer:          Subcarpeta de Bronze (ej. 'webscraping', 'twitter').
        execution_date: Fecha lógica del DAG (ISO 8601). Si es None usa hoy.

    Returns:
        Ruta absoluta del archivo creado.
    """
    date_str  = execution_date or datetime.utcnow().strftime("%Y-%m-%d")
    dest_dir  = Path(BRONZE_BASE_PATH) / layer
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_file = dest_dir / f"{date_str}.json"

    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=MongoEncoder, ensure_ascii=False, indent=2)

    return str(dest_file)
