"""
DAG: bronze_ingestion_dag
Descripción: Ingesta unificada desde MongoDB hacia datalake_bronze en JSON crudo.

  - webscraping : diario (@daily) — trae el último documento por createdAt
  - twitter     : bisemanal (lunes y jueves 06:00 UTC) — trae todos los documentos
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from airflow import DAG
from airflow.operators.python import PythonOperator
from pymongo import MongoClient

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
MONGO_URI        = os.getenv("MONGO_URI")
MONGO_DB         = os.getenv("MONGO_DB")
BRONZE_BASE_PATH = os.getenv("BRONZE_BASE_PATH")

COLLECTION_WEBSCRAPING = "newsnapshots"   
COLLECTION_TWITTER     = "apicomments"       

DEFAULT_ARGS = {
    "owner":            "data-team",
    "depends_on_past":  False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


# ──────────────────────────────────────────────
# Serialización (ObjectId, datetime, bytes)
# ──────────────────────────────────────────────
class MongoEncoder(json.JSONEncoder):
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
# Helpers
# ──────────────────────────────────────────────
def get_client() -> MongoClient:
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=5_000)


def write_bronze(data: list[dict], source: str, execution_date: str) -> str:
    """Escribe lista de documentos en datalake_bronze/<source>/YYYY-MM-DD.json."""
    dest_dir = Path(BRONZE_BASE_PATH) / source
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"{execution_date}.json"
    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=MongoEncoder, ensure_ascii=False, indent=2)
    return str(dest_file)


# ──────────────────────────────────────────────
# Tasks compartidos
# ──────────────────────────────────────────────
def check_connection(**context) -> None:
    """Verifica conectividad con MongoDB."""
    try:
        client = get_client()
        client.admin.command("ping")
        client.close()
        log.info("Conexión a MongoDB OK.")
    except Exception as e:
        raise RuntimeError(f"No se pudo conectar a MongoDB: {e}") from e


# ──────────────────────────────────────────────
# Tasks de extracción
# ──────────────────────────────────────────────
def extract_webscraping(**context) -> None:
    """
    Extrae el documento más reciente de la colección de webscraping
    ordenando por createdAt descendente.
    """
    execution_date: str = context["ds"]

    with get_client() as client:
        doc = client[MONGO_DB][COLLECTION_WEBSCRAPING].find_one(
            sort=[("createdAt", -1)]
        )

    if not doc:
        log.warning("Colección '%s' vacía.", COLLECTION_WEBSCRAPING)
        return

    log.info("Documento webscraping extraído — createdAt: %s", doc.get("createdAt"))
    path = write_bronze([doc], "webscraping", execution_date)
    log.info("Bronze escrito: %s", path)


def extract_twitter(**context) -> None:
    """
    Extrae todos los documentos de la colección de Twitter.
    """
    execution_date: str = context["ds"]

    with get_client() as client:
        docs = list(client[MONGO_DB][COLLECTION_TWITTER].find({}))

    if not docs:
        log.warning("Colección '%s' vacía.", COLLECTION_TWITTER)
        return

    log.info("Documentos Twitter extraídos: %d", len(docs))
    path = write_bronze(docs, "twitter", execution_date)
    log.info("Bronze escrito: %s", path)


# ══════════════════════════════════════════════
# DAG 1 — Webscraping (diario)
# ══════════════════════════════════════════════
with DAG(
    dag_id="bronze_ingestion_webscraping",
    description="Ingesta diaria: último documento de webscraping → datalake_bronze/webscraping",
    default_args=DEFAULT_ARGS,
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["bronze", "ingesta", "webscraping"],
) as dag_webscraping:

    t1 = PythonOperator(task_id="check_mongo_connection", python_callable=check_connection)
    t2 = PythonOperator(task_id="extract_webscraping",    python_callable=extract_webscraping)

    t1 >> t2


# ══════════════════════════════════════════════
# DAG 2 — Twitter (lunes y jueves 06:00 UTC)
# ══════════════════════════════════════════════
with DAG(
    dag_id="bronze_ingestion_twitter",
    description="Ingesta bisemanal: todos los docs de Twitter → datalake_bronze/twitter",
    default_args=DEFAULT_ARGS,
    schedule="0 6 * * 1,4",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["bronze", "ingesta", "twitter"],
) as dag_twitter:

    t1 = PythonOperator(task_id="check_mongo_connection", python_callable=check_connection)
    t2 = PythonOperator(task_id="extract_twitter",        python_callable=extract_twitter)

    t1 >> t2
