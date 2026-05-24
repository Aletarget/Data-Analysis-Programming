"""
DAG: silver_processing_dag
Descripción: Procesa los JSON crudos de datalake_bronze y los convierte
             a Parquet limpio en datalake_silver.

  Transformaciones:
    - Elimina campos con valor None / null
    - Convierte tipos de datos (fechas a datetime, números a int/float)
    - Normaliza campos anidados simples a columnas planas
    - Escribe un Parquet por fuente con nomenclatura {topic}_processed_YYYYMMDD_HHMMSS.parquet
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
BRONZE_BASE_PATH = os.getenv("BRONZE_BASE_PATH")
SILVER_BASE_PATH = os.getenv("SILVER_BASE_PATH")

DEFAULT_ARGS = {
    "owner":            "data-team",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_latest_bronze(source: str) -> list[dict]:
    """Busca el JSON más reciente que tenga el prefijo source_ en el nombre."""
    bronze_dir = Path(BRONZE_BASE_PATH)
    json_files = sorted(bronze_dir.glob(f"{source}_*.json"), reverse=True)

    if not json_files:
        raise FileNotFoundError(
            f"No hay archivos JSON con prefijo '{source}_' en {bronze_dir}"
        )

    latest = json_files[0]
    log.info("Leyendo Bronze: %s", latest)

    with open(latest, encoding="utf-8") as f:
        return json.load(f)


def clean_document(doc: dict) -> dict:
    """
    Limpia un documento individual:
      - Elimina claves con valor None o string vacío
      - Convierte strings ISO a datetime donde corresponda
      - Intenta convertir strings numéricos a int/float
      - Aplana dicts anidados un nivel con prefijo key_subkey
    """
    cleaned = {}
    for key, value in doc.items():

        if value is None:
            continue

        if isinstance(value, str) and value.strip() == "":
            continue

        if isinstance(value, str) and ("T" in value or "-" in value):
            try:
                cleaned[key] = datetime.fromisoformat(value)
                continue
            except ValueError:
                pass

        if isinstance(value, str):
            try:
                cleaned[key] = int(value)
                continue
            except ValueError:
                pass
            try:
                cleaned[key] = float(value)
                continue
            except ValueError:
                pass

        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if sub_val is not None:
                    cleaned[f"{key}_{sub_key}"] = sub_val
            continue

        cleaned[key] = value

    return cleaned


def process_and_write_parquet(source: str, topic: str) -> str:
    """
    Lee el JSON más reciente de bronze, aplica limpieza
    y escribe datalake_silver/{topic}_processed_YYYYMMDD_HHMMSS.parquet
    """
    import pandas as pd

    raw_docs = load_latest_bronze(source)
    log.info("Documentos cargados desde Bronze (%s): %d", source, len(raw_docs))

    cleaned_docs = [clean_document(doc) for doc in raw_docs]

    df = pd.DataFrame(cleaned_docs)
    log.info("Shape después de limpieza: %s", df.shape)
    log.info("Columnas: %s", list(df.columns))

    df = df.infer_objects()

    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = pd.to_datetime(df[col], utc=True)
        except Exception:
            pass

    silver_dir = Path(SILVER_BASE_PATH)
    silver_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest_file = silver_dir / f"{topic}_processed_{timestamp}.parquet"
    df.to_parquet(dest_file, index=False, engine="pyarrow")

    log.info("Silver escrito: %s — %d filas, %d columnas", dest_file, len(df), len(df.columns))
    return str(dest_file)


# ──────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────
def process_webscraping(**context) -> None:
    process_and_write_parquet("webscraping", "noticias")


def process_twitter(**context) -> None:
    process_and_write_parquet("twitter", "tweets")


# ══════════════════════════════════════════════
# DAG — Silver Processing
# ══════════════════════════════════════════════
with DAG(
    dag_id="silver_processing_dag",
    description="Procesa los JSON crudos de bronze y los convierte a Parquet en silver",
    default_args=DEFAULT_ARGS,
    schedule="0 7 * * 1,4",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["silver", "procesamiento"],
) as dag:

    t2_webscraping = PythonOperator(
        task_id="process_webscraping_to_parquet",
        python_callable=process_webscraping,
    )

    t3_twitter = PythonOperator(
        task_id="process_twitter_to_parquet",
        python_callable=process_twitter,
    )

    t2_webscraping
    t3_twitter