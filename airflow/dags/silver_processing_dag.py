"""
DAG: silver_processing_dag
Descripción: Procesa los JSON crudos de datalake_bronze y los convierte
             a Parquet limpio en datalake_silver.

  Transformaciones:
    - Elimina campos con valor None / null
    - Convierte tipos de datos (fechas a datetime, números a int/float)
    - Normaliza campos anidados simples a columnas planas
    - Escribe un Parquet por fuente: webscraping.parquet / twitter.parquet
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
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
BRONZE_BASE_PATH = os.getenv("BRONZE_BASE_PATH", "/opt/airflow/datalake_bronze")
SILVER_BASE_PATH = os.getenv("SILVER_BASE_PATH", "/opt/airflow/datalake_silver")

DEFAULT_ARGS = {
    "owner":            "data-team",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


# ──────────────────────────────────────────────
# Helpers de transformación
# ──────────────────────────────────────────────
def load_latest_bronze(source: str) -> list[dict]:
    """
    Carga el JSON más reciente de datalake_bronze/<source>/.
    Busca el archivo con la fecha más alta en el nombre.
    """
    bronze_dir = Path(BRONZE_BASE_PATH) / source
    json_files = sorted(bronze_dir.glob("*.json"), reverse=True)

    if not json_files:
        raise FileNotFoundError(f"No hay archivos JSON en {bronze_dir}")

    latest = json_files[0]
    log.info("Leyendo Bronze: %s", latest)

    with open(latest, encoding="utf-8") as f:
        return json.load(f)


def clean_document(doc: dict) -> dict:
    """
    Limpia un documento individual:
      - Elimina claves con valor None
      - Convierte strings ISO a datetime donde corresponda
      - Intenta convertir strings numéricos a int/float
    """
    cleaned = {}
    for key, value in doc.items():

        # Eliminar nulos
        if value is None:
            continue

        # Strings vacíos → omitir
        if isinstance(value, str) and value.strip() == "":
            continue

        # Intentar parsear fechas ISO (ej. "2024-01-15T10:30:00")
        if isinstance(value, str) and ("T" in value or "-" in value):
            try:
                cleaned[key] = datetime.fromisoformat(value)
                continue
            except ValueError:
                pass

        # Intentar convertir strings numéricos
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

        # Dicts anidados simples → aplanar con prefijo (un nivel)
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if sub_val is not None:
                    cleaned[f"{key}_{sub_key}"] = sub_val
            continue

        cleaned[key] = value

    return cleaned


def process_and_write_parquet(source: str) -> str:
    """
    Lee el JSON más reciente de bronze, aplica limpieza
    y escribe datalake_silver/<source>.parquet.
    """
    import pandas as pd

    raw_docs = load_latest_bronze(source)
    log.info("Documentos cargados desde Bronze (%s): %d", source, len(raw_docs))

    cleaned_docs = [clean_document(doc) for doc in raw_docs]

    df = pd.DataFrame(cleaned_docs)
    log.info("Shape después de limpieza: %s", df.shape)
    log.info("Columnas: %s", list(df.columns))

    # Inferir tipos automáticamente
    df = df.infer_objects()

    # Convertir columnas object que sean fechas
    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = pd.to_datetime(df[col], utc=True)
        except Exception:
            pass

    silver_dir = Path(SILVER_BASE_PATH)
    silver_dir.mkdir(parents=True, exist_ok=True)

    dest_file = silver_dir / f"{source}.parquet"
    df.to_parquet(dest_file, index=False, engine="pyarrow")

    log.info("Silver escrito: %s — %d filas, %d columnas", dest_file, len(df), len(df.columns))
    return str(dest_file)


# ──────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────
def process_webscraping(**context) -> None:
    process_and_write_parquet("webscraping")


def process_twitter(**context) -> None:
    process_and_write_parquet("twitter")


def check_bronze_exists(**context) -> None:
    """Verifica que existan archivos en bronze antes de procesar."""
    for source in ("webscraping", "twitter"):
        bronze_dir = Path(BRONZE_BASE_PATH) / source
        files = list(bronze_dir.glob("*.json")) if bronze_dir.exists() else []
        if not files:
            raise FileNotFoundError(
                f"No hay archivos JSON en {bronze_dir}. "
                f"Ejecuta primero bronze_ingestion_dag."
            )
        log.info("Bronze OK para '%s': %d archivos encontrados.", source, len(files))


# ══════════════════════════════════════════════
# DAG — Silver Processing
# ══════════════════════════════════════════════
with DAG(
    dag_id="silver_processing_dag",
    description="Procesa JSON de bronze → Parquet limpio en datalake_silver",
    default_args=DEFAULT_ARGS,
    schedule="0 7 * * 1,4",   # Corre después del bronze de Twitter (06:00)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["silver", "procesamiento"],
) as dag:

    t1_check = PythonOperator(
        task_id="check_bronze_exists",
        python_callable=check_bronze_exists,
    )

    t2_webscraping = PythonOperator(
        task_id="process_webscraping_to_parquet",
        python_callable=process_webscraping,
    )

    t3_twitter = PythonOperator(
        task_id="process_twitter_to_parquet",
        python_callable=process_twitter,
    )

    # Primero verifica, luego procesa ambas fuentes en paralelo
    t1_check >> [t2_webscraping, t3_twitter]
