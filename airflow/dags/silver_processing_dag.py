from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
BRONZE_BASE_PATH = os.getenv("BRONZE_BASE_PATH", "/tmp/bronze")
SILVER_BASE_PATH = os.getenv("SILVER_BASE_PATH", "/tmp/silver")

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
    bronze_dir = Path(BRONZE_BASE_PATH)
    json_files = sorted(bronze_dir.glob(f"{source}_*.json"), reverse=True)

    if not json_files:
        raise FileNotFoundError(f"No hay archivos JSON con prefijo '{source}_' en {bronze_dir}")

    with open(json_files[0], encoding="utf-8") as f:
        return json.load(f)

def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def clean_document(doc: dict) -> dict:
    doc = flatten_dict(doc)
    cleaned = {}
    for key, value in doc.items():
        if value is None or (isinstance(value, str) and value.strip() == ""):
            continue
        cleaned[key] = value
    return cleaned

def process_and_write_parquet(source: str, topic: str) -> str:
    import pandas as pd
    import json

    raw_data = load_latest_bronze(source)
    
    # 1. Normalización de la estructura (ajustado para twitter o webscraping)
    # Asumimos que raw_data es un dict (el snapshot) o una lista [snapshot]
    doc = raw_data[0] if isinstance(raw_data, list) else raw_data
    
    if source == "twitter":
        raw_docs = doc.get("tweets", [])
    elif source == "webscraping":
        raw_docs = doc.get("news", [])
    else:
        # Por seguridad, si no es ninguno, intentamos tratarlo como lista de documentos
        raw_docs = raw_data if isinstance(raw_data, list) else [raw_data]

    # 2. Aplanamiento y serialización
    cleaned_docs = []
    for doc_item in raw_docs:
        flat = flatten_dict(doc_item)
        # Convertimos CUALQUIER valor que sea dict o list a una cadena JSON
        for k, v in flat.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v, ensure_ascii=False)
        cleaned_docs.append(flat)

    df = pd.DataFrame(cleaned_docs)

    # 3. Conversión de fechas inteligente
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col].astype(str), utc=True, errors='coerce')

    # 4. Asegurar que las columnas object sean strings planos
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).replace('None', '')

    # 5. Escritura a Parquet
    silver_dir = Path(SILVER_BASE_PATH)
    silver_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest_file = silver_dir / f"{topic}_processed_{timestamp}.parquet"
    
    df.to_parquet(dest_file, index=False, engine="pyarrow")
    
    log.info("Silver escrito para %s: %s — %d filas", source, dest_file, len(df))
    return str(dest_file)
# ──────────────────────────────────────────────
# DAG
# ──────────────────────────────────────────────
with DAG(
    dag_id="silver_processing_dag",
    description="Procesa JSON de bronze a Parquet en silver",
    default_args=DEFAULT_ARGS,
    schedule="0 7 * * 1,4",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["silver", "procesamiento"],
) as dag:

    t_web = PythonOperator(
        task_id="process_webscraping_to_parquet",
        python_callable=lambda: process_and_write_parquet("webscraping", "noticias"),
    )

    t_twitter = PythonOperator(
        task_id="process_twitter_to_parquet",
        python_callable=lambda: process_and_write_parquet("twitter", "tweets"),
    )

    t_web >> t_twitter