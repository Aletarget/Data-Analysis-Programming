from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import re
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

log = logging.getLogger(__name__)


# Configuración

BRONZE_BASE_PATH = os.getenv("BRONZE_BASE_PATH")
SILVER_BASE_PATH = os.getenv("SILVER_BASE_PATH")

DEFAULT_ARGS = {
    "owner":            "data-team",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


# Helpers

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

def clean_text_for_nlp(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Eliminar URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Eliminar etiquetas HTML
    text = re.sub(r'<.*?>', '', text)
    # Eliminar menciones de Twitter (@usuario)
    text = re.sub(r'@\w+', '', text)
    # Reemplazar saltos de línea extra y espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_and_write_parquet(source: str, topic: str, **kwargs) -> str:
    import pandas as pd
    import json

    log.info("Iniciando procesamiento — source: %s, topic: %s", source, topic)

    raw_data = load_latest_bronze(source)

    doc = raw_data[0] if isinstance(raw_data, list) else raw_data

    if source == "twitter":
        snapshot_date = doc.get("date", "")
        raw_docs = []
        for tweet in doc.get("tweets", []):
            tweet["snapshot_date"] = snapshot_date
            raw_docs.append(tweet)
        log.info("Tweets extraídos: %d", len(raw_docs))

    elif source == "webscraping":
        snapshot_id   = doc.get("_id", "")
        snapshot_date = doc.get("date", "")
        raw_docs = []
        for news_item in doc.get("news", []):
            news_item["snapshot_id"]   = snapshot_id
            news_item["snapshot_date"] = snapshot_date
            raw_docs.append(news_item)
        log.info("Noticias extraídas: %d", len(raw_docs))

    else:
        raw_docs = raw_data if isinstance(raw_data, list) else [raw_data]


    cleaned_docs = []
    for doc_item in raw_docs:
        flat = flatten_dict(doc_item)
        for k, v in flat.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v, ensure_ascii=False)
        cleaned_docs.append(flat)


    df = pd.DataFrame(cleaned_docs)

    date_cols = [
        c for c in df.columns
        if ('date' in c.lower() or 'time' in c.lower()) and c != 'snapshot_date'
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col].astype(str), utc=True, errors='coerce')

    silver_dir = Path(SILVER_BASE_PATH)
    silver_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest_file = silver_dir / f"{topic}_processed_{timestamp}.parquet"

    log.info("Aplicando limpieza NLP a columnas de texto...")
    text_cols = [c for c in df.columns if c in ['text', 'content', 'description', 'title']]
    for col in text_cols:
        df[f"{col}_cleaned"] = df[col].apply(clean_text_for_nlp)
        # Calcular la longitud del texto limpio (KPI de Gobernanza futuro)
        df[f"{col}_length"] = df[f"{col}_cleaned"].apply(len)
    
    
    df.to_parquet(
        dest_file,                          
        engine='pyarrow',
        coerce_timestamps='us',              
        allow_truncated_timestamps=True      
    )

    log.info("Silver escrito: %s — %d filas", dest_file, len(df))
    return str(dest_file)

# DAG to clean and convert webscrapping .json file to .parquet file 
with DAG(
    dag_id="silver_webscraping",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["silver", "clean", "webscraping"],
) as dag_silver_webscraping:

    process = PythonOperator(
        task_id="process_webscraping",
        python_callable=process_and_write_parquet,
        op_kwargs={
            "source": "webscraping",
            "topic":  "noticias"
        },
    )
    
    trigger_gold = TriggerDagRunOperator(
        task_id="trigger_gold_webscraping",
        trigger_dag_id="gold_processing_dag",
        wait_for_completion=False,
    )

    process >> trigger_gold

# DAG to clean and convert tweets .json file to .parquet file 
with DAG(
    dag_id="silver_twitter",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["silver", "clean", "tweets"],
) as dag_silver_twitter:

    process = PythonOperator(
        task_id="process_twitter",
        python_callable=process_and_write_parquet,
        op_kwargs={
            "source": "twitter",
            "topic":  "tweets"
        },
    )
    
    trigger_gold = TriggerDagRunOperator(
        task_id="trigger_gold_twitter",
        trigger_dag_id="gold_processing_dag",
        wait_for_completion=False,
    )

    process >> trigger_gold