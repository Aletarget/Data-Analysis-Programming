"""
DAG: webscraping_to_bronze
Frecuencia: diaria (@daily — medianoche UTC)
Fuente:     Colección 'webscraping_raw' en MongoDB
Destino:    datalake_bronze/webscraping/YYYY-MM-DD.json
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ajusta el import según donde coloques utils/ dentro de tu carpeta dags/
from mongo_bronze import fetch_collection, get_mongo_client, write_to_bronze

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
COLLECTION_NAME = "newsnapshots"   # <— cambia por tu nombre real
BRONZE_LAYER    = "webscraping"

DEFAULT_ARGS = {
    "owner":            "data-team",
    "depends_on_past":  False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


# ──────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────
def check_mongo_connection(**context) -> None:
    """Verifica que MongoDB sea accesible antes de intentar extraer datos."""
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        client.close()
        log.info("Conexión a MongoDB OK.")
    except Exception as e:
        raise RuntimeError(f"No se pudo conectar a MongoDB: {e}") from e


def extract_and_load(**context) -> None:
    """
    Extrae documentos de la colección de webscraping y los escribe en Bronze.

    Usa la fecha lógica del DAG (ds) para filtrar sólo los documentos
    del día de ejecución si tu colección tiene campo 'fecha'.
    Ajusta el filtro según tu esquema real.
    """
    execution_date: str = context["ds"]          # formato YYYY-MM-DD

    log.info("Extrayendo colección '%s' para la fecha %s", COLLECTION_NAME, execution_date)

    # ── Filtro opcional por fecha ──────────────────────────────────────
    # Si tu colección tiene un campo de timestamp, filtra sólo los docs
    # del día correspondiente para evitar re-procesar todo el histórico.
    #
    # Ejemplo con campo 'scraped_at' tipo datetime:
    #
    #   from datetime import datetime, timezone
    #   day_start = datetime.fromisoformat(execution_date).replace(tzinfo=timezone.utc)
    #   day_end   = day_start + timedelta(days=1)
    #   query     = {"scraped_at": {"$gte": day_start, "$lt": day_end}}
    #
    # Si prefieres traer TODO sin filtro, deja query = {}
    # ──────────────────────────────────────────────────────────────────
    query = {}

    documents = fetch_collection(COLLECTION_NAME, query)

    if not documents:
        log.warning("La colección '%s' no devolvió documentos para %s.", COLLECTION_NAME, execution_date)
        return

    log.info("Documentos extraídos: %d", len(documents))

    output_path = write_to_bronze(documents, BRONZE_LAYER, execution_date)
    log.info("Datos escritos en Bronze: %s", output_path)


# ──────────────────────────────────────────────
# Definición del DAG
# ──────────────────────────────────────────────
with DAG(
    dag_id="webscraping_to_bronze",
    description="Ingesta diaria: MongoDB webscraping_raw → datalake_bronze/webscraping",
    default_args=DEFAULT_ARGS,
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["bronze", "ingesta", "webscraping"],
) as dag:

    t1_check = PythonOperator(
        task_id="check_mongo_connection",
        python_callable=check_mongo_connection,
    )

    t2_load = PythonOperator(
        task_id="extract_and_load_to_bronze",
        python_callable=extract_and_load,
    )

    t1_check >> t2_load
