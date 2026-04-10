"""
DAG: twitter_to_bronze
Frecuencia: dos veces por semana (lunes y jueves a las 06:00 UTC)
Fuente:     Colección 'twitter_raw' en MongoDB
Destino:    datalake_bronze/twitter/YYYY-MM-DD.json
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from mongo_bronze import fetch_collection, get_mongo_client, write_to_bronze

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
COLLECTION_NAME = "apicomments"   # <— cambia por tu nombre real
BRONZE_LAYER    = "twitter"

DEFAULT_ARGS = {
    "owner":            "data-team",
    "depends_on_past":  False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=10),
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
    Extrae documentos de la colección de Twitter y los escribe en Bronze.

    Como el DAG corre 2 veces/semana, típicamente querrás los documentos
    de los últimos N días. Ajusta el filtro según tu esquema real.
    """
    execution_date: str = context["ds"]      # YYYY-MM-DD

    log.info("Extrayendo colección '%s' para la fecha %s", COLLECTION_NAME, execution_date)

    # ── Filtro opcional ────────────────────────────────────────────────
    # Ejemplo: traer tweets registrados desde el lunes anterior
    # para cubrir el intervalo de 3-4 días entre ejecuciones.
    #
    #   from datetime import datetime, timezone
    #   exec_dt   = datetime.fromisoformat(execution_date).replace(tzinfo=timezone.utc)
    #   week_start = exec_dt - timedelta(days=exec_dt.weekday())  # lunes
    #   query = {"collected_at": {"$gte": week_start, "$lt": exec_dt + timedelta(days=1)}}
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
    dag_id="twitter_to_bronze",
    description="Ingesta bisemanal: MongoDB twitter_raw → datalake_bronze/twitter",
    default_args=DEFAULT_ARGS,
    schedule="0 6 * * 1,4",   # Lunes y jueves a las 06:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["bronze", "ingesta", "twitter"],
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
