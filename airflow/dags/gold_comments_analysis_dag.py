"""
DAG: gold_webscraping_analytics_dag

Descripción:
    Genera datasets analíticos GOLD a partir de noticias procesadas
    en Silver Layer.

Outputs:
    - news_comment_metrics_*.parquet
    - top_news_by_comments_*.parquet
    - comments_length_metrics_*.parquet
"""

from __future__ import annotations

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
SILVER_BASE_PATH = os.getenv("SILVER_BASE_PATH")
GOLD_BASE_PATH = os.getenv("GOLD_BASE_PATH")

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
def load_latest_news_parquet():
    """
    Carga el parquet más reciente de noticias.
    """
    import pandas as pd

    silver_dir = Path(SILVER_BASE_PATH)

    parquet_files = sorted(
        silver_dir.glob("noticias_processed_*.parquet"),
        reverse=True
    )

    if not parquet_files:
        raise FileNotFoundError(
            "No existen archivos noticias_processed en Silver"
        )

    latest = parquet_files[0]

    log.info("Leyendo Silver: %s", latest)

    return pd.read_parquet(latest)


def write_gold(df, prefix: str):
    """
    Escribe parquet GOLD.
    """
    gold_dir = Path(GOLD_BASE_PATH)
    gold_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    dest = gold_dir / f"{prefix}_{timestamp}.parquet"

    df.to_parquet(dest, index=False, engine="pyarrow")

    log.info(
        "Gold escrito: %s (%d filas)",
        dest,
        len(df)
    )

    return str(dest)


# ──────────────────────────────────────────────
# GOLD — Métricas de comentarios
# ──────────────────────────────────────────────
def build_news_comment_metrics():
    """
    Calcula métricas generales por noticia.
    """
    import pandas as pd

    df = load_latest_news_parquet()

    if "comments" not in df.columns:
        log.warning("No existe columna comments")
        return

    df["total_comments"] = df["comments"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    df["avg_comment_length"] = df["comments"].apply(
        lambda comments: (
            sum(len(c) for c in comments) / len(comments)
            if isinstance(comments, list) and len(comments) > 0
            else 0
        )
    )

    metrics = df[
        [
            "newsLink",
            "total_comments",
            "avg_comment_length",
        ]
    ]

    write_gold(
        metrics,
        "news_comment_metrics"
    )


# ──────────────────────────────────────────────
# GOLD — Noticias más comentadas
# ──────────────────────────────────────────────
def build_top_news_by_comments():
    """
    Ranking de noticias con más comentarios.
    """
    df = load_latest_news_parquet()

    df["total_comments"] = df["comments"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    top_news = (
        df[
            [
                "newsLink",
                "total_comments",
            ]
        ]
        .sort_values("total_comments", ascending=False)
    )

    write_gold(
        top_news,
        "top_news_by_comments"
    )


# ──────────────────────────────────────────────
# GOLD — Métricas individuales comentarios
# ──────────────────────────────────────────────
def build_comment_length_metrics():
    """
    Explota comentarios individuales para análisis textual.
    """
    import pandas as pd

    df = load_latest_news_parquet()

    rows = []

    for _, row in df.iterrows():

        news_link = row.get("newsLink")

        comments = row.get("comments", [])

        if not isinstance(comments, list):
            continue

        for comment in comments:

            rows.append({
                "newsLink": news_link,
                "comment": comment,
                "comment_length": len(comment),
                "word_count": len(comment.split())
            })

    comments_df = pd.DataFrame(rows)

    write_gold(
        comments_df,
        "comments_length_metrics"
    )


# ──────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────
def process_gold_webscraping(**context):
    build_news_comment_metrics()
    build_top_news_by_comments()
    build_comment_length_metrics()


# ══════════════════════════════════════════════
# DAG
# ══════════════════════════════════════════════
with DAG(
    dag_id="gold_webscraping_analytics_dag",
    description="Genera datasets analíticos GOLD para noticias scrapeadas",
    default_args=DEFAULT_ARGS,
    schedule="45 7 * * 1,4",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["gold", "analytics", "webscraping"],
) as dag:

    t1_gold_webscraping = PythonOperator(
        task_id="build_gold_webscraping_analytics",
        python_callable=process_gold_webscraping,
    )

    t1_gold_webscraping