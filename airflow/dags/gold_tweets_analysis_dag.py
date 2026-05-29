"""
DAG: gold_twitter_analytics_dag

Descripción:
    Genera datasets analíticos desde los Parquet de silver.

Outputs:
    - tweets_daily_metrics_*.parquet
    - top_authors_*.parquet
    - tweets_language_distribution_*.parquet
    - engagement_metrics_*.parquet
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
def load_latest_tweets_parquet():
    """
    Carga el parquet más reciente de tweets.
    """
    import pandas as pd

    silver_dir = Path(SILVER_BASE_PATH)

    parquet_files = sorted(
        silver_dir.glob("tweets_processed_*.parquet"),
        reverse=True
    )

    if not parquet_files:
        raise FileNotFoundError(
            "No existen Parquet de tweets en Silver"
        )

    latest = parquet_files[0]

    log.info("Leyendo Silver: %s", latest)

    return pd.read_parquet(latest)


def write_gold(df, prefix: str):
    """
    Escribe dataset GOLD.
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
# GOLD — Métricas diarias
# ──────────────────────────────────────────────
def build_daily_tweet_metrics():
    """
    Métricas diarias de actividad.
    """
    import pandas as pd

    df = load_latest_tweets_parquet()
    
    log.info("COLUMNAS DISPONIBLES:") 
    log.info(df.columns.tolist())

    df["createdAt"] = pd.to_datetime(df["createdAt"])

    df["date"] = df["createdAt"].dt.date

    metrics = (
        df.groupby("date")
        .agg(
            total_tweets=("id", "count"),
            total_likes=("likeCount", "sum"),
            total_retweets=("retweetCount", "sum"),
            total_replies=("replyCount", "sum"),
            total_views=("viewCount", "sum"),
        )
        .reset_index()
        .sort_values("date")
    )

    write_gold(metrics, "tweets_daily_metrics")


# ──────────────────────────────────────────────
# GOLD — Top autores
# ──────────────────────────────────────────────
def build_top_authors():
    """
    Usuarios con más tweets y engagement.
    """
    df = load_latest_tweets_parquet()

    if "author_userName" not in df.columns:
        log.warning("No existe author_userName")
        return

    top_authors = (
        df.groupby("author_userName")
        .agg(
            total_tweets=("id", "count"),
            total_likes=("likeCount", "sum"),
            avg_views=("viewCount", "mean"),
            followers=("author_followers", "max"),
        )
        .reset_index()
        .sort_values("total_likes", ascending=False)
    )

    write_gold(top_authors, "top_authors")


# ──────────────────────────────────────────────
# GOLD — Distribución por idioma
# ──────────────────────────────────────────────
def build_language_distribution():
    """
    Distribución de tweets por idioma.
    """
    df = load_latest_tweets_parquet()

    if "lang" not in df.columns:
        log.warning("No existe columna lang")
        return

    lang_dist = (
        df.groupby("lang")
        .size()
        .reset_index(name="total_tweets")
        .sort_values("total_tweets", ascending=False)
    )

    write_gold(
        lang_dist,
        "tweets_language_distribution"
    )


# ──────────────────────────────────────────────
# GOLD — Engagement Score
# ──────────────────────────────────────────────
def build_engagement_metrics():
    """
    Calcula engagement total por tweet.
    """
    df = load_latest_tweets_parquet()

    required_cols = [
        "likeCount",
        "retweetCount",
        "replyCount",
        "quoteCount",
    ]

    for col in required_cols:
        if col not in df.columns:
            log.warning("Falta columna %s", col)
            return

    df["engagement_score"] = (
        df["likeCount"]
        + df["retweetCount"] * 2
        + df["replyCount"] * 2
        + df["quoteCount"] * 3
    )

    cols = [
        "id",
        "text",
        "author_userName",
        "lang",
        "engagement_score",
        "viewCount",
        "createdAt",
    ]

    cols = [c for c in cols if c in df.columns]

    engagement_df = (
        df[cols]
        .sort_values("engagement_score", ascending=False)
    )

    write_gold(
        engagement_df,
        "engagement_metrics"
    )


# ──────────────────────────────────────────────
# Wrapper task
# ──────────────────────────────────────────────
def process_gold_twitter(**context):
    build_daily_tweet_metrics()
    build_top_authors()
    build_language_distribution()
    build_engagement_metrics()


# ══════════════════════════════════════════════
# DAG
# ══════════════════════════════════════════════
with DAG(
    dag_id="gold_twitter_analytics_dag",
    description="Genera datasets GOLD analíticos desde Silver",
    default_args=DEFAULT_ARGS,
    schedule="30 7 * * 1,4",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["gold", "analytics", "twitter"],
) as dag:

    t1_gold_twitter = PythonOperator(
        task_id="build_gold_twitter_analytics",
        python_callable=process_gold_twitter,
    )

    t1_gold_twitter