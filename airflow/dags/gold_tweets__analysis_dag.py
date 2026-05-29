from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

log = logging.getLogger(__name__)

# Configuración
SILVER_BASE_PATH = os.getenv("SILVER_BASE_PATH", "/opt/airflow/datalake_silver")
GOLD_BASE_PATH   = os.getenv("GOLD_BASE_PATH",   "/opt/airflow/datalake_gold")

DEFAULT_ARGS = {
    "owner":            "data-team",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}
# Helpers
def read_silver(prefix: str) -> pd.DataFrame:
    """Lee todos los parquet de silver con el prefijo dado."""
    silver_dir = Path(SILVER_BASE_PATH)
    files = sorted(silver_dir.glob(f"{prefix}_processed_*.parquet"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No hay archivos silver con prefijo '{prefix}' en {silver_dir}")
    # Lee el más reciente
    df = pd.read_parquet(files[0])
    log.info("Silver leído: %s — %d filas", files[0].name, len(df))
    return df


def write_gold(df: pd.DataFrame, name: str) -> str:
    """Escribe un DataFrame como parquet en gold."""
    gold_dir = Path(GOLD_BASE_PATH)
    gold_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest = gold_dir / f"{name}_{timestamp}.parquet"
    df.to_parquet(dest, index=False, engine="pyarrow")
    log.info("Gold escrito: %s — %d filas", dest.name, len(df))
    return str(dest)


# GOVERNANCE KPIs — Noticias
def governance_noticias():
    import ast

    df = read_silver("noticias")

    # Parsear comments
    def parse_comments(val):
        try:
            return ast.literal_eval(val) if isinstance(val, str) else val
        except Exception:
            return []

    df["comments_list"] = df["comments"].apply(parse_comments)
    df["comment_count"] = df["comments_list"].apply(len)

    # ── KPI 1: Tasa de duplicados por newsLink
    total_rows       = len(df)
    duplicate_rows   = df.duplicated(subset=["newsLink"]).sum()
    duplicate_rate   = round(duplicate_rows / total_rows * 100, 2) if total_rows else 0

    # ── KPI 2: Tasa de noticias sin comentarios
    no_comments      = (df["comment_count"] == 0).sum()
    no_comments_rate = round(no_comments / total_rows * 100, 2) if total_rows else 0

    # ── KPI 3: Distribución de comentarios por noticia
    comment_stats = df["comment_count"].describe().rename("comment_count_stats")

    # ── KPI 4: Noticias únicas
    unique_links = df["newsLink"].nunique()

    # ── Construir resumen governance
    gov_df = pd.DataFrame([{
        "snapshot_date":        df["snapshot_date"].iloc[0] if "snapshot_date" in df.columns else None,
        "total_news":           total_rows,
        "unique_news":          unique_links,
        "duplicate_rows":       int(duplicate_rows),
        "duplicate_rate_pct":   duplicate_rate,
        "news_no_comments":     int(no_comments),
        "no_comments_rate_pct": no_comments_rate,
        "avg_comments":         round(df["comment_count"].mean(), 2),
        "max_comments":         int(df["comment_count"].max()),
        "min_comments":         int(df["comment_count"].min()),
        "std_comments":         round(df["comment_count"].std(), 2),
        "total_comments":       int(df["comment_count"].sum()),
    }])

    write_gold(gov_df, "governance_noticias")
    log.info("Governance noticias completado.")

# GOVERNANCE KPIs — Tweets
def governance_tweets():
    TWEET_COLS = [
        "id", "text", "createdAt", "lang", "source", "snapshot_date",
        "retweetCount", "replyCount", "likeCount", "quoteCount", "viewCount",
        "isReply", "author_userName", "author_followers", "author_isVerified",
    ]

    df = read_silver("tweets")
    cols = [c for c in TWEET_COLS if c in df.columns]
    df   = df[cols].copy()

    total_rows     = len(df)
    duplicate_rows = df.duplicated(subset=["id"]).sum() if "id" in df.columns else 0
    duplicate_rate = round(duplicate_rows / total_rows * 100, 2) if total_rows else 0

    # Nulos por columna relevante
    null_rates = {
        f"null_rate_{c}": round(df[c].isna().sum() / total_rows * 100, 2)
        for c in ["text", "createdAt", "lang", "author_userName"]
        if c in df.columns
    }

    # Métricas de engagement
    engagement_cols = ["retweetCount", "replyCount", "likeCount", "quoteCount", "viewCount"]
    engagement_stats = {}
    for col in engagement_cols:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            engagement_stats[f"avg_{col}"] = round(numeric.mean(), 2)
            engagement_stats[f"max_{col}"] = int(numeric.max()) if not numeric.isna().all() else 0

    # Idiomas
    lang_dist = df["lang"].value_counts().to_dict() if "lang" in df.columns else {}

    gov_df = pd.DataFrame([{
        "snapshot_date":      df["snapshot_date"].iloc[0] if "snapshot_date" in df.columns else None,
        "total_tweets":       total_rows,
        "duplicate_rows":     int(duplicate_rows),
        "duplicate_rate_pct": duplicate_rate,
        "unique_langs":       len(lang_dist),
        "top_lang":           max(lang_dist, key=lang_dist.get) if lang_dist else None,
        **null_rates,
        **engagement_stats,
    }])

    write_gold(gov_df, "governance_tweets")
    log.info("Governance tweets completado.")


# STORYTELLING — Noticias
def storytelling_noticias():
    import ast

    df = read_silver("noticias")

    def parse_comments(val):
        try:
            return ast.literal_eval(val) if isinstance(val, str) else val
        except Exception:
            return []

    df["comments_list"] = df["comments"].apply(parse_comments)
    df["comment_count"] = df["comments_list"].apply(len)

    # ── Aggregation 1: Noticias más comentadas
    top_news = (
        df[["newsLink", "comment_count"]]
        .sort_values("comment_count", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_news["rank"] = top_news.index + 1
    write_gold(top_news, "storytelling_top_news")

    # ── Aggregation 2: Explotar comentarios para análisis de texto
    df_exploded = df.explode("comments_list").reset_index(drop=True)
    df_exploded = df_exploded.rename(columns={"comments_list": "comment"})
    df_exploded["comment_length"] = df_exploded["comment"].astype(str).apply(len)

    # Distribución de longitud de comentarios
    length_stats = df_exploded.groupby("newsLink")["comment_length"].agg(
        avg_length="mean",
        max_length="max",
        min_length="min",
        total_comments="count"
    ).reset_index()
    length_stats["avg_length"] = length_stats["avg_length"].round(2)
    write_gold(length_stats, "storytelling_comment_length")

    # ── Aggregation 3: Todos los comentarios expandidos (para NLP futuro)
    df_comments_flat = df_exploded[["newsLink", "comment", "comment_length", "snapshot_date"]].copy() \
        if "snapshot_date" in df_exploded.columns \
        else df_exploded[["newsLink", "comment", "comment_length"]].copy()
    write_gold(df_comments_flat, "storytelling_comments_flat")

    log.info("Storytelling noticias completado.")



# STORYTELLING — Tweets

def storytelling_tweets():
    TWEET_COLS = [
        "id", "text", "createdAt", "lang", "snapshot_date",
        "retweetCount", "replyCount", "likeCount", "quoteCount", "viewCount",
        "isReply", "author_userName", "author_name",
        "author_followers", "author_isVerified", "author_isBlueVerified",
        "author_location",
    ]

    df = read_silver("tweets")
    cols = [c for c in TWEET_COLS if c in df.columns]
    df   = df[cols].copy()

    # Convertir métricas a numérico
    engagement_cols = ["retweetCount", "replyCount", "likeCount", "quoteCount", "viewCount"]
    for col in engagement_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── Aggregation 1: Top tweets por engagement total
    df["engagement_total"] = df[
        [c for c in engagement_cols if c in df.columns]
    ].sum(axis=1)

    top_tweets = (
        df[["id", "text", "author_userName", "engagement_total", "likeCount", "retweetCount"]]
        .sort_values("engagement_total", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    top_tweets["rank"] = top_tweets.index + 1
    write_gold(top_tweets, "storytelling_top_tweets")

    # ── Aggregation 2: Engagement promedio por idioma
    if "lang" in df.columns:
        lang_engagement = df.groupby("lang").agg(
            tweet_count=("id", "count"),
            avg_likes=("likeCount", "mean"),
            avg_retweets=("retweetCount", "mean"),
            avg_engagement=("engagement_total", "mean"),
        ).reset_index()
        lang_engagement = lang_engagement.round(2)
        write_gold(lang_engagement, "storytelling_engagement_by_lang")

    # ── Aggregation 3: Top autores por engagement
    if "author_userName" in df.columns:
        top_authors = df.groupby("author_userName").agg(
            tweet_count=("id", "count"),
            total_engagement=("engagement_total", "sum"),
            avg_engagement=("engagement_total", "mean"),
            avg_followers=("author_followers", "mean"),
        ).reset_index().sort_values("total_engagement", ascending=False).head(20)
        top_authors = top_authors.round(2).reset_index(drop=True)
        write_gold(top_authors, "storytelling_top_authors")

    # ── Aggregation 4: Tweets planos para análisis de texto
    df["text_length"] = df["text"].astype(str).apply(len)
    write_gold(df, "storytelling_tweets_flat")

    log.info("Storytelling tweets completado.")



# DAG

with DAG(
    dag_id="gold_processing_dag",
    description="Genera capa Gold: governance KPIs y storytelling summaries",
    default_args=DEFAULT_ARGS,
    schedule="30 7 * * 1,4",  # 30 min después del silver
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["gold", "governance", "storytelling"],
) as dag:

    t_gov_noticias = PythonOperator(
        task_id="governance_noticias",
        python_callable=governance_noticias,
    )

    t_gov_tweets = PythonOperator(
        task_id="governance_tweets",
        python_callable=governance_tweets,
    )

    t_story_noticias = PythonOperator(
        task_id="storytelling_noticias",
        python_callable=storytelling_noticias,
    )

    t_story_tweets = PythonOperator(
        task_id="storytelling_tweets",
        python_callable=storytelling_tweets,
    )

    # Governance primero, luego storytelling
    t_gov_noticias >> t_story_noticias
    t_gov_tweets   >> t_story_tweets