from __future__ import annotations

import ast
import logging
import os
import re
from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path

import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml.feature import NGram, StopWordsRemover
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, DoubleType, StringType

try:
    from airflow import DAG
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:  # pragma: no cover - allows local smoke tests without Airflow installed.
    class DAG:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False


    class PythonOperator:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

log = logging.getLogger(__name__)

SILVER_BASE_PATH = os.getenv("SILVER_BASE_PATH", "/opt/airflow/datalake_silver")
GOLD_BASE_PATH = os.getenv("GOLD_BASE_PATH", "/opt/airflow/datalake_gold")
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

DEFAULT_ARGS = {
    "owner": "data-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "love",
    "like",
    "happy",
    "positive",
    "best",
    "win",
    "success",
    "perfect",
    "bonito",
    "bueno",
    "excelente",
    "genial",
    "feliz",
    "mejor",
    "gracias",
    "positivo",
}

NEGATIVE_WORDS = {
    "bad",
    "worse",
    "worst",
    "poor",
    "hate",
    "sad",
    "negative",
    "fail",
    "error",
    "problem",
    "issue",
    "terrible",
    "awful",
    "malo",
    "peor",
    "falla",
    "fallo",
    "problema",
    "triste",
    "negativo",
}

CUSTOM_STOPWORDS = [
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "con",
    "de",
    "del",
    "el",
    "en",
    "es",
    "for",
    "from",
    "http",
    "https",
    "i",
    "la",
    "las",
    "le",
    "lo",
    "los",
    "me",
    "mi",
    "my",
    "na",
    "no",
    "not",
    "of",
    "on",
    "or",
    "para",
    "pero",
    "por",
    "que",
    "rt",
    "si",
    "sin",
    "the",
    "to",
    "un",
    "una",
    "y",
]


def build_spark_session() -> SparkSession:
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    return (
        SparkSession.builder.appName("silver_to_gold_workshop_3")
        .master(SPARK_MASTER)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.parquet.mergeSchema", "true")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )


def load_silver_folder(spark: SparkSession) -> DataFrame:
    silver_path = Path(SILVER_BASE_PATH)
    if not silver_path.exists():
        raise FileNotFoundError(f"Silver path not found: {silver_path}")

    try:
        return (
            spark.read.option("recursiveFileLookup", "true")
            .option("mergeSchema", "true")
            .parquet(str(silver_path))
            .withColumn("source_file", F.input_file_name())
        )
    except Exception as exc:
        log.warning("Spark could not read Silver Parquet directly; falling back to pandas/pyarrow. Error: %s", exc)

    tweet_columns = [
        "type",
        "id",
        "text",
        "source",
        "retweetCount",
        "replyCount",
        "likeCount",
        "quoteCount",
        "viewCount",
        "createdAt",
        "lang",
        "isReply",
        "author_userName",
        "author_name",
        "author_followers",
        "author_isVerified",
        "author_isBlueVerified",
        "author_location",
        "snapshot_date",
    ]
    news_columns = [
        "newsLink",
        "comments",
        "_id",
        "snapshot_id",
        "snapshot_date",
    ]

    tweet_frames: list[pd.DataFrame] = []
    news_frames: list[pd.DataFrame] = []
    for parquet_file in sorted(silver_path.glob("*.parquet")):
        file_name = parquet_file.name.lower()
        if "tweets" in file_name:
            source_type = "tweets"
            columns = tweet_columns
        elif "noticias" in file_name:
            source_type = "news"
            columns = news_columns
        else:
            continue

        pdf = pd.read_parquet(parquet_file)
        selected = [column for column in columns if column in pdf.columns]
        pdf = pdf[selected].copy()
        pdf["source_type"] = source_type
        pdf["source_file"] = parquet_file.name
        if source_type == "tweets":
            for numeric_column in ["retweetCount", "replyCount", "likeCount", "quoteCount", "viewCount", "author_followers"]:
                if numeric_column in pdf.columns:
                    pdf[numeric_column] = pd.to_numeric(pdf[numeric_column], errors="coerce")
            for text_column in [column for column in pdf.columns if column not in {"retweetCount", "replyCount", "likeCount", "quoteCount", "viewCount", "author_followers"}]:
                pdf[text_column] = pdf[text_column].astype("string")
            tweet_frames.append(pdf)
        else:
            for text_column in pdf.columns:
                pdf[text_column] = pdf[text_column].astype("string")
            news_frames.append(pdf)

    if not tweet_frames and not news_frames:
        raise FileNotFoundError(f"No compatible Silver Parquet files found in {silver_path}")

    tweet_spark = spark.createDataFrame(pd.concat(tweet_frames, ignore_index=True, sort=False)) if tweet_frames else None
    news_spark = spark.createDataFrame(pd.concat(news_frames, ignore_index=True, sort=False)) if news_frames else None

    if tweet_spark is not None and news_spark is not None:
        return tweet_spark.unionByName(news_spark, allowMissingColumns=True)
    return tweet_spark or news_spark


def safe_write_parquet(df: DataFrame, output_dir: Path) -> str:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    df.coalesce(1).write.mode("overwrite").parquet(str(output_dir))
    return str(output_dir)


# UDF placeholders — actual UDFs are registered at runtime inside `run_gold_pipeline`
parse_comments_udf = None
sentiment_score_udf = None


def required_fields_ok(required_fields: list[str]) -> F.Column:
    return reduce(lambda acc, field: acc & F.col(field).isNotNull(), required_fields, F.lit(True))


def compute_outlier_rows(df: DataFrame, value_col: str, source_name: str, generated_at: str) -> list[dict]:
    quantiles = df.select(
        F.percentile_approx(F.col(value_col), 0.25).alias("q1"),
        F.percentile_approx(F.col(value_col), 0.75).alias("q3"),
        F.count(F.lit(1)).alias("sample_size"),
    ).first()

    q1 = quantiles["q1"]
    q3 = quantiles["q3"]
    sample_size = int(quantiles["sample_size"] or 0)
    if q1 is None or q3 is None or sample_size == 0:
        return [
            {
                "generated_at": generated_at,
                "section": "outlier_rate",
                "source_type": source_name,
                "metric_name": "outlier_rate",
                "metric_label": value_col,
                "period": None,
                "metric_value": 0.0,
                "metric_pct": 0.0,
                "sample_size": sample_size,
                "notes": "insufficient data",
            }
        ]

    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    outliers = df.filter((F.col(value_col) < lower) | (F.col(value_col) > upper)).count()
    outlier_rate = round((outliers / sample_size) * 100, 2) if sample_size else 0.0

    return [
        {
            "generated_at": generated_at,
            "section": "outlier_rate",
            "source_type": source_name,
            "metric_name": "outlier_rate",
            "metric_label": value_col,
            "period": None,
            "metric_value": float(outliers),
            "metric_pct": float(outlier_rate),
            "sample_size": sample_size,
            "notes": f"IQR bounds [{lower}, {upper}]",
        }
    ]


def normalize_event_date(df: DataFrame) -> DataFrame:
    snapshot_date_value = F.col("snapshot_date").cast("string")
    snapshot_date_ts = F.when(
        snapshot_date_value.rlike(r"^[0-9]+$"),
        F.to_timestamp(F.from_unixtime(snapshot_date_value.cast("double") / 1000.0)),
    ).otherwise(F.to_timestamp(snapshot_date_value))

    if "createdAt" in df.columns:
        created_at_ts = F.coalesce(
            F.to_timestamp(F.col("createdAt"), "EEE MMM dd HH:mm:ss Z yyyy"),
            F.to_timestamp(F.col("createdAt")),
        )
        return df.withColumn(
            "event_date",
            F.coalesce(
                F.to_date(created_at_ts),
                F.to_date(snapshot_date_ts),
            ),
        )
    return df.withColumn("event_date", F.to_date(snapshot_date_ts))


def build_governance_summary(spark: SparkSession, silver_df: DataFrame, generated_at: str) -> DataFrame:
    tweets_df = silver_df.filter(F.col("id").isNotNull())
    news_df = silver_df.filter(F.col("newsLink").isNotNull())

    rows: list[dict] = []

    for source_name, source_df, required_cols, duplicate_key in [
        ("tweets", tweets_df, ["id", "text", "createdAt", "lang", "author_userName"], "id"),
        ("news", news_df, ["newsLink", "comments", "snapshot_date"], "newsLink"),
    ]:
        total_records = source_df.count()
        distinct_records = source_df.select(F.countDistinct(F.col(duplicate_key)).alias("distinct_records")).first()["distinct_records"] if total_records else 0
        duplicate_rows = max(total_records - int(distinct_records or 0), 0)
        duplicate_rate = round((duplicate_rows / total_records) * 100, 2) if total_records else 0.0
        schema_ok = source_df.filter(required_fields_ok(required_cols)).count()
        schema_rate = round((schema_ok / total_records) * 100, 2) if total_records else 0.0

        rows.extend(
            [
                {
                    "generated_at": generated_at,
                    "section": "summary",
                    "source_type": source_name,
                    "metric_name": "total_records",
                    "metric_label": "total_records",
                    "period": None,
                    "metric_value": float(total_records),
                    "metric_pct": None,
                    "sample_size": total_records,
                    "notes": None,
                },
                {
                    "generated_at": generated_at,
                    "section": "summary",
                    "source_type": source_name,
                    "metric_name": "duplicate_rows",
                    "metric_label": duplicate_key,
                    "period": None,
                    "metric_value": float(duplicate_rows),
                    "metric_pct": duplicate_rate,
                    "sample_size": total_records,
                    "notes": None,
                },
                {
                    "generated_at": generated_at,
                    "section": "summary",
                    "source_type": source_name,
                    "metric_name": "schema_compliance_rate",
                    "metric_label": ",".join(required_cols),
                    "period": None,
                    "metric_value": float(schema_ok),
                    "metric_pct": schema_rate,
                    "sample_size": total_records,
                    "notes": None,
                },
            ]
        )

        for field_name in required_cols:
            null_count = source_df.filter(F.col(field_name).isNull() | (F.trim(F.col(field_name)) == "")).count()
            null_rate = round((null_count / total_records) * 100, 2) if total_records else 0.0
            rows.append(
                {
                    "generated_at": generated_at,
                    "section": "null_rates",
                    "source_type": source_name,
                    "metric_name": "null_rate",
                    "metric_label": field_name,
                    "period": None,
                    "metric_value": float(null_count),
                    "metric_pct": float(null_rate),
                    "sample_size": total_records,
                    "notes": None,
                }
            )

        volume_rows = [
            {
                "generated_at": generated_at,
                "section": "volume_trend",
                "source_type": source_name,
                "metric_name": "records_per_day",
                "metric_label": source_name,
                "period": row["period"],
                "metric_value": float(row["metric_value"]),
                "metric_pct": None,
                "sample_size": total_records,
                "notes": None,
            }
            for row in (
                normalize_event_date(source_df)
                .withColumn("period", F.date_format(F.col("event_date"), "yyyy-MM-dd"))
                .groupBy("period")
                .agg(F.count(F.lit(1)).alias("metric_value"))
                .orderBy("period")
                .collect()
            )
        ]
        rows.extend(volume_rows)

        if source_name == "tweets":
            text_stats = (
                source_df.withColumn("text_length", F.length(F.coalesce(F.col("text"), F.lit(""))))
                .select(
                    F.round(F.avg("text_length"), 2).alias("avg_length"),
                    F.percentile_approx(F.col("text_length"), 0.5).alias("median_length"),
                    F.min("text_length").alias("min_length"),
                    F.max("text_length").alias("max_length"),
                    F.count(F.lit(1)).alias("sample_size"),
                )
                .first()
            )
            rows.append(
                {
                    "generated_at": generated_at,
                    "section": "text_stats",
                    "source_type": source_name,
                    "metric_name": "text_length",
                    "metric_label": "tweets_text",
                    "period": None,
                    "metric_value": float(text_stats["avg_length"] or 0.0),
                    "metric_pct": None,
                    "sample_size": int(text_stats["sample_size"] or 0),
                    "notes": f"median={text_stats['median_length']}, min={text_stats['min_length']}, max={text_stats['max_length']}",
                }
            )

            lang_rows = [
                {
                    "generated_at": generated_at,
                    "section": "language_distribution",
                    "source_type": source_name,
                    "metric_name": "language_count",
                    "metric_label": row["lang_label"] or "unknown",
                    "period": None,
                    "metric_value": float(row["language_count"]),
                    "metric_pct": round((row["language_count"] / total_records) * 100, 2) if total_records else 0.0,
                    "sample_size": total_records,
                    "notes": None,
                }
                for row in (
                    source_df.groupBy(F.coalesce(F.col("lang"), F.lit("unknown")).alias("lang_label"))
                    .agg(F.count(F.lit(1)).alias("language_count"))
                    .collect()
                )
            ]
            rows.extend(lang_rows)

            for numeric_column in ["retweetCount", "replyCount", "likeCount", "quoteCount", "viewCount"]:
                if numeric_column in source_df.columns:
                    numeric_df = source_df.withColumn(numeric_column, F.col(numeric_column).cast("double")).filter(F.col(numeric_column).isNotNull())
                    rows.extend(compute_outlier_rows(numeric_df, numeric_column, source_name, generated_at))

        if source_name == "news":
            news_counts = source_df.withColumn("comment_count", F.size(parse_comments_udf(F.col("comments"))))
            text_stats = (
                news_counts.select(
                    F.round(F.avg("comment_count"), 2).alias("avg_length"),
                    F.percentile_approx(F.col("comment_count"), 0.5).alias("median_length"),
                    F.min("comment_count").alias("min_length"),
                    F.max("comment_count").alias("max_length"),
                    F.count(F.lit(1)).alias("sample_size"),
                )
                .first()
            )
            rows.append(
                {
                    "generated_at": generated_at,
                    "section": "text_stats",
                    "source_type": source_name,
                    "metric_name": "comment_count",
                    "metric_label": "news_comments",
                    "period": None,
                    "metric_value": float(text_stats["avg_length"] or 0.0),
                    "metric_pct": None,
                    "sample_size": int(text_stats["sample_size"] or 0),
                    "notes": f"median={text_stats['median_length']}, min={text_stats['min_length']}, max={text_stats['max_length']}",
                }
            )
            rows.extend(compute_outlier_rows(news_counts.filter(F.col("comment_count").isNotNull()), "comment_count", source_name, generated_at))

    return spark.createDataFrame(rows)


def build_storytelling_summary(spark: SparkSession, silver_df: DataFrame, generated_at: str) -> DataFrame:
    tweets_df = silver_df.filter(F.col("id").isNotNull())
    news_df = silver_df.filter(F.col("newsLink").isNotNull())

    # Normalize dates robustly (use the same logic as governance) before selecting
    tweets_norm = normalize_event_date(tweets_df) if "createdAt" in tweets_df.columns or "snapshot_date" in tweets_df.columns else tweets_df
    news_norm = normalize_event_date(news_df) if "snapshot_date" in news_df.columns else news_df

    tweets_text = tweets_norm.select(
        F.lit("tweets").alias("source_type"),
        F.col("id").alias("record_id"),
        F.col("event_date"),
        F.col("text").cast("string").alias("text_body"),
    )

    news_comments = news_norm.select(
        F.lit("news").alias("source_type"),
        F.col("newsLink").alias("record_id"),
        F.col("event_date"),
        F.explode_outer(parse_comments_udf(F.col("comments"))).alias("text_body"),
    )

    text_df = tweets_text.unionByName(news_comments, allowMissingColumns=True).filter(F.col("text_body").isNotNull() & (F.trim(F.col("text_body")) != ""))
    text_df = text_df.withColumn("text_length", F.length(F.col("text_body")))
    text_df = text_df.withColumn("sentiment_score", sentiment_score_udf(F.col("text_body")))
    text_df = text_df.withColumn(
        "sentiment_label",
        F.when(F.col("sentiment_score") > 0.05, F.lit("positive"))
        .when(F.col("sentiment_score") < -0.05, F.lit("negative"))
        .otherwise(F.lit("neutral")),
    )

    rows: list[dict] = []
    total_text_records = text_df.count()

    overall_sentiment = text_df.groupBy("sentiment_label").agg(
        F.count(F.lit(1)).alias("sentiment_count"),
        F.round(F.avg("sentiment_score"), 4).alias("avg_sentiment"),
    ).collect()
    rows.extend(
        [
            {
                "generated_at": generated_at,
                "section": "sentiment_distribution",
                "source_type": "all",
                "metric_name": "sentiment_count",
                "metric_label": row["sentiment_label"],
                "period": None,
                "metric_value": float(row["sentiment_count"]),
                "metric_pct": round((row["sentiment_count"] / total_text_records) * 100, 2) if total_text_records else 0.0,
                "sample_size": total_text_records,
                "notes": f"avg_sentiment={row['avg_sentiment']}",
            }
            for row in overall_sentiment
        ]
    )

    rows.extend(
        [
            {
                "generated_at": generated_at,
                "section": "sentiment_trend",
                "source_type": row["source_type"],
                "metric_name": "avg_daily_sentiment",
                "metric_label": None,
                "period": row["event_date"].isoformat() if row["event_date"] else None,
                "metric_value": float(row["avg_sentiment"]),
                "metric_pct": None,
                "sample_size": int(row["n_records"]),
                "notes": None,
            }
            for row in text_df.groupBy("source_type", "event_date").agg(
                F.round(F.avg("sentiment_score"), 4).alias("avg_sentiment"),
                F.count(F.lit(1)).alias("n_records"),
            ).orderBy("source_type", "event_date").collect()
        ]
    )

    rows.extend(
        [
            {
                "generated_at": generated_at,
                "section": "source_comparison",
                "source_type": row["source_type"],
                "metric_name": "avg_sentiment",
                "metric_label": "all_text",
                "period": None,
                "metric_value": float(row["avg_sentiment"]),
                "metric_pct": None,
                "sample_size": int(row["n_records"]),
                "notes": None,
            }
            for row in text_df.groupBy("source_type").agg(
                F.round(F.avg("sentiment_score"), 4).alias("avg_sentiment"),
                F.count(F.lit(1)).alias("n_records"),
            ).collect()
        ]
    )

    rows.extend(
        [
            {
                "generated_at": generated_at,
                "section": "volume_trend",
                "source_type": row["source_type"],
                "metric_name": "records_per_day",
                "metric_label": None,
                "period": row["event_date"].isoformat() if row["event_date"] else None,
                "metric_value": float(row["n_records"]),
                "metric_pct": None,
                "sample_size": total_text_records,
                "notes": None,
            }
            for row in text_df.groupBy("source_type", "event_date").agg(F.count(F.lit(1)).alias("n_records")).orderBy("source_type", "event_date").collect()
        ]
    )

    clean_tokens = F.split(
        F.regexp_replace(
            F.regexp_replace(
                F.regexp_replace(
                    F.lower(F.col("text_body")),
                    r"http\S+|www\S+|<[^>]+>|[@#]\w+",
                    " ",
                ),
                r"[^a-zA-ZÀ-ÿ0-9 ]",
                " ",
            ),
            r"\s+",
            " ",
        ),
        r"\s+",
    )

    token_df = text_df.select("source_type", "sentiment_score", clean_tokens.alias("tokens"))
    stop_words = sorted(set(StopWordsRemover.loadDefaultStopWords("english")).union(CUSTOM_STOPWORDS))
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words, caseSensitive=False)
    filtered_df = remover.transform(token_df).withColumn("keyword", F.explode("filtered_tokens")).filter((F.length("keyword") > 2) & F.col("keyword").isNotNull())

    rows.extend(
        [
            {
                "generated_at": generated_at,
                "section": "top_keywords",
                "source_type": row["source_type"],
                "metric_name": "keyword_frequency",
                "metric_label": row["keyword"],
                "period": None,
                "metric_value": float(row["keyword_count"]),
                "metric_pct": None,
                "sample_size": total_text_records,
                "notes": None,
            }
            for row in filtered_df.groupBy("source_type", "keyword").agg(F.count(F.lit(1)).alias("keyword_count")).orderBy(F.col("keyword_count").desc(), F.col("keyword").asc()).limit(25).collect()
        ]
    )

    rows.extend(
        [
            {
                "generated_at": generated_at,
                "section": "keyword_sentiment",
                "source_type": row["source_type"],
                "metric_name": "avg_sentiment_by_keyword",
                "metric_label": row["keyword"],
                "period": None,
                "metric_value": float(row["avg_sentiment"]),
                "metric_pct": None,
                "sample_size": int(row["mentions"]),
                "notes": None,
            }
            for row in filtered_df.groupBy("source_type", "keyword").agg(
                F.round(F.avg("sentiment_score"), 4).alias("avg_sentiment"),
                F.count(F.lit(1)).alias("mentions"),
            ).orderBy(F.col("mentions").desc(), F.col("keyword").asc()).limit(25).collect()
        ]
    )

    bigram_df = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words, caseSensitive=False).transform(token_df)
    bigram_tokens = NGram(n=2, inputCol="filtered_tokens", outputCol="bigrams").transform(bigram_df).withColumn("bigram", F.explode("bigrams")).filter(F.trim(F.col("bigram")) != "")

    rows.extend(
        [
            {
                "generated_at": generated_at,
                "section": "top_bigrams",
                "source_type": row["source_type"],
                "metric_name": "bigram_frequency",
                "metric_label": row["bigram"],
                "period": None,
                "metric_value": float(row["bigram_count"]),
                "metric_pct": None,
                "sample_size": total_text_records,
                "notes": None,
            }
            for row in bigram_tokens.groupBy("source_type", "bigram").agg(F.count(F.lit(1)).alias("bigram_count")).orderBy(F.col("bigram_count").desc(), F.col("bigram").asc()).limit(25).collect()
        ]
    )

    rows.extend(
        [
            {
                "generated_at": generated_at,
                "section": "topic_insights",
                "source_type": row["source_type"],
                "metric_name": "top_topic_term",
                "metric_label": row["keyword"],
                "period": None,
                "metric_value": float(row["keyword_count"]),
                "metric_pct": None,
                "sample_size": total_text_records,
                "notes": "proxy topic term",
            }
            for row in filtered_df.groupBy("source_type", "keyword").agg(F.count(F.lit(1)).alias("keyword_count")).orderBy(F.col("keyword_count").desc(), F.col("keyword").asc()).limit(10).collect()
        ]
    )

    return spark.createDataFrame(rows)


def run_gold_pipeline() -> None:
    spark = build_spark_session()
    # Ensure helper module is available to Spark workers and register UDFs
    try:
        spark.sparkContext.addPyFile("/opt/airflow/dags/gold_helpers.py")
    except Exception:
        log.warning("Could not add gold_helpers.py with addPyFile; proceeding (single-node may already have access).")
    try:
        import importlib

        gold_helpers = importlib.import_module("gold_helpers")
        global parse_comments_udf, sentiment_score_udf
        parse_comments_udf = F.udf(gold_helpers.parse_comments, ArrayType(StringType()))
        sentiment_score_udf = F.udf(gold_helpers.sentiment_score, DoubleType())
    except Exception as exc:  # pragma: no cover - robust fallback
        log.warning("Failed to import/register gold_helpers UDFs: %s", exc)
    try:
        silver_df = load_silver_folder(spark)
        generated_at = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        governance_df = build_governance_summary(spark, silver_df, generated_at)
        storytelling_df = build_storytelling_summary(spark, silver_df, generated_at)

        gold_dir = Path(GOLD_BASE_PATH)
        governance_path = gold_dir / f"governance_{generated_at}.parquet"
        storytelling_path = gold_dir / f"storytelling_{generated_at}.parquet"

        safe_write_parquet(governance_df, governance_path)
        safe_write_parquet(storytelling_df, storytelling_path)

        log.info("Gold governance written to %s", governance_path)
        log.info("Gold storytelling written to %s", storytelling_path)
    finally:
        spark.stop()


with DAG(
    dag_id="gold_processing_dag",
    default_args=DEFAULT_ARGS,
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["gold", "pyspark"],
    description="Reads Silver Parquet files with PySpark and produces governance and storytelling Gold summaries.",
) as dag:
    generate_gold_outputs = PythonOperator(
        task_id="generate_gold_outputs",
        python_callable=run_gold_pipeline,
    )
