from __future__ import annotations

import logging
import os
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, countDistinct, avg, max, min, 
    sum as spark_sum, when, isnan, isnull, length, to_date, desc,
    from_unixtime, from_json, size, explode, substring
)
from pyspark.sql.types import ArrayType, StringType

log = logging.getLogger(__name__)

# CONFIGURACIÓN
SILVER_BASE_PATH = os.getenv("SILVER_BASE_PATH")
GOLD_BASE_PATH = os.getenv("GOLD_BASE_PATH")

DEFAULT_ARGS = {
    "owner": "data-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def get_spark():
    """
    Configuración de Spark requerida por el Workshop #3.
    """
    return (
        SparkSession.builder
        .appName("gold_layer_processing")
        .master("local[*]") 
        .config("spark.driver.memory", "4g") 
        .config("spark.sql.parquet.inferTimestampNTZ.enabled", "true")
        .getOrCreate()
    )

def write_gold(df, name):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"{GOLD_BASE_PATH}/{name}_{timestamp}.parquet"
    df.coalesce(1).write.mode("overwrite").parquet(path)
    log.info("Archivo Gold guardado en: %s", path)


# 1. LÓGICA DE WEBSCRAPING (NOTICIAS)
def governance_news():
    spark = get_spark()
    files = [str(p) for p in Path(SILVER_BASE_PATH).glob("noticias_processed_*.parquet")]
    if not files:
        log.warning("No hay archivos de noticias en Silver.")
        return
        
    df = spark.read.parquet(*files)
    total_news = df.count()
    unique_news = df.select(countDistinct("newsLink")).collect()[0][0]
    duplicate_rows = total_news - unique_news
    
    # Dado que solo tenemos 'comments' como texto, calcularemos las métricas sobre ellos.
    # Convertimos el string JSON de comentarios a un Array en Spark
    df_with_comments = df.withColumn(
        "comments_array", 
        from_json(col("comments"), ArrayType(StringType()))
    )
    
    # Contamos la cantidad de comentarios por noticia
    df_with_comments = df_with_comments.withColumn("num_comments", size(col("comments_array")))
    
    # Calculamos cuántas noticias no tienen comentarios
    news_without_comments = df_with_comments.filter(col("num_comments") <= 0).count()

    result_df = spark.createDataFrame([(
        total_news,
        unique_news,
        duplicate_rows,
        round((duplicate_rows / total_news * 100), 2) if total_news else 0,
        news_without_comments,
        round((news_without_comments / total_news * 100), 2) if total_news else 0
    )], [
        "total_news", "unique_news", "duplicate_rows", "duplicate_rate_pct",
        "news_without_comments", "no_comments_rate_pct"
    ])
    
    write_gold(result_df, "governance_news")
    spark.stop()


def storytelling_news():
    spark = get_spark()
    files = [str(p) for p in Path(SILVER_BASE_PATH).glob("noticias_processed_*.parquet")]
    if not files: return
    
    df = spark.read.parquet(*files)
    
    #Normalizamos todo a un string con formato de fecha seguro
    safe_date_string = when(
        col("snapshot_date").rlike("^[0-9]+$"), # Si la cadena contiene ÚNICAMENTE números (el timestamp UNIX)
        from_unixtime(col("snapshot_date").cast("double") / 1000)
    ).otherwise(
        substring(col("snapshot_date"), 1, 10) # Si es una fecha ISO (ej. 2026-05-29T...), tomamos solo YYYY-MM-DD
    )

    #Ahora sí, convertimos el string normalizado a Date
    df = df.withColumn("date_day", to_date(safe_date_string))
    
    # Volumen temporal completo (por noticia extraída)
    volume = df.groupBy("date_day").count().orderBy("date_day")
    write_gold(volume, "storytelling_news_daily_volume")
    
    # Base analítica para NLP: Explotamos los comentarios
    # Como no tenemos 'content', el Análisis de Sentimiento se hará sobre los comentarios de los usuarios.
    # Usamos explode() para que cada comentario tenga su propia fila.
    df_comments = df.withColumn("comments_array", from_json(col("comments"), ArrayType(StringType())))
    nlp_base = df_comments.select(
        "newsLink", 
        "date_day",
        explode(col("comments_array")).alias("comment_text")
    ).filter(col("comment_text") != "")
    
    write_gold(nlp_base, "storytelling_news_nlp_base")
    spark.stop()


# 2. LÓGICA DE TWITTER (API X)


def governance_tweets():
    spark = get_spark()
    files = [str(p) for p in Path(SILVER_BASE_PATH).glob("tweets_processed_*.parquet")]
    if not files: return
        
    df = spark.read.parquet(*files)
    total_tweets = df.count()
    unique_tweets = df.select(countDistinct("id")).collect()[0][0]
    duplicate_rows = total_tweets - unique_tweets
    
    # Usuarios verificados
    verified_users = df.filter(col("author_isVerified") == True).count()

    # KPI: Estadísticas de longitud de texto del tweet
    text_stats = df.agg(
        avg(length(col("text"))).alias("avg_tweet_length"),
        max(length(col("text"))).alias("max_tweet_length"),
        min(length(col("text"))).alias("min_tweet_length")
    ).collect()[0]

    result_df = spark.createDataFrame([(
        total_tweets,
        unique_tweets,
        duplicate_rows,
        round((duplicate_rows / total_tweets * 100), 2) if total_tweets else 0,
        verified_users,
        text_stats["avg_tweet_length"],
        text_stats["min_tweet_length"],
        text_stats["max_tweet_length"]
    )], [
        "total_tweets", "unique_tweets", "duplicate_rows", "duplicate_rate_pct",
        "verified_user_tweets", "avg_tweet_length", "min_tweet_length", "max_tweet_length"
    ])
    
    write_gold(result_df, "governance_tweets")
    spark.stop()

def storytelling_tweets():
    spark = get_spark()
    files = [str(p) for p in Path(SILVER_BASE_PATH).glob("tweets_processed_*.parquet")]
    if not files: return
    
    df = spark.read.parquet(*files)
    safe_date_string = when(
        col("snapshot_date").rlike("^[0-9]+$"), # Si la cadena contiene ÚNICAMENTE números (el timestamp UNIX)
        from_unixtime(col("snapshot_date").cast("double") / 1000)
    ).otherwise(
        substring(col("snapshot_date"), 1, 10) # Si es una fecha ISO (ej. 2026-05-29T...), tomamos solo YYYY-MM-DD
    )
    df = df.withColumn("date_day", to_date(safe_date_string))
    
    df = df.withColumn(
        "total_engagement",
        col("likeCount") + col("replyCount") + col("retweetCount") + col("quoteCount")
    )
    
    # Evolución temporal
    volume = df.groupBy("date_day").count().orderBy("date_day")
    write_gold(volume, "storytelling_tweets_daily_volume")
    
    # Distribución de idiomas
    lang_dist = df.groupBy("lang").count().orderBy(desc("count"))
    write_gold(lang_dist, "storytelling_tweets_lang_dist")
    
    # Base analítica completa para Análisis de Sentimientos (Sin limit())
    # Utilizamos 'text_cleaned' asumiendo que agregaste la limpieza NLP en Silver
    text_column = "text_cleaned" if "text_cleaned" in df.columns else "text"
    nlp_base = df.select(
        "id", "date_day", "author_userName", text_column, "total_engagement", "lang", "likeCount", "replyCount", "retweetCount", "quoteCount"
    )
    write_gold(nlp_base, "storytelling_tweets_nlp_base")
    spark.stop()


# 3. DEFINICIÓN DE DAGS

with DAG(
    dag_id="gold_webscraping",
    schedule=None,  # <-- Se mantiene None porque es activado por un TriggerDagRunOperator
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["gold", "webscraping", "nlp"],
    default_args=DEFAULT_ARGS,
) as dag_news:

    t_gov_news = PythonOperator(
        task_id="governance_news",
        python_callable=governance_news,
    )

    t_story_news = PythonOperator(
        task_id="storytelling_news",
        python_callable=storytelling_news,
    )

    t_gov_news >> t_story_news

with DAG(
    dag_id="gold_twitter",
    schedule=None,  # <-- Se mantiene None porque es activado por un TriggerDagRunOperator
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["gold", "twitter", "nlp"],
    default_args=DEFAULT_ARGS,
) as dag_twitter:

    t_gov_tweets = PythonOperator(
        task_id="governance_tweets",
        python_callable=governance_tweets,
    )

    t_story_tweets = PythonOperator(
        task_id="storytelling_tweets",
        python_callable=storytelling_tweets,
    )

    t_gov_tweets >> t_story_tweets