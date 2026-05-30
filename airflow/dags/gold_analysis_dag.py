from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    explode,
    from_json,
    from_unixtime,
    substring,
    to_date,
    weekofyear,
    when,
)
from pyspark.sql.types import ArrayType, StringType

log = logging.getLogger(__name__)


# Environment

SILVER_BASE_PATH = os.getenv("SILVER_BASE_PATH", "/data/silver")
GOLD_BASE_PATH = os.getenv("GOLD_BASE_PATH", "/data/gold")

DEFAULT_ARGS = {
    "owner": "data-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


# Brand / Product / Topic taxonomy

BRANDS = {
    # Phones
    "apple":    ["apple", "iphone", "iphones", "airpods", "macbook", "macbooks"],
    "samsung":  ["samsung", "galaxy"],
    "google":   ["google", "pixel"],
    "xiaomi":   ["xiaomi"],
    "oneplus":  ["oneplus"],
    "motorola": ["motorola"],
    "huawei":   ["huawei"],
    "honor":    ["honor"],
    "realme":   ["realme"],
    "oppo":     ["oppo"],
    "vivo":     ["vivo"],
    # Computers / GPU
    "nvidia":   ["nvidia", "rtx"],
    "amd":      ["amd", "ryzen"],
    "intel":    ["intel"],
    "lenovo":   ["lenovo", "thinkpad"],
    "dell":     ["dell", "alienware"],
    "asus":     ["asus"],
    "acer":     ["acer"],
    "hp":       ["hp"],
    "razer":    ["razer"],
    "msi":      ["msi"],
    "microsoft":["microsoft", "surface"],
    # Audio / Accessories
    "sony":     ["sony"],
    "bose":     ["bose"],
    "jbl":      ["jbl"],
    "anker":    ["anker"],
    "logitech": ["logitech"],
    "corsair":  ["corsair"],
    "steelseries": ["steelseries"],
    # Wearables
    "fitbit":   ["fitbit"],
    "garmin":   ["garmin"],
    "amazfit":  ["amazfit"],
    "oura":     ["oura"],
}

PRODUCTS = {
    # Phones
    "iphone":    ["iphone", "iphones"],
    "galaxy":    ["galaxy"],
    "pixel":     ["pixel"],
    # Computers
    "macbook":   ["macbook", "macbooks"],
    "thinkpad":  ["thinkpad"],
    "surface":   ["surface"],
    # GPU
    "rtx":       ["rtx"],
    "ryzen":     ["ryzen"],
    # Audio
    "airpods":   ["airpods"],
    "earbuds":   ["earbuds"],
    "headphones":["headphones"],
    # Wearables
    "galaxy_watch": ["galaxy watch"],
    "apple_watch":  ["apple watch"],
    "oura_ring":    ["oura ring"],
}

TOPICS = {
    "battery":      ["battery", "battery life", "mah", "charging speed"],
    "camera":       ["camera", "photo", "photography", "video", "megapixel", "lens"],
    "performance":  ["performance", "speed", "fps", "lag", "benchmark", "fast", "slow"],
    "display":      ["display", "screen", "oled", "amoled", "refresh rate", "brightness"],
    "charging":     ["charging", "charger", "fast charge", "wireless charge"],
    "software":     ["software", "android", "ios", "update", "ui", "os", "bug"],
    "price":        ["price", "cost", "expensive", "cheap", "worth it", "value"],
    "audio":        ["audio", "sound", "speaker", "microphone", "anc", "noise cancel"],
    "connectivity": ["bluetooth", "wifi", "5g", "connectivity", "signal"],
    "design":       ["design", "build", "quality", "premium", "plastic", "glass", "weight"],
    # New topics GSM ARENA
    "foldable":     ["fold", "foldable", "flip"],
    "ai":           ["ai", "artificial intelligence", "galaxy ai", "on-device ai"],
    "ev":           ["electric vehicle", "ev", "car", "yu7"],
    "smartwatch":   ["watch", "smartwatch", "wearable"],
}


# VADER singleton
_vader = None

def _get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader

def _batch_sentiment(texts: list[str]) -> tuple[list[str], list[float]]:
    analyzer = _get_vader()
    labels, scores = [], []
    for text in texts:
        compound = analyzer.polarity_scores(str(text))["compound"]
        if compound >= 0.05:
            labels.append("positive")
        elif compound <= -0.05:
            labels.append("negative")
        else:
            labels.append("neutral")
        scores.append(round(abs(compound), 4))
    return labels, scores


# Spark helper

def _get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("weekly_gold_pipeline")
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )

def _safe_date_col():
    return when(
        col("snapshot_date").rlike(r"^[0-9]+$"),
        from_unixtime(col("snapshot_date").cast("double") / 1000),
    ).otherwise(substring(col("snapshot_date"), 1, 10))


# Gold writer helpers

def _gold_path(name: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{GOLD_BASE_PATH}/{name}_{timestamp}.parquet"

def _write_gold_parquet(df: pd.DataFrame, name: str, spark: SparkSession) -> None:
    path = _gold_path(name)
    spark.createDataFrame(df).coalesce(1).write.mode("overwrite").parquet(path)
    log.info("Gold written: %s (%d rows)", path, len(df))

def _write_gold_json(data: dict | list, name: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = Path(GOLD_BASE_PATH) / f"{name}_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("Gold JSON written: %s", path)


# Taxonomy matching helpers

def _match_taxonomy(text: str, taxonomy: dict[str, list[str]]) -> list[str]:
    """Return all keys in taxonomy whose keywords appear in text (lowercased)."""
    text_lower = text.lower()
    return [key for key, kws in taxonomy.items() if any(kw in text_lower for kw in kws)]

def _first_or_none(matches: list[str]) -> str | None:
    return matches[0] if matches else None

def _enrich_df(pdf: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Add brand, product, topic columns based on text content."""
    texts = pdf[text_col].fillna("").astype(str)
    pdf["brand"]   = texts.apply(lambda t: _first_or_none(_match_taxonomy(t, BRANDS)))
    pdf["product"] = texts.apply(lambda t: _first_or_none(_match_taxonomy(t, PRODUCTS)))
    pdf["topic"]   = texts.apply(lambda t: _first_or_none(_match_taxonomy(t, TOPICS)))
    return pdf


# TASK 1 — Governance: Tweets
# Folder: gold/governance/tweets/
# Artifact: governance_tweets_weekly

def process_governance_tweets() -> None:
    spark = _get_spark()
    files = list(Path(SILVER_BASE_PATH).glob("tweets_processed_*.parquet"))
    if not files:
        log.warning("No tweet parquet files found in %s", SILVER_BASE_PATH)
        spark.stop()
        return

    df = spark.read.parquet(*[str(f) for f in files])
    df = (
        df
        .withColumn("date_day", to_date(_safe_date_col()))
        .withColumn("week", weekofyear(col("date_day")))
    )
    current_week = datetime.utcnow().isocalendar()[1]
    df = df.filter(col("week") == current_week)
    df = df.withColumn(
        "total_engagement",
        col("likeCount") + col("replyCount") + col("retweetCount") + col("quoteCount"),
    )

    text_col = "text_cleaned" if "text_cleaned" in df.columns else "text"
    cols = ["id", "date_day", "week", "author_userName", "author_followers",
            text_col, "lang", "total_engagement",
            "likeCount", "replyCount", "retweetCount", "quoteCount"]
    cols_ok = [c for c in cols if c in df.columns]

    pdf = df.select(cols_ok).toPandas()
    pdf = _enrich_df(pdf, text_col)

    labels, scores = _batch_sentiment(pdf[text_col].fillna("").astype(str).tolist())
    pdf["sentiment"]       = labels
    pdf["sentiment_score"] = scores

    # experience_type: simple heuristic
    experience_kws = ["i bought", "i got", "i've been using", "my experience",
                      "daily driver", "worth it", "happy with", "not worth it"]
    pdf["experience_type"] = pdf[text_col].str.lower().apply(
        lambda t: "personal_experience" if any(kw in t for kw in experience_kws) else "opinion"
    )

    # Rename for canonical schema
    pdf = pdf.rename(columns={text_col: "text", "id": "tweet_id"})

    # Write to governance/tweets/ subfolder
    gov_path = f"{GOLD_BASE_PATH}/governance/tweets"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    spark.createDataFrame(pdf).coalesce(1).write.mode("overwrite").parquet(
        f"{gov_path}/governance_tweets_weekly_{ts}.parquet"
    )
    log.info("Governance tweets written: %d rows", len(pdf))
    spark.stop()


# TASK 2 — Governance: News comments
# Folder: gold/governance/news/
# Artifact: governance_news_weekly

def process_governance_news() -> None:
    spark = _get_spark()
    files = list(Path(SILVER_BASE_PATH).glob("noticias_processed_*.parquet"))
    if not files:
        log.warning("No news parquet files found in %s", SILVER_BASE_PATH)
        spark.stop()
        return

    df = spark.read.parquet(*[str(f) for f in files])
    df = (
        df
        .withColumn("date_day", to_date(_safe_date_col()))
        .withColumn("week", weekofyear(col("date_day")))
    )
    current_week = datetime.utcnow().isocalendar()[1]
    df = df.filter(col("week") == current_week)

    df_comments = (
        df
        .withColumn("comments_array", from_json(col("comments"), ArrayType(StringType())))
        .select("newsLink", "date_day", "week", explode("comments_array").alias("comment_text"))
    )

    pdf = df_comments.toPandas()
    if pdf.empty:
        log.warning("No news comments found for current week.")
        spark.stop()
        return

    pdf = _enrich_df(pdf, "comment_text")
    labels, scores = _batch_sentiment(pdf["comment_text"].fillna("").astype(str).tolist())
    pdf["sentiment"]       = labels
    pdf["sentiment_score"] = scores

    gov_path = f"{GOLD_BASE_PATH}/governance/news"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    spark.createDataFrame(pdf).coalesce(1).write.mode("overwrite").parquet(
        f"{gov_path}/governance_news_weekly_{ts}.parquet"
    )
    log.info("Governance news written: %d rows", len(pdf))
    spark.stop()


# Helper: load latest governance parquets as pandas DataFrames

def _load_gov_tweets(spark: SparkSession) -> pd.DataFrame:
    gov_dir = Path(GOLD_BASE_PATH) / "governance" / "tweets"
    files = list(gov_dir.glob("governance_tweets_weekly_*.parquet"))
    if not files:
        raise FileNotFoundError("No governance tweets found. Run process_governance_tweets first.")
    return spark.read.parquet(str(max(files, key=lambda p: p.stat().st_mtime))).toPandas()

def _load_gov_news(spark: SparkSession) -> pd.DataFrame:
    gov_dir = Path(GOLD_BASE_PATH) / "governance" / "news"
    files = list(gov_dir.glob("governance_news_weekly_*.parquet"))
    if not files:
        raise FileNotFoundError("No governance news found. Run process_governance_news first.")
    return spark.read.parquet(str(max(files, key=lambda p: p.stat().st_mtime))).toPandas()

def _sentiment_pcts(sub: pd.DataFrame, score_col: str = "sentiment_score") -> dict:
    total = len(sub)
    if total == 0:
        return {"positive_pct": 0, "negative_pct": 0, "neutral_pct": 0, "avg_score": 0.0}
    counts = sub["sentiment"].value_counts()
    return {
        "positive_pct": round(counts.get("positive", 0) / total * 100, 1),
        "negative_pct": round(counts.get("negative", 0) / total * 100, 1),
        "neutral_pct":  round(counts.get("neutral",  0) / total * 100, 1),
        "avg_score":    round(sub[score_col].mean(), 4),
    }


# TASK 3 — Brand Analytics (Tweets + News unified)
# Artifact: brand_analytics_weekly

def build_brand_analytics() -> None:
    spark = _get_spark()
    tweets = _load_gov_tweets(spark)
    news   = _load_gov_news(spark)
    current_week = datetime.utcnow().isocalendar()[1]

    rows = []
    all_brands = set(tweets["brand"].dropna().unique()) | set(news["brand"].dropna().unique())

    for brand in all_brands:
        tw = tweets[tweets["brand"] == brand]
        nw = news[news["brand"] == brand]
        combined = pd.concat([
            tw[["sentiment", "sentiment_score"]],
            nw[["sentiment", "sentiment_score"]],
        ])
        pcts = _sentiment_pcts(combined)
        rows.append({
            "week":             current_week,
            "brand":            brand,
            "mentions_twitter": len(tw),
            "mentions_news":    len(nw),
            "total_mentions":   len(combined),
            "total_engagement": int(tw["total_engagement"].sum()) if "total_engagement" in tw.columns else 0,
            **pcts,
        })

    pdf_out = pd.DataFrame(rows).sort_values("total_mentions", ascending=False)
    _write_gold_parquet(pdf_out, "brand_analytics_weekly", spark)
    spark.stop()


# TASK 4 — Topic Analytics (Tweets + News unified)
# Artifact: topic_analytics_weekly

def build_topic_analytics() -> None:
    spark = _get_spark()
    tweets = _load_gov_tweets(spark)
    news   = _load_gov_news(spark)
    current_week = datetime.utcnow().isocalendar()[1]

    combined = pd.concat([
        tweets[["topic", "sentiment", "sentiment_score"]],
        news[["topic", "sentiment", "sentiment_score"]],
    ])
    combined = combined.dropna(subset=["topic"])

    rows = []
    for topic in combined["topic"].unique():
        sub = combined[combined["topic"] == topic]
        pcts = _sentiment_pcts(sub)
        rows.append({
            "week":     current_week,
            "topic":    topic,
            "mentions": len(sub),
            **pcts,
        })

    pdf_out = pd.DataFrame(rows).sort_values("mentions", ascending=False)
    _write_gold_parquet(pdf_out, "topic_analytics_weekly", spark)
    spark.stop()


# TASK 5 — Product Analytics
# Artifact: product_analytics_weekly

def build_product_analytics() -> None:
    spark = _get_spark()
    tweets = _load_gov_tweets(spark)
    news   = _load_gov_news(spark)
    current_week = datetime.utcnow().isocalendar()[1]

    combined = pd.concat([
        tweets[["product", "brand", "sentiment", "sentiment_score"]],
        news[["product", "brand", "sentiment", "sentiment_score"]],
    ])
    combined = combined.dropna(subset=["product"])

    rows = []
    for product in combined["product"].unique():
        sub = combined[combined["product"] == product]
        # Use modal brand for this product
        brand = sub["brand"].mode()[0] if not sub["brand"].dropna().empty else None
        pcts = _sentiment_pcts(sub)
        rows.append({
            "week":     current_week,
            "product":  product,
            "brand":    brand,
            "mentions": len(sub),
            **pcts,
        })

    pdf_out = pd.DataFrame(rows).sort_values("mentions", ascending=False)
    _write_gold_parquet(pdf_out, "product_analytics_weekly", spark)
    spark.stop()


# TASK 6 — Top Influencers (Twitter only — news has no author data)
# Artifact: top_influencers_weekly

def build_top_influencers() -> None:
    spark = _get_spark()
    tweets = _load_gov_tweets(spark)
    current_week = datetime.utcnow().isocalendar()[1]

    required = {"author_userName", "author_followers", "total_engagement", "brand", "sentiment"}
    if not required.issubset(tweets.columns):
        log.warning("Missing columns for influencers: %s", required - set(tweets.columns))
        spark.stop()
        return

    tweets = tweets.dropna(subset=["brand"])
    tweets["author_followers"] = pd.to_numeric(tweets["author_followers"], errors="coerce").fillna(0)
    tweets["total_engagement"] = pd.to_numeric(tweets["total_engagement"], errors="coerce").fillna(0)

    # Aggregate by author + brand
    agg = (
        tweets
        .groupby(["brand", "author_userName"], as_index=False)
        .agg(
            followers    = ("author_followers", "max"),
            engagement   = ("total_engagement", "sum"),
            tweet_count  = ("tweet_id", "count"),
            sentiment    = ("sentiment", lambda s: s.mode()[0] if not s.empty else "neutral"),
        )
    )
    agg["week"] = current_week
    # Top 20 by engagement
    pdf_out = agg.sort_values("engagement", ascending=False).head(20)
    _write_gold_parquet(pdf_out, "top_influencers_weekly", spark)
    spark.stop()


# TASK 7 — Representative Comments
# Artifact: representative_comments_weekly

def build_representative_comments() -> None:
    spark = _get_spark()
    tweets = _load_gov_tweets(spark)
    news   = _load_gov_news(spark)
    current_week = datetime.utcnow().isocalendar()[1]

    # Unify text + sentiment + score + brand from both sources
    tw_sub = tweets[["brand", "text", "sentiment", "sentiment_score"]].copy() if "text" in tweets.columns else pd.DataFrame()
    nw_sub = news[["brand", "comment_text", "sentiment", "sentiment_score"]].rename(columns={"comment_text": "text"}).copy()

    combined = pd.concat([tw_sub, nw_sub]).dropna(subset=["brand", "text"])
    combined["text_len"] = combined["text"].str.len()
    # Filter out very short comments (noise)
    combined = combined[combined["text_len"] > 20]

    rows = []
    for brand in combined["brand"].dropna().unique():
        sub = combined[combined["brand"] == brand]
        positives = sub[sub["sentiment"] == "positive"].sort_values("sentiment_score", ascending=False)
        negatives = sub[sub["sentiment"] == "negative"].sort_values("sentiment_score", ascending=False)

        rows.append({
            "week":         current_week,
            "brand":        brand,
            "best_comment": positives.iloc[0]["text"][:300] if not positives.empty else None,
            "worst_comment": negatives.iloc[0]["text"][:300] if not negatives.empty else None,
        })

    pdf_out = pd.DataFrame(rows)
    _write_gold_parquet(pdf_out, "representative_comments_weekly", spark)
    spark.stop()


# TASK 8 — Storytelling

def build_storytelling() -> None:
    """
    Reads brand / topic / product analytics (already computed in previous tasks)
    and produces a single JSON executive summary for the week.
    """
    spark = _get_spark()
    current_week = datetime.utcnow().isocalendar()[1]

    # Load analytics parquets (most recent file for each)
    def _latest(pattern: str) -> pd.DataFrame:
        files = sorted(Path(GOLD_BASE_PATH).glob(pattern))
        if not files:
            return pd.DataFrame()
        return spark.read.parquet(str(files[-1])).toPandas()

    brands   = _latest("brand_analytics_weekly_*.parquet")
    topics   = _latest("topic_analytics_weekly_*.parquet")
    products = _latest("product_analytics_weekly_*.parquet")
    comments = _latest("representative_comments_weekly_*.parquet")

    story: dict = {"week": current_week}

    # --- Brand insights ---
    if not brands.empty:
        top_brand   = brands.sort_values("total_mentions", ascending=False).iloc[0]["brand"]
        best_brand  = brands.sort_values("positive_pct",   ascending=False).iloc[0]["brand"]
        worst_brand = brands.sort_values("negative_pct",   ascending=False).iloc[0]["brand"]
        top_brand_mentions = int(brands.iloc[0]["total_mentions"])

        story["executive_summary"] = (
            f"{str(top_brand).capitalize()} fue la marca más mencionada durante la semana "
            f"con {top_brand_mentions} referencias entre redes sociales y comentarios de noticias."
        )
        story["top_brand"]   = top_brand
        story["best_brand"]  = best_brand
        story["worst_brand"] = worst_brand

        # Brand sentiment snapshot (all brands)
        story["brand_sentiment_snapshot"] = brands[
            ["brand", "total_mentions", "positive_pct", "negative_pct", "neutral_pct", "avg_score"]
        ].to_dict(orient="records")

    # --- Topic insights ---
    if not topics.empty:
        pos_topics = topics.sort_values("positive_pct", ascending=False)["topic"].head(3).tolist()
        neg_topics = topics.sort_values("negative_pct", ascending=False)["topic"].head(3).tolist()
        story["top_positive_topics"] = pos_topics
        story["top_negative_topics"] = neg_topics

    # --- Product insights ---
    if not products.empty:
        story["top_products"] = products.sort_values("mentions", ascending=False)["product"].head(5).tolist()

    # --- Representative comments per brand ---
    if not comments.empty:
        story["representative_comments"] = comments[
            ["brand", "best_comment", "worst_comment"]
        ].dropna(subset=["brand"]).to_dict(orient="records")

    # Flatten nested lists/dicts to JSON strings so Spark can write parquet
    story_flat = {
        k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
        for k, v in story.items()
    }
    pdf_out = pd.DataFrame([story_flat])
    _write_gold_parquet(pdf_out, "storytelling_weekly", spark)
    log.info("Storytelling parquet written for week %d", current_week)
    spark.stop()


# DAG definition

with DAG(
    dag_id="gold_sentiment_weekly",
    start_date=datetime(2024, 1, 1),
    schedule="0 0 * * 0",   # every Sunday at midnight
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["gold", "sentiment", "weekly"],
    doc_md="""
## Gold Sentiment Weekly DAG

Produces **8 gold artifacts** every week:

| # | Artifact | Location |
|---|----------|----------|
| 1 | `governance_tweets_weekly` | `gold/governance/tweets/` |
| 2 | `governance_news_weekly` | `gold/governance/news/` |
| 3 | `brand_analytics_weekly` | `gold/` |
| 4 | `topic_analytics_weekly` | `gold/` |
| 5 | `product_analytics_weekly` | `gold/` |
| 6 | `top_influencers_weekly` | `gold/` |
| 7 | `representative_comments_weekly` | `gold/` |
| 8 | `storytelling_weekly.json` | `gold/` |

### Task graph
```
gov_tweets ──┐
             ├──> brand_analytics ──┐
gov_news ────┘                      ├──> storytelling
             ├──> topic_analytics ──┤
             ├──> product_analytics─┤
             ├──> top_influencers ──┤
             └──> rep_comments ─────┘
```
    """,
) as dag:

    gov_tweets_task = PythonOperator(
        task_id="governance_tweets",
        python_callable=process_governance_tweets,
    )

    gov_news_task = PythonOperator(
        task_id="governance_news",
        python_callable=process_governance_news,
    )

    brand_task = PythonOperator(
        task_id="brand_analytics",
        python_callable=build_brand_analytics,
    )

    topic_task = PythonOperator(
        task_id="topic_analytics",
        python_callable=build_topic_analytics,
    )

    product_task = PythonOperator(
        task_id="product_analytics",
        python_callable=build_product_analytics,
    )

    influencers_task = PythonOperator(
        task_id="top_influencers",
        python_callable=build_top_influencers,
    )

    comments_task = PythonOperator(
        task_id="representative_comments",
        python_callable=build_representative_comments,
    )

    storytelling_task = PythonOperator(
        task_id="storytelling",
        python_callable=build_storytelling,
    )

    # Task graph
    [gov_tweets_task, gov_news_task] >> brand_task
    [gov_tweets_task, gov_news_task] >> topic_task
    [gov_tweets_task, gov_news_task] >> product_task
    [gov_tweets_task]                >> influencers_task
    [gov_tweets_task, gov_news_task] >> comments_task
    [brand_task, topic_task, product_task, influencers_task, comments_task] >> storytelling_task