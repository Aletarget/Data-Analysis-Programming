import os
import glob
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Dynamic path resolver for Gold layers to support both Docker and local developer laptops
def resolve_gold_path():
    env_path = os.getenv("GOLD_PATH")
    if env_path:
        return env_path
    
    # Resolve relative to the script location so it works from any execution working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, "../datalake_gold"),
        os.path.join(script_dir, "datalake_gold"),
        os.path.join(os.getcwd(), "datalake_gold"),
        os.path.join(os.getcwd(), "../datalake_gold"),
    ]
    for p in possible_paths:
        if os.path.exists(p) and os.path.isdir(p):
            return os.path.abspath(p)
            
    return "/opt/airflow/datalake_gold"

GOLD_PATH = resolve_gold_path()
GOV_TWEETS_PATH = os.path.join(GOLD_PATH, "governance", "tweets")
GOV_NEWS_PATH = os.path.join(GOLD_PATH, "governance", "news")

# Helper to find latest Parquet folder/file
def get_latest_parquet(dir_path, pattern):
    search_pattern = os.path.join(dir_path, pattern)
    folders = glob.glob(search_pattern)
    if not folders:
        return None
    return max(folders, key=os.path.getmtime)

# Load governance data
def load_data():
    tweets_dir = get_latest_parquet(GOV_TWEETS_PATH, "governance_tweets_weekly_*.parquet")
    news_dir = get_latest_parquet(GOV_NEWS_PATH, "governance_news_weekly_*.parquet")
    
    df_tweets = pd.read_parquet(tweets_dir) if tweets_dir else pd.DataFrame()
    df_news = pd.read_parquet(news_dir) if news_dir else pd.DataFrame()
    
    # Ensure text length column exists
    if not df_tweets.empty and "text" in df_tweets.columns:
        df_tweets["text_length"] = df_tweets["text"].fillna("").str.len()
    if not df_news.empty and "comment_text" in df_news.columns:
        df_news["text_length"] = df_news["comment_text"].fillna("").str.len()
        
    return df_tweets, df_news, tweets_dir, news_dir

# Compute Governance KPIs
def compute_kpis(df_tweets, df_news, selected_source):
    # Filter based on selected stream source
    if selected_source == "twitter":
        active_tweets = df_tweets
        active_news = pd.DataFrame()
    elif selected_source == "news":
        active_tweets = pd.DataFrame()
        active_news = df_news
    else:
        active_tweets = df_tweets
        active_news = df_news

    total_tweets = len(active_tweets)
    total_news = len(active_news)
    total_records = total_tweets + total_news
    
    if total_records == 0:
        return {
            "total_records": 0,
            "null_rate": 0.0,
            "duplicate_rate": 0.0,
            "schema_compliance": 0.0,
            "unique_languages": 0,
            "total_engagement": 0,
            "null_by_col": pd.DataFrame(),
            "outlier_by_col": pd.DataFrame(),
            "volume_over_time": pd.DataFrame(),
            "text_length_data": pd.DataFrame(),
            "kpi_table_data": []
        }
    
    # 1. Null Rate computation on critical columns
    tweets_cols = ["tweet_id", "date_day", "author_userName", "text", "sentiment", "sentiment_score"]
    news_cols = ["newsLink", "date_day", "comment_text", "sentiment", "sentiment_score"]
    
    tweets_nulls = active_tweets[active_tweets.columns.intersection(tweets_cols)].isna().sum().sum() if total_tweets > 0 else 0
    news_nulls = active_news[active_news.columns.intersection(news_cols)].isna().sum().sum() if total_news > 0 else 0
    
    total_expected_cells = (total_tweets * len(tweets_cols)) + (total_news * len(news_cols))
    overall_null_rate = (tweets_nulls + news_nulls) / total_expected_cells * 100 if total_expected_cells > 0 else 0.0
    
    # Null rate by column for visualization
    null_rates = []
    if total_tweets > 0:
        for c in active_tweets.columns:
            null_rates.append({
                "Source": "Twitter / X", 
                "Column": c, 
                "Null Rate (%)": active_tweets[c].isna().mean() * 100
            })
    if total_news > 0:
        for c in active_news.columns:
            null_rates.append({
                "Source": "GSM Arena (News)", 
                "Column": c, 
                "Null Rate (%)": active_news[c].isna().mean() * 100
            })
    df_null_by_col = pd.DataFrame(null_rates).sort_values("Null Rate (%)", ascending=False)
    
    # 2. Duplicate Rate
    dup_tweets = active_tweets["tweet_id"].duplicated().sum() if "tweet_id" in active_tweets.columns else 0
    dup_news = active_news.duplicated(subset=["newsLink", "comment_text"]).sum() if ("newsLink" in active_news.columns and "comment_text" in active_news.columns) else 0
    overall_dup_rate = (dup_tweets + dup_news) / total_records * 100
    
    # 3. Schema Compliance
    expected_tweets_count = 18
    expected_news_count = 9
    
    actual_tweets_count = len(active_tweets.columns) if total_tweets > 0 else 0
    actual_news_count = len(active_news.columns) if total_news > 0 else 0
    
    compliance_tweets = min(actual_tweets_count / expected_tweets_count * 100, 100.0) if expected_tweets_count > 0 else 0.0
    compliance_news = min(actual_news_count / expected_news_count * 100, 100.0) if expected_news_count > 0 else 0.0
    
    if total_tweets > 0 and total_news > 0:
        overall_schema_compliance = (compliance_tweets + compliance_news) / 2
    elif total_tweets > 0:
        overall_schema_compliance = compliance_tweets
    else:
        overall_schema_compliance = compliance_news
        
    # 4. Diversity metrics
    unique_languages = active_tweets["lang"].nunique() if "lang" in active_tweets.columns else 0
    total_engagement = int(active_tweets["total_engagement"].sum()) if "total_engagement" in active_tweets.columns else 0
    
    # 5. Outliers (IQR Method)
    outlier_metrics = []
    numeric_cols = ["likeCount", "replyCount", "retweetCount", "quoteCount", "author_followers"]
    if total_tweets > 0:
        for c in numeric_cols:
            if c in active_tweets.columns:
                series = pd.to_numeric(active_tweets[c], errors="coerce").dropna()
                if len(series) > 0:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = series[(series < lower) | (series > upper)].count()
                    rate = outliers / len(series) * 100
                    outlier_metrics.append({
                        "Metric": c, 
                        "Outlier Rate (%)": rate, 
                        "Outliers": int(outliers), 
                        "Sample Size": len(series)
                    })
    df_outliers = pd.DataFrame(outlier_metrics)
    
    # 6. Ingested volume
    vol_tweets = active_tweets.groupby("date_day").size().reset_index(name="Volume").assign(Source="Twitter / X") if "date_day" in active_tweets.columns else pd.DataFrame()
    vol_news = active_news.groupby("date_day").size().reset_index(name="Volume").assign(Source="News Comments") if "date_day" in active_news.columns else pd.DataFrame()
    df_volume = pd.concat([vol_tweets, vol_news], ignore_index=True)
    if not df_volume.empty:
        df_volume["date_day"] = pd.to_datetime(df_volume["date_day"]).dt.strftime("%Y-%m-%d")
        df_volume = df_volume.sort_values("date_day")
        
    # 7. Text lengths
    text_lengths = []
    if total_tweets > 0 and "text_length" in active_tweets.columns:
        text_lengths.append(active_tweets[["text_length"]].assign(Source="Twitter / X"))
    if total_news > 0 and "text_length" in active_news.columns:
        text_lengths.append(active_news[["text_length"]].assign(Source="News Comments"))
    df_lengths = pd.concat(text_lengths, ignore_index=True) if text_lengths else pd.DataFrame()
    
    # Tabular compliance records
    kpi_table = [
        {"kpi": "Total Records Ingested", "value": f"{total_records:,}", "threshold": "N/A", "status": "INFO", "desc": "Sum total of comments and tweets processed successfully in this batch."},
        {"kpi": "Null Rate (Critical Fields)", "value": f"{overall_null_rate:.2f}%", "threshold": "< 5.00%", "status": "PASS" if overall_null_rate < 5 else "FAIL", "desc": "Percentage of missing cells in critical fields. Low values ensure complete profiles."},
        {"kpi": "Duplicate Rate", "value": f"{overall_dup_rate:.2f}%", "threshold": "< 2.00%", "status": "PASS" if overall_dup_rate < 2 else "WARNING", "desc": "Percentage of duplicate unique identifiers. Helps detect spam or ingestion repeats."},
        {"kpi": "Schema Compliance", "value": f"{overall_schema_compliance:.1f}%", "threshold": "> 95.0%", "status": "PASS" if overall_schema_compliance > 95 else "FAIL", "desc": "Percentage of expected columns present. Protects analytics engines from breaking."},
        {"kpi": "Unique Languages Detected", "value": f"{unique_languages}", "threshold": "N/A", "status": "INFO", "desc": "Diversity of languages recorded in the ingested X/Twitter dataset."},
        {"kpi": "Total Social Engagement", "value": f"{total_engagement:,}", "threshold": "N/A", "status": "INFO", "desc": "Accumulated social interactions (likes + retweets + replies + quotes)."}
    ]
    
    return {
        "total_records": total_records,
        "null_rate": overall_null_rate,
        "duplicate_rate": overall_dup_rate,
        "schema_compliance": overall_schema_compliance,
        "unique_languages": unique_languages,
        "total_engagement": total_engagement,
        "null_by_col": df_null_by_col,
        "outlier_by_col": df_outliers,
        "volume_over_time": df_volume,
        "text_length_data": df_lengths,
        "kpi_table_data": kpi_table
    }

# Initialize Dash application
app = dash.Dash(__name__, title="Governance Quality Dashboard")

# App Layout
app.layout = html.Div(className="dashboard-container", children=[
    # Header
    html.Div(className="dashboard-header", children=[
        html.H1("Governance & Data Quality Dashboard", className="dashboard-title"),
        html.P("Real-time data quality monitoring, schema compliance, text profiles, and outlier tracking across Medallion layers", className="dashboard-subtitle"),
        html.Div(className="header-meta", children=[
            html.Div(className="meta-badge accent", id="meta-last-updated"),
            html.Div(className="meta-badge", children="Audience: Data Engineers & Quality Analysts"),
            html.Div(className="meta-badge", id="meta-files-loaded")
        ])
    ]),
    
    # Informative panel explaining business rules
    html.Div(className="card-premium", style={"marginBottom": "2rem", "borderLeft": "6px solid #1b4fbf", "background": "linear-gradient(to right, #eff6ff, #ffffff)"}, children=[
        html.H3("Business Rules for Data Quality (Thresholds)", style={"margin": "0 0 0.75rem 0", "color": "#1b4fbf", "fontWeight": "800", "fontSize": "1.15rem"}),
        html.P([
            html.Span("This panel automatically monitors data status against the quality standards defined for this project:", style={"fontWeight": "600", "color": "#334155"}),
            html.Br(),
            html.Strong("1. Null Rate (< 5.0%): "), "Ensures completeness of ingested attributes. High rates block downstream classification engines.",
            html.Br(),
            html.Strong("2. Duplicate Rate (< 2.0%): "), "Monitors for data replication, spam accounts, or network ingestion retries. Crucial for statistical accuracy.",
            html.Br(),
            html.Strong("3. Schema Compliance (> 95.0%): "), "Ensures the structure of Parquet layers matches expected columns to prevent runtime query crashes."
        ], style={"lineHeight": "1.6", "fontSize": "0.95rem", "color": "#475569", "margin": "0"})
    ]),
    
    # Input panel
    html.Div(className="controls-panel", children=[
        html.Span("Filter Data Stream Source:", className="control-label"),
        dcc.Dropdown(
            id="source-selector",
            options=[
                {"label": "All Streams Unified (Twitter + GSM Arena)", "value": "all"},
                {"label": "Twitter / X Stream Only", "value": "twitter"},
                {"label": "GSM Arena (News Comments) Only", "value": "news"}
            ],
            value="all",
            clearable=False,
            className="control-dropdown",
            style={"width": "350px"}
        )
    ]),
    
    # 5-minute reload interval
    dcc.Interval(id="interval-reload", interval=300000, n_intervals=0),
    
    # KPI Grid cards
    html.Div(className="kpi-grid", children=[
        # Card 1: Total Records
        html.Div(className="card-premium kpi-card kpi-success", children=[
            html.Div("Total Records Ingested", className="kpi-label"),
            html.Div("-", className="kpi-value", id="kpi-total-records"),
            html.Span("Active", className="kpi-status-badge status-success", id="status-total-records")
        ]),
        # Card 2: Null Rate
        html.Div(className="card-premium kpi-card", id="card-null-rate", children=[
            html.Div("Null Rate", className="kpi-label"),
            html.Div("-", className="kpi-value", id="kpi-null-rate"),
            html.Span("-", className="kpi-status-badge", id="status-null-rate")
        ]),
        # Card 3: Duplicate Rate
        html.Div(className="card-premium kpi-card", id="card-dup-rate", children=[
            html.Div("Duplicate Rate", className="kpi-label"),
            html.Div("-", className="kpi-value", id="kpi-dup-rate"),
            html.Span("-", className="kpi-status-badge", id="status-dup-rate")
        ]),
        # Card 4: Schema Compliance
        html.Div(className="card-premium kpi-card", id="card-compliance-rate", children=[
            html.Div("Schema Compliance", className="kpi-label"),
            html.Div("-", className="kpi-value", id="kpi-compliance-rate"),
            html.Span("-", className="kpi-status-badge", id="status-compliance-rate")
        ])
    ]),
    
    # Grid 1: Volume Line Chart & Null rates per column
    html.Div(className="chart-grid-2", children=[
        # Volume
        html.Div(className="card-premium", children=[
            html.H3("Ingested Data Volume over Time", className="card-title"),
            html.P("Displays the daily volume of tweets and comments successfully processed, helping analysts spot collection spikes or server downtime.", style={"fontSize": "0.85rem", "color": "#64748b", "margin": "-0.5rem 0 1rem 0"}),
            dcc.Graph(id="chart-volume", config={"displayModeBar": False})
        ]),
        # Null columns
        html.Div(className="card-premium", children=[
            html.H3("Field Null Rates (Top 10)", className="card-title"),
            html.P("Maps which columns contain the highest ratio of missing attributes, showing if API responses or scrapers are dropping parameters.", style={"fontSize": "0.85rem", "color": "#64748b", "margin": "-0.5rem 0 1rem 0"}),
            dcc.Graph(id="chart-null-by-col", config={"displayModeBar": False})
        ])
    ]),
    
    # Grid 2: Text Length boxplot & Outlier Rates
    html.Div(className="chart-grid-2", children=[
        # Length BoxPlot
        html.Div(className="card-premium", children=[
            html.H3("Text Length Profile by Source (Character Count)", className="card-title"),
            html.P("Measures the distribution of characters per comment/tweet. Useful for separating short noisy statements from deep reviews.", style={"fontSize": "0.85rem", "color": "#64748b", "margin": "-0.5rem 0 1rem 0"}),
            dcc.Graph(id="chart-text-lengths", config={"displayModeBar": False})
        ]),
        # Outliers Bar
        html.Div(className="card-premium", children=[
            html.H3("Outlier Rates in Social Metrics (IQR)", className="card-title"),
            html.P("Identifies extreme viral posts or bot accounts whose interaction count (likes, retweets) falls way outside normal distributions.", style={"fontSize": "0.85rem", "color": "#64748b", "margin": "-0.5rem 0 1rem 0"}),
            dcc.Graph(id="chart-outliers", config={"displayModeBar": False})
        ])
    ]),
    
    # Grid 3: Full KPI Tabular View
    html.Div(className="chart-grid-1", children=[
        html.Div(className="card-premium", children=[
            html.H3("Data Quality Control Framework Audit", className="card-title"),
            html.P("Detailed tabular log of each monitored quality metric, its business rule threshold, and current verification status.", style={"fontSize": "0.85rem", "color": "#64748b", "margin": "-0.5rem 0 1.5rem 0"}),
            html.Div(style={"marginTop": "0.5rem"}, children=[
                dash_table.DataTable(
                    id="quality-table",
                    columns=[
                        {"name": "Quality Control KPI", "id": "kpi"},
                        {"name": "Current Value", "id": "value"},
                        {"name": "Target Threshold", "id": "threshold"},
                        {"name": "Compliance Status", "id": "status"},
                        {"name": "Description & Traceability", "id": "desc"}
                    ],
                    style_as_list_view=True,
                    style_cell={
                        "textAlign": "left",
                        "padding": "12px",
                        "fontFamily": "Inter, sans-serif"
                    },
                    style_header={
                        "backgroundColor": "#f8fafc",
                        "fontWeight": "bold",
                        "borderBottom": "2px solid #e2e8f0"
                    },
                    style_data_conditional=[
                        {
                            "if": {"column_id": "status", "filter_query": "{status} eq 'PASS'"},
                            "color": "#27ae60",
                            "fontWeight": "bold"
                        },
                        {
                            "if": {"column_id": "status", "filter_query": "{status} eq 'WARNING'"},
                            "color": "#d35400",
                            "fontWeight": "bold"
                        },
                        {
                            "if": {"column_id": "status", "filter_query": "{status} eq 'FAIL'"},
                            "color": "#c0392b",
                            "fontWeight": "bold"
                        },
                        {
                            "if": {"column_id": "status", "filter_query": "{status} eq 'INFO'"},
                            "color": "#1b4fbf",
                            "fontWeight": "bold"
                        }
                    ]
                )
            ])
        ])
    ])
])

# Callbacks to dynamically update quality metrics & charts
@app.callback(
    [
        Output("kpi-total-records", "children"),
        Output("kpi-null-rate", "children"),
        Output("status-null-rate", "children"),
        Output("status-null-rate", "className"),
        Output("card-null-rate", "className"),
        
        Output("kpi-dup-rate", "children"),
        Output("status-dup-rate", "children"),
        Output("status-dup-rate", "className"),
        Output("card-dup-rate", "className"),
        
        Output("kpi-compliance-rate", "children"),
        Output("status-compliance-rate", "children"),
        Output("status-compliance-rate", "className"),
        Output("card-compliance-rate", "className"),
        
        Output("chart-volume", "figure"),
        Output("chart-null-by-col", "figure"),
        Output("chart-text-lengths", "figure"),
        Output("chart-outliers", "figure"),
        Output("quality-table", "data"),
        
        Output("meta-last-updated", "children"),
        Output("meta-files-loaded", "children")
    ],
    [
        Input("source-selector", "value"),
        Input("interval-reload", "n_intervals")
    ]
)
def update_dashboard(selected_source, n):
    # 1. Load Data
    df_tweets, df_news, tweets_file, news_file = load_data()
    
    # 2. Compute KPIs
    metrics = compute_kpis(df_tweets, df_news, selected_source)
    
    # Formatting and status bindings
    total_rec_str = f"{metrics['total_records']:,}"
    
    # Null rate status
    null_rate = metrics["null_rate"]
    null_rate_str = f"{null_rate:.2f}%"
    if null_rate < 2:
        null_status, null_badge_class, null_card_class = "Optimal", "kpi-status-badge status-success", "card-premium kpi-card kpi-success"
    elif null_rate < 5:
        null_status, null_badge_class, null_card_class = "Warning", "kpi-status-badge status-warning", "card-premium kpi-card kpi-warning"
    else:
        null_status, null_badge_class, null_card_class = "Critical", "kpi-status-badge status-danger", "card-premium kpi-card kpi-danger"
        
    # Duplicate rate status
    dup_rate = metrics["duplicate_rate"]
    dup_rate_str = f"{dup_rate:.2f}%"
    if dup_rate < 1:
        dup_status, dup_badge_class, dup_card_class = "Optimal", "kpi-status-badge status-success", "card-premium kpi-card kpi-success"
    elif dup_rate < 2:
        dup_status, dup_badge_class, dup_card_class = "Warning", "kpi-status-badge status-warning", "card-premium kpi-card kpi-warning"
    else:
        dup_status, dup_badge_class, dup_card_class = "Critical", "kpi-status-badge status-danger", "card-premium kpi-card kpi-danger"
        
    # Schema compliance status
    comp_rate = metrics["schema_compliance"]
    comp_rate_str = f"{comp_rate:.1f}%"
    if comp_rate >= 98:
        comp_status, comp_badge_class, comp_card_class = "Compliant", "kpi-status-badge status-success", "card-premium kpi-card kpi-success"
    elif comp_rate >= 95:
        comp_status, comp_badge_class, comp_card_class = "Minor Drift", "kpi-status-badge status-warning", "card-premium kpi-card kpi-warning"
    else:
        comp_status, comp_badge_class, comp_card_class = "Critical", "kpi-status-badge status-danger", "card-premium kpi-card kpi-danger"
        
    # 3. Create Plots
    # Ingested Volume Over Time
    df_volume = metrics["volume_over_time"]
    if not df_volume.empty:
        fig_volume = px.line(
            df_volume, x="date_day", y="Volume", color="Source",
            labels={"date_day": "Ingestion Date", "Volume": "Records Count"},
            color_discrete_map={"Twitter / X": "#1b4fbf", "News Comments": "#00c2cb"}
        )
        fig_volume.update_traces(line=dict(width=3), marker=dict(size=6))
    else:
        fig_volume = px.line(title="No volume records found")
    
    fig_volume.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", title="Date"),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", title="Record Count"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Null rate by Column (Top 10 null columns)
    df_null = metrics["null_by_col"]
    if not df_null.empty:
        df_null_filtered = df_null.head(10)
        fig_null = px.bar(
            df_null_filtered, x="Null Rate (%)", y="Column", color="Source",
            orientation="h",
            labels={"Column": "Field Name", "Null Rate (%)": "Null Rate (%)"},
            color_discrete_map={"Twitter / X": "#1b4fbf", "GSM Arena (News)": "#00c2cb"},
            category_orders={"Column": df_null_filtered["Column"].tolist()}
        )
    else:
        fig_null = px.bar(title="No null rate data")
        
    fig_null.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", range=[0, 100], title="Null Ratio (%)"),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", title="Field Name"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Text length distribution boxplot
    df_len = metrics["text_length_data"]
    if not df_len.empty:
        fig_len = px.box(
            df_len, x="Source", y="text_length", color="Source",
            labels={"text_length": "Character Count", "Source": "Stream Source"},
            color_discrete_map={"Twitter / X": "#1b4fbf", "News Comments": "#00c2cb"}
        )
    else:
        fig_len = px.box(title="No text length metrics available")
        
    fig_len.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", title="Stream Source"),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", title="Character Count"),
        showlegend=False
    )
    
    # Outlier Rate by Column Bar Chart
    df_out = metrics["outlier_by_col"]
    if not df_out.empty:
        fig_out = px.bar(
            df_out, x="Outlier Rate (%)", y="Metric",
            orientation="h",
            labels={"Metric": "Social Metric", "Outlier Rate (%)": "Outlier Ratio (%)"},
            color_discrete_sequence=["#e74c3c"]
        )
    else:
        fig_out = px.bar(title="No outlier metrics found (Twitter only)")
        
    fig_out.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", range=[0, 20], title="Outlier Rate (%)"),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", title="Metric")
    )
    
    # 4. Generate metadata timestamps
    last_mod_time = max(os.path.getmtime(tweets_file) if tweets_file else 0, os.path.getmtime(news_file) if news_file else 0)
    if last_mod_time > 0:
        import datetime
        dt = datetime.datetime.fromtimestamp(last_mod_time).strftime("%Y-%m-%d %H:%M:%S")
        updated_str = f"Data Updated: {dt}"
    else:
        updated_str = "Data Updated: N/A"
        
    files_str = f"Active Parquets: {os.path.basename(tweets_file) if tweets_file else 'None'} | {os.path.basename(news_file) if news_file else 'None'}"
    
    return (
        total_rec_str,
        null_rate_str, null_status, null_badge_class, null_card_class,
        dup_rate_str, dup_status, dup_badge_class, dup_card_class,
        comp_rate_str, comp_status, comp_badge_class, comp_card_class,
        fig_volume, fig_null, fig_len, fig_out, metrics["kpi_table_data"],
        updated_str, files_str
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
