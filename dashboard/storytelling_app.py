import os
import glob
import json
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
GOLD_PATH = os.path.join(PROJECT_ROOT, "datalake_gold")
GOV_TWEETS_PATH = os.path.join(GOLD_PATH, "governance", "tweets")
GOV_NEWS_PATH = os.path.join(GOLD_PATH, "governance", "news")

# Helpers to load latest Parquet files
def get_latest_parquet(pattern):
    search_pattern = os.path.join(GOLD_PATH, pattern)
    folders = glob.glob(search_pattern)
    if not folders:
        return None
    return max(folders, key=os.path.getmtime)

def load_gold_storytelling():
    brand_file = get_latest_parquet("brand_analytics_weekly_*.parquet")
    df_brand = pd.read_parquet(brand_file) if brand_file else pd.DataFrame()
    
    topic_file = get_latest_parquet("topic_analytics_weekly_*.parquet")
    df_topic = pd.read_parquet(topic_file) if topic_file else pd.DataFrame()
    
    product_file = get_latest_parquet("product_analytics_weekly_*.parquet")
    df_product = pd.read_parquet(product_file) if product_file else pd.DataFrame()
    
    comments_file = get_latest_parquet("representative_comments_weekly_*.parquet")
    df_comments = pd.read_parquet(comments_file) if comments_file else pd.DataFrame()
    
    story_file = get_latest_parquet("storytelling_weekly_*.parquet")
    df_story = pd.read_parquet(story_file) if story_file else pd.DataFrame()
    
    # Load raw governance records for advanced engagement analysis (Twitter)
    tweets_dir = get_latest_parquet(os.path.join(GOV_TWEETS_PATH, "governance_tweets_weekly_*.parquet"))
    df_raw_tweets = pd.read_parquet(tweets_dir) if tweets_dir else pd.DataFrame()
    
    return {
        "brand": df_brand,
        "topic": df_topic,
        "product": df_product,
        "comments": df_comments,
        "story": df_story,
        "raw_tweets": df_raw_tweets,
        "meta": {
            "brand_file": os.path.basename(brand_file) if brand_file else "None",
            "story_file": os.path.basename(story_file) if story_file else "None",
            "mod_time": os.path.getmtime(story_file) if story_file else 0
        }
    }

# Initialize Dash application
app = dash.Dash(__name__, title="Storytelling Insights Dashboard")

# Load initial data to populate dropdown
initial_data = load_gold_storytelling()
brands_list = []
if not initial_data["brand"].empty:
    brands_list = sorted(initial_data["brand"]["brand"].dropna().unique())

app.layout = html.Div(className="dashboard-container", children=[
    # Header
    html.Div(className="dashboard-header", children=[
        html.H1("Technology Sentiment Storytelling Dashboard", className="dashboard-title"),
        html.P("Translating public mood, brand perception, hot topics and social viral reach into plain-language business insights", className="dashboard-subtitle"),
        html.Div(className="header-meta", children=[
            html.Div(className="meta-badge accent", id="story-meta-last-updated"),
            html.Div(className="meta-badge", children="Audience: Product Managers & Marketing Teams"),
            html.Div(className="meta-badge", id="story-meta-files-loaded")
        ])
    ]),
    
    # Controls Panel (Brand Selector Dropdown)
    html.Div(className="controls-panel", children=[
        html.Span("Focus Brand Analysis:", className="control-label"),
        dcc.Dropdown(
            id="brand-selector",
            options=[{"label": "All Brands Unified", "value": "all"}] + [{"label": b.capitalize(), "value": b} for b in brands_list],
            value="all",
            clearable=False,
            className="control-dropdown"
        )
    ]),
    
    # Reload interval (5 minutes)
    dcc.Interval(id="story-interval-reload", interval=300000, n_intervals=0),
    
    # Executive Narrative Summary
    html.Div(className="narrative-card", children=[
        html.H3("Executive Business Insight", className="narrative-title", id="text-narrative-title"),
        html.P("Loading weekly executive narrative...", className="narrative-body", id="text-executive-summary")
    ]),
    
    # Grid 1: Brand Performance & General Sentiment Mood
    html.Div(className="chart-grid-2", children=[
        # Donut Chart
        html.Div(className="card-premium", children=[
            html.H3("Consumer Mood Snapshot (Sentiment)", className="card-title"),
            html.P("Breakdown of positive, negative, and neutral mentions for the focus brand.", style={"fontSize": "0.85rem", "color": "#64748b", "margin": "-0.5rem 0 1rem 0"}),
            dcc.Graph(id="chart-sentiment-donut", config={"displayModeBar": False})
        ]),
        # Brand Rankings or Brand Details
        html.Div(className="card-premium", children=[
            html.H3("Market Discussion Share", className="card-title", id="chart-brand-title"),
            html.P("Discussion volume share and dominant sentiment mapping across sources.", style={"fontSize": "0.85rem", "color": "#64748b", "margin": "-0.5rem 0 1rem 0"}),
            dcc.Graph(id="chart-brand-ranking", config={"displayModeBar": False})
        ])
    ]),
    
    # Grid 2: Topic Analysis & Product Breakdown
    html.Div(className="chart-grid-2", children=[
        # Topic Heat / Bar
        html.Div(className="card-premium", children=[
            html.H3("Key Discussion Tópicos", className="card-title"),
            html.P("Hot topics ranked by mentions, color encodes average sentiment score.", style={"fontSize": "0.85rem", "color": "#64748b", "margin": "-0.5rem 0 1rem 0"}),
            dcc.Graph(id="chart-topic-sentiment", config={"displayModeBar": False})
        ]),
        # Product Sentiment Stacked
        html.Div(className="card-premium", children=[
            html.H3("Focus Brand Product Models", className="card-title"),
            html.P("Volume of conversations for specific product lines, split by sentiment.", style={"fontSize": "0.85rem", "color": "#64748b", "margin": "-0.5rem 0 1rem 0"}),
            dcc.Graph(id="chart-product-sentiment", config={"displayModeBar": False})
        ])
    ]),
    
    # Grid 3: Advanced Viral Reach & Engagement (NEW ADDITION)
    html.Div(className="chart-grid-1", children=[
        html.Div(className="card-premium", children=[
            html.H3("Social Virality: Sentiment vs Consumer Engagement", className="card-title"),
            html.P("Scatter analysis mapping how consumer sentiment influences viral reach (engagement count). Larger bubbles indicate larger follower bases.", style={"fontSize": "0.85rem", "color": "#64748b", "margin": "-0.5rem 0 1rem 0"}),
            dcc.Graph(id="chart-engagement-scatter", config={"displayModeBar": False})
        ])
    ]),
    
    # Row 4: Representative Community Comments
    html.Div(className="chart-grid-1", children=[
        html.Div(className="card-premium", children=[
            html.H3("Representative Voice of the Community", className="card-title"),
            html.P("High-score and low-score customer reviews that reveal the real context behind the numbers.", style={"fontSize": "0.85rem", "color": "#64748b", "margin": "-0.5rem 0 1.5rem 0"}),
            html.Div(id="container-comments-grid", className="quotes-grid")
        ])
    ])
])

# Callbacks to dynamically populate charts, selector, and text narratives
@app.callback(
    [
        Output("text-narrative-title", "children"),
        Output("text-executive-summary", "children"),
        Output("chart-sentiment-donut", "figure"),
        Output("chart-brand-ranking", "figure"),
        Output("chart-topic-sentiment", "figure"),
        Output("chart-product-sentiment", "figure"),
        Output("chart-engagement-scatter", "figure"),
        Output("container-comments-grid", "children"),
        Output("story-meta-last-updated", "children"),
        Output("story-meta-files-loaded", "children"),
        Output("chart-brand-title", "children")
    ],
    [
        Input("brand-selector", "value"),
        Input("story-interval-reload", "n_intervals")
    ]
)
def update_story_dashboard(selected_brand, n):
    # 1. Load Data
    data = load_gold_storytelling()
    df_brand = data["brand"]
    df_topic = data["topic"]
    df_product = data["product"]
    df_comments = data["comments"]
    df_story = data["story"]
    df_raw_tweets = data["raw_tweets"]
    meta = data["meta"]
    
    # 2. Executive Narrative Text (Dynamic based on selected brand)
    if selected_brand == "all" or df_brand.empty:
        narrative_title = "Weekly Executive Summary"
        exec_summary = "No narrative summary found. Run the Gold weekly DAG."
        if not df_story.empty and "executive_summary" in df_story.columns:
            exec_summary = df_story.iloc[0]["executive_summary"]
    else:
        narrative_title = f"{selected_brand.capitalize()} - Brand Performance Focus"
        brand_row = df_brand[df_brand["brand"] == selected_brand]
        if not brand_row.empty:
            row_data = brand_row.iloc[0]
            total_m = int(row_data["total_mentions"])
            pos = row_data["positive_pct"]
            neg = row_data["negative_pct"]
            score = row_data["avg_score"]
            sentiment_summary = "mostly positive" if score > 0.05 else ("mostly negative" if score < -0.05 else "mostly neutral")
            exec_summary = (
                f"During the current week, {selected_brand.capitalize()} accumulated {total_m:,} mentions across social media "
                f"and tech news comments. Consumer sentiment is {sentiment_summary} with a net sentiment score of {score:+.4f}. "
                f"Specifically, {pos:.1f}% of discussions were positive, while {neg:.1f}% were negative. "
                f"Below is a detailed breakdown of the products, topics, and community comments driving this feedback."
            )
        else:
            exec_summary = f"No detailed records found for brand: {selected_brand.capitalize()}."

    # 3. Donut chart (Global mood - Dynamic)
    if not df_brand.empty:
        if selected_brand == "all":
            # Unified across all brands
            pos_mentions = (df_brand["total_mentions"] * df_brand["positive_pct"] / 100).sum()
            neg_mentions = (df_brand["total_mentions"] * df_brand["negative_pct"] / 100).sum()
            neu_mentions = (df_brand["total_mentions"] * df_brand["neutral_pct"] / 100).sum()
        else:
            # Single brand focus
            brand_row = df_brand[df_brand["brand"] == selected_brand]
            if not brand_row.empty:
                r = brand_row.iloc[0]
                pos_mentions = r["total_mentions"] * r["positive_pct"] / 100
                neg_mentions = r["total_mentions"] * r["negative_pct"] / 100
                neu_mentions = r["total_mentions"] * r["neutral_pct"] / 100
            else:
                pos_mentions, neg_mentions, neu_mentions = 0, 0, 0
                
        total_m = pos_mentions + neg_mentions + neu_mentions
        dominant_label = "Neutral"
        if total_m > 0:
            if pos_mentions > neg_mentions and pos_mentions > neu_mentions:
                dominant_label = "Positive"
            elif neg_mentions > pos_mentions and neg_mentions > neu_mentions:
                dominant_label = "Negative"
            
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[pos_mentions, neu_mentions, neg_mentions],
            hole=0.6,
            marker=dict(colors=["#2ecc71", "#7f8c8d", "#e74c3c"]),
            hoverinfo="label+percent+value",
            textinfo="label+percent",
            textposition="inside"
        )])
        fig_donut.update_layout(
            annotations=[dict(text=f"Mood:<br><b>{dominant_label}</b>", x=0.5, y=0.5, font_size=20, showarrow=False)],
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
    else:
        fig_donut = go.Figure()
        
    # 4. Brand rankings horizontal bar / Brand Details (Dynamic)
    brand_title = "Market Discussion Share"
    if not df_brand.empty:
        if selected_brand == "all":
            df_brand_long = df_brand.copy()
            df_brand_long["Positive"] = df_brand_long["total_mentions"] * df_brand_long["positive_pct"] / 100
            df_brand_long["Neutral"] = df_brand_long["total_mentions"] * df_brand_long["neutral_pct"] / 100
            df_brand_long["Negative"] = df_brand_long["total_mentions"] * df_brand_long["negative_pct"] / 100
            
            df_melt = pd.melt(
                df_brand_long, id_vars=["brand"], value_vars=["Positive", "Neutral", "Negative"],
                var_name="Sentiment", value_name="Mentions"
            ).sort_values("Mentions", ascending=True)
            
            brand_order = df_brand.sort_values("total_mentions", ascending=True)["brand"].tolist()
            
            fig_brand = px.bar(
                df_melt, x="Mentions", y="brand", color="Sentiment",
                orientation="h",
                color_discrete_map={"Positive": "#2ecc71", "Neutral": "#7f8c8d", "Negative": "#e74c3c"},
                category_orders={"brand": brand_order}
            )
        else:
            # Show Twitter vs News Comments split for the selected brand
            brand_row = df_brand[df_brand["brand"] == selected_brand].iloc[0]
            brand_title = f"{selected_brand.capitalize()} - Channel Breakdown"
            
            fig_brand = go.Figure(data=[
                go.Bar(name="Twitter Mentions", x=["Twitter / X"], y=[brand_row["mentions_twitter"]], marker_color="#1b4fbf"),
                go.Bar(name="News Comments", x=["GSM Arena Comments"], y=[brand_row["mentions_news"]], marker_color="#00c2cb")
            ])
            fig_brand.update_layout(barmode="group")
            
        fig_brand.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(l=40, r=40, t=10, b=40),
            xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
            yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else:
        fig_brand = go.Figure()
        
    # 5. Topic Sentiment Net Chart (Dynamic)
    if not df_topic.empty:
        # If a specific brand is selected, we should filter topics matching that brand!
        # Since topic_analytics doesn't have brand column, we can do it by analyzing raw_tweets
        if selected_brand != "all" and not df_raw_tweets.empty and "brand" in df_raw_tweets.columns:
            brand_tweets = df_raw_tweets[df_raw_tweets["brand"] == selected_brand]
            if not brand_tweets.empty and "topic" in brand_tweets.columns:
                df_sub_topic = brand_tweets.groupby("topic").agg(
                    mentions=("tweet_id", "count"),
                    avg_score=("sentiment_score", "mean")
                ).reset_index()
                df_top = df_sub_topic.sort_values("mentions", ascending=False).head(12)
            else:
                df_top = df_topic.sort_values("mentions", ascending=False).head(12)
        else:
            df_top = df_topic.sort_values("mentions", ascending=False).head(12)
            
        if not df_top.empty:
            fig_topic = px.bar(
                df_top, x="mentions", y="topic", color="avg_score",
                orientation="h",
                labels={"topic": "Topic Field", "mentions": "Menciones", "avg_score": "Sent. Promedio"},
                color_continuous_scale=["#e74c3c", "#7f8c8d", "#2ecc71"],
                color_continuous_midpoint=0.0,
                category_orders={"topic": df_top.sort_values("mentions", ascending=True)["topic"].tolist()}
            )
            fig_topic.update_layout(
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                margin=dict(l=40, r=40, t=10, b=40),
                xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
                yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
                coloraxis_colorbar=dict(title="Score", thickness=15, len=0.8)
            )
        else:
            fig_topic = go.Figure()
    else:
        fig_topic = go.Figure()
        
    # 6. Product Sentiment stacked bar (Dynamic)
    if not df_product.empty:
        if selected_brand == "all":
            df_prod_filtered = df_product
        else:
            df_prod_filtered = df_product[df_product["brand"] == selected_brand]
            
        df_prod = df_prod_filtered.sort_values("mentions", ascending=False).head(10)
        
        if not df_prod.empty:
            df_prod_long = df_prod.copy()
            df_prod_long["Positive"] = df_prod_long["mentions"] * df_prod_long["positive_pct"] / 100
            df_prod_long["Neutral"] = df_prod_long["mentions"] * df_prod_long["neutral_pct"] / 100
            df_prod_long["Negative"] = df_prod_long["mentions"] * df_prod_long["negative_pct"] / 100
            
            df_pmelt = pd.melt(
                df_prod_long, id_vars=["product"], value_vars=["Positive", "Neutral", "Negative"],
                var_name="Sentiment", value_name="Mentions"
            )
            
            fig_prod = px.bar(
                df_pmelt, x="product", y="Mentions", color="Sentiment",
                color_discrete_map={"Positive": "#2ecc71", "Neutral": "#7f8c8d", "Negative": "#e74c3c"},
                category_orders={"product": df_prod["product"].tolist()}
            )
            fig_prod.update_layout(
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                margin=dict(l=40, r=40, t=10, b=40),
                xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
                yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
        else:
            fig_prod = go.Figure()
            fig_prod.update_layout(title="No products recorded for this brand")
    else:
        fig_prod = go.Figure()
        
    # 8. Advanced Social Engagement Scatter (NEW CHART)
    if not df_raw_tweets.empty:
        if selected_brand == "all":
            df_scatter_filtered = df_raw_tweets
        else:
            df_scatter_filtered = df_raw_tweets[df_raw_tweets["brand"] == selected_brand]
            
        if not df_scatter_filtered.empty:
            # Map sentiment tags to specific hex codes
            df_scatter_filtered = df_scatter_filtered.copy()
            df_scatter_filtered["Engagement Count"] = df_scatter_filtered["total_engagement"].fillna(0)
            df_scatter_filtered["Follower Base"] = pd.to_numeric(df_scatter_filtered["author_followers"], errors="coerce").fillna(0)
            
            # Scatter Plot
            fig_scatter = px.scatter(
                df_scatter_filtered,
                x="sentiment_score",
                y="Engagement Count",
                size="Follower Base",
                color="sentiment",
                hover_data=["author_userName", "brand", "topic"],
                labels={
                    "sentiment_score": "VADER Net Sentiment Intensity Score",
                    "Engagement Count": "User Engagement (Likes+Retweets)",
                    "sentiment": "Mood Tag"
                },
                color_discrete_map={"positive": "#2ecc71", "neutral": "#7f8c8d", "negative": "#e74c3c"},
                size_max=40
            )
            fig_scatter.update_layout(
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                margin=dict(l=40, r=40, t=10, b=40),
                xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
                yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
        else:
            fig_scatter = go.Figure()
            fig_scatter.update_layout(title="No raw social engagement data")
    else:
        fig_scatter = go.Figure()
        
    # 7. Representative comment cards (quotes - Dynamic & Slicing Safe)
    comments_html = []
    if not df_comments.empty:
        if selected_brand == "all":
            df_comm_filtered = df_comments
        else:
            df_comm_filtered = df_comments[df_comments["brand"] == selected_brand]
            
        for idx, row in df_comm_filtered.iterrows():
            brand_name = str(row["brand"]).capitalize()
            best_val = row.get("best_comment")
            worst_val = row.get("worst_comment")
            
            best = str(best_val) if (pd.notna(best_val) and best_val is not None) else ""
            worst = str(worst_val) if (pd.notna(worst_val) and worst_val is not None) else ""
            
            brand_cards = []
            if best.strip() != "" and best.strip().lower() != "none" and best.strip().lower() != "nan":
                brand_cards.append(html.Div(className="quote-card positive", children=[
                    html.P(f'"{best[:260]}..."' if len(best) > 260 else f'"{best}"', className="quote-text"),
                    html.Div(f"★ Positive Feedback on {brand_name}", className="quote-author")
                ]))
            if worst.strip() != "" and worst.strip().lower() != "none" and worst.strip().lower() != "nan":
                brand_cards.append(html.Div(className="quote-card negative", children=[
                    html.P(f'"{worst[:260]}..."' if len(worst) > 260 else f'"{worst}"', className="quote-text"),
                    html.Div(f"⚠ Critical Feedback on {brand_name}", className="quote-author")
                ]))
                
            if brand_cards:
                comments_html.append(html.Div(className="card-premium", style={"boxShadow": "none", "border": "1px solid #f1f5f9", "padding": "1rem"}, children=[
                    html.H4(brand_name, style={"margin": "0 0 1rem 0", "fontWeight": "800", "color": "#0d1b2a", "fontSize": "1.1rem"}),
                    html.Div(children=brand_cards)
                ]))
    if not comments_html:
        comments_html = [html.P("No representative comments loaded for this selection.", style={"color": "#64748b"})]
        
    # Meta tags
    if meta["mod_time"] > 0:
        import datetime
        dt = datetime.datetime.fromtimestamp(meta["mod_time"]).strftime("%Y-%m-%d %H:%M:%S")
        updated_str = f"Insights Updated: {dt}"
    else:
        updated_str = "Insights Updated: N/A"
        
    files_str = f"Loaded parquets: {meta['brand_file']} | {meta['story_file']}"
    
    return (
        narrative_title,
        exec_summary,
        fig_donut,
        fig_brand,
        fig_topic,
        fig_prod,
        fig_scatter,
        comments_html,
        updated_str,
        files_str,
        brand_title
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8051, debug=False)
