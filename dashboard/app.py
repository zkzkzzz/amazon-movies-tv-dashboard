"""Streamlit dashboard for the Amazon Movies & TV review project."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


TEAM_NAME = "kai"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PAGES = ["Overview", "Product Explorer", "Recommender", "Sentiment Analyzer"]
UNSPECIFIED_LABELS = {"", ".", "nan", "none", "unknown", "various"}
CHART_BLUE = "#2563eb"
CHART_TEAL = "#0f766e"
CHART_AMBER = "#d97706"


REQUIRED_DATA_FILES = [
    "overview_metrics.csv",
    "rating_distribution.csv",
    "rating_by_year.csv",
    "rating_by_month.csv",
    "verified_stats.csv",
    "top_products.csv",
    "top_brands.csv",
    "user_activity.csv",
    "als_recommendations_sample.csv",
    "sentiment_metrics.csv",
    "sentiment_mismatch_examples.csv",
    "sentiment_top_words.csv",
]


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "sentiment_model.pkl")


def missing_files() -> list[str]:
    missing = [str(DATA_DIR / name) for name in REQUIRED_DATA_FILES if not (DATA_DIR / name).exists()]
    model_path = MODEL_DIR / "sentiment_model.pkl"
    if not model_path.exists():
        missing.append(str(model_path))
    return missing


def metric_map() -> dict[str, str]:
    df = load_csv("overview_metrics.csv")
    return dict(zip(df["metric"], df["value"], strict=False))


def fmt_int(value) -> str:
    return f"{int(float(value)):,}"


def fmt_float(value, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def is_unspecified_label(value) -> bool:
    return str(value).strip().lower() in UNSPECIFIED_LABELS


def shorten_text(value, limit: int = 68) -> str:
    text = "" if pd.isna(value) else str(value)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def prepare_ranked_category_chart(
    categories: pd.DataFrame,
    label_col: str,
    value_col: str = "n_reviews",
    top_n: int = 20,
    include_unspecified: bool = False,
) -> tuple[pd.DataFrame, int]:
    chart = categories.copy()
    chart[label_col] = chart[label_col].fillna("Unknown").astype(str).str.strip()
    unspecified = chart[label_col].map(is_unspecified_label)
    hidden_reviews = int(chart.loc[unspecified, value_col].sum())
    if not include_unspecified:
        chart = chart.loc[~unspecified].copy()
    sort_cols = [value_col]
    ascending = [False]
    if "avg_rating" in chart.columns:
        sort_cols.append("avg_rating")
        ascending.append(False)
    chart = chart.sort_values(sort_cols, ascending=ascending).head(top_n)
    return chart.sort_values(value_col), hidden_reviews


def prepare_recommendation_display(recommendations: pd.DataFrame) -> pd.DataFrame:
    display = recommendations.copy()
    display["display_rating"] = display["predicted_rating"].clip(lower=1, upper=5).round(2)
    return display


def predict_sentiment(model, text: str) -> dict[str, float | str | None]:
    try:
        prob = model.predict_proba([text])[0]
    except Exception as exc:
        return {"label": None, "positive_probability": None, "error": str(exc)}
    positive_probability = float(prob[1])
    return {
        "label": "Positive" if positive_probability >= 0.5 else "Negative",
        "positive_probability": positive_probability,
        "error": None,
    }


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f8fb;
            color: #172033;
        }
        .block-container {
            max-width: 1280px;
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #1f2937;
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dde4ee;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stPlotlyChart"] {
            background: #ffffff;
            border: 1px solid #dde4ee;
            border-radius: 8px;
            padding: 0.6rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        h1, h2, h3 {
            color: #111827;
        }
        .dashboard-kicker {
            color: #0f766e;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .dashboard-subtitle {
            color: #64748b;
            font-size: 0.98rem;
            margin-top: -0.35rem;
            margin-bottom: 1.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="dashboard-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="dashboard-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def style_figure(fig, height: int = 360):
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=12, t=24, b=8),
        font=dict(color="#334155", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def overview_page() -> None:
    metrics = metric_map()
    page_header(
        "Overview",
        "Amazon Movies & TV Reviews",
        f"Team {TEAM_NAME} | Amazon Reviews 2018 | Movies_and_TV 5-core",
    )

    cols = st.columns(5)
    cols[0].metric("Reviews", fmt_int(metrics.get("total_reviews", 0)))
    cols[1].metric("Users", fmt_int(metrics.get("total_users", 0)))
    cols[2].metric("Items", fmt_int(metrics.get("total_reviewed_items", 0)))
    cols[3].metric("Avg rating", fmt_float(metrics.get("avg_rating", 0), 2))
    cols[4].metric("Metadata match", f"{float(metrics.get('metadata_matched_review_ratio', 0)):.1%}")

    rating = load_csv("rating_distribution.csv")
    yearly = load_csv("rating_by_year.csv")
    monthly = load_csv("rating_by_month.csv")
    verified = load_csv("verified_stats.csv")

    left, right = st.columns(2)
    rating_fig = px.bar(
        rating,
        x="overall",
        y="pct_reviews",
        text="pct_reviews",
        labels={"overall": "Rating", "pct_reviews": "Share"},
        color_discrete_sequence=[CHART_BLUE],
    )
    rating_fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
    left.plotly_chart(
        style_figure(rating_fig, 330),
        use_container_width=True,
    )
    yearly_stable = yearly[yearly["n_reviews"] >= 100].copy()
    year_fig = px.line(
        yearly_stable,
        x="review_year",
        y="avg_rating",
        markers=True,
        labels={"review_year": "Year", "avg_rating": "Avg rating"},
        color_discrete_sequence=[CHART_TEAL],
    )
    right.plotly_chart(
        style_figure(year_fig, 330),
        use_container_width=True,
    )
    st.caption("Average-rating trend omits years with fewer than 100 reviews so a one-review year does not dominate the line.")

    volume_fig = px.area(
        monthly,
        x="review_ym",
        y="n_reviews",
        labels={"review_ym": "Month", "n_reviews": "Reviews"},
        color_discrete_sequence=[CHART_BLUE],
    )
    st.plotly_chart(style_figure(volume_fig, 340), use_container_width=True)

    verified_display = verified.copy()
    verified_display["verified_label"] = verified_display["verified"].map({True: "Verified purchase", False: "Unverified"})
    verified_fig = px.bar(
        verified_display,
        x="verified_label",
        y="avg_rating",
        color="verified_label",
        labels={"verified_label": "", "avg_rating": "Avg rating"},
        color_discrete_sequence=[CHART_AMBER, CHART_TEAL],
    )
    st.plotly_chart(style_figure(verified_fig, 300), use_container_width=True)


def product_page() -> None:
    page_header(
        "Products",
        "Product Explorer",
        "Review volume, rating quality, and metadata coverage across top Movies & TV items.",
    )
    products = load_csv("top_products.csv")
    products["brand"] = products["brand"].fillna("Unknown")
    products["primary_category"] = products["primary_category"].fillna("Unknown")

    brand_options = ["All"] + sorted(products["brand"].dropna().unique().tolist())[:200]
    category_options = ["All"] + sorted(products["primary_category"].dropna().unique().tolist())
    col1, col2, col3 = st.columns(3)
    brand = col1.selectbox("Brand", brand_options)
    category = col2.selectbox("Category", category_options)
    min_reviews = col3.number_input("Min reviews", min_value=0, value=50, step=25)

    filtered = products[products["n_reviews"] >= min_reviews].copy()
    if brand != "All":
        filtered = filtered[filtered["brand"] == brand]
    if category != "All":
        filtered = filtered[filtered["primary_category"] == category]
    filtered = filtered.sort_values(["n_reviews", "avg_rating"], ascending=False)

    filtered_chart = filtered.head(20).copy()
    filtered_chart["title_short"] = filtered_chart["title"].map(lambda value: shorten_text(value, 58))
    product_fig = px.bar(
        filtered_chart.sort_values("n_reviews"),
        x="n_reviews",
        y="title_short",
        orientation="h",
        hover_data={"title": True, "title_short": False, "avg_rating": ":.2f", "brand": True},
        labels={"n_reviews": "Reviews", "title_short": ""},
        color_discrete_sequence=[CHART_BLUE],
    )
    st.plotly_chart(
        style_figure(product_fig, 500),
        use_container_width=True,
    )
    display_cols = ["asin", "title", "brand", "primary_category", "n_reviews", "avg_rating", "price_num"]
    st.dataframe(filtered[[col for col in display_cols if col in filtered.columns]].head(200), use_container_width=True, hide_index=True)

    brands_raw = load_csv("top_brands.csv")
    include_unspecified = st.checkbox("Include unspecified/generic brand labels", value=False)
    brands, hidden_reviews = prepare_ranked_category_chart(brands_raw, "brand", top_n=20, include_unspecified=include_unspecified)
    hidden_share = hidden_reviews / max(float(brands_raw["n_reviews"].sum()), 1.0)
    st.metric(
        "Unspecified/generic brand reviews",
        fmt_int(hidden_reviews),
        "included" if include_unspecified else f"{hidden_share:.1%} hidden from chart",
    )
    brand_fig = px.bar(
        brands,
        x="n_reviews",
        y="brand",
        orientation="h",
        hover_data={"avg_rating": ":.2f", "n_products": True},
        labels={"n_reviews": "Reviews", "brand": ""},
        color_discrete_sequence=[CHART_TEAL],
    )
    st.plotly_chart(style_figure(brand_fig, 460), use_container_width=True)


def recommender_page() -> None:
    page_header(
        "Recommendations",
        "Recommender",
        "ALS sample recommendations with a popularity/high-rating fallback for uncovered users and items.",
    )
    recs = load_csv("als_recommendations_sample.csv")
    top_products = load_csv("top_products.csv")
    fallback = top_products.sort_values(["avg_rating", "n_reviews"], ascending=False).head(10)

    st.caption("ALS recommendations are shown for sampled known users. Cold-start or uncovered cases use globally popular, high-rating fallback products.")
    if recs.empty:
        st.dataframe(fallback[["asin", "title", "n_reviews", "avg_rating"]], use_container_width=True, hide_index=True)
        return

    recs["display_user"] = recs["reviewerID"].fillna(recs["user_id_idx"].astype(str))
    user = st.selectbox("Sample user", sorted(recs["display_user"].unique().tolist()))
    selected = recs[recs["display_user"] == user].sort_values("rank")
    if selected.empty:
        st.dataframe(fallback[["asin", "title", "n_reviews", "avg_rating"]], use_container_width=True, hide_index=True)
        return

    st.subheader("ALS Top-10 Recommendations")
    display = prepare_recommendation_display(selected)
    st.caption("ALS raw scores are used for ranking; displayed scores are clamped to the original 1-5 rating scale.")
    st.dataframe(display[["rank", "asin", "title", "display_rating"]], use_container_width=True, hide_index=True)
    st.subheader("Popularity/High-Rating Fallback")
    st.dataframe(fallback[["asin", "title", "n_reviews", "avg_rating"]], use_container_width=True, hide_index=True)


def sentiment_page() -> None:
    page_header(
        "Sentiment",
        "Sentiment Analyzer",
        "Weak-label review polarity model, coefficient signals, and mismatch examples.",
    )
    metrics = load_csv("sentiment_metrics.csv")
    metric_values = dict(zip(metrics["metric"], metrics["value"], strict=False))
    cols = st.columns(4)
    cols[0].metric("AUC", fmt_float(metric_values.get("auc", 0), 3))
    cols[1].metric("F1", fmt_float(metric_values.get("f1", 0), 3))
    cols[2].metric("Accuracy", fmt_float(metric_values.get("accuracy", 0), 3))
    cols[3].metric("Weighted recall", fmt_float(metric_values.get("weighted_recall", 0), 3))

    model = load_model()
    text = st.text_area("Review text", value="This movie was beautifully made and I would watch it again.", height=120)
    if text.strip():
        result = predict_sentiment(model, text)
        if result["error"]:
            st.warning(f"Sentiment model prediction is unavailable in this runtime: {result['error']}")
        else:
            st.metric("Prediction", result["label"], f"{result['positive_probability']:.1%} positive")

    words = load_csv("sentiment_top_words.csv")
    words = pd.concat([words[words["direction"] == "negative"].head(12), words[words["direction"] == "positive"].head(12)])
    word_fig = px.bar(
        words.sort_values("coefficient"),
        x="coefficient",
        y="word",
        color="direction",
        orientation="h",
        color_discrete_map={"positive": CHART_TEAL, "negative": "#be123c"},
    )
    st.plotly_chart(style_figure(word_fig, 420), use_container_width=True)

    mismatches = load_csv("sentiment_mismatch_examples.csv")
    mismatch_display = mismatches.copy()
    mismatch_display["review_preview"] = mismatch_display["reviewText"].map(lambda value: shorten_text(value, 220))
    mismatch_display["probability_positive"] = mismatch_display["probability_positive"].round(3)
    st.dataframe(
        mismatch_display[["asin", "title", "overall", "prediction", "probability_positive", "review_preview", "summary"]].head(50),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="Amazon Movies & TV Dashboard", layout="wide")
    apply_theme()
    st.sidebar.title("Amazon Reviews")
    st.sidebar.caption(f"Team {TEAM_NAME}")

    missing = missing_files()
    if missing:
        st.error("Missing deployment files:")
        st.write(missing)
        return

    page = st.sidebar.radio("Page", PAGES)
    if page == "Overview":
        overview_page()
    elif page == "Product Explorer":
        product_page()
    elif page == "Recommender":
        recommender_page()
    else:
        sentiment_page()


if __name__ == "__main__":
    main()
