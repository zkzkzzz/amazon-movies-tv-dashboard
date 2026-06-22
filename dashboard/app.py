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


def overview_page() -> None:
    metrics = metric_map()
    st.title("Amazon Movies & TV Reviews")
    st.caption(f"Team {TEAM_NAME} | Amazon Reviews 2018 | Movies_and_TV 5-core")

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
    left.plotly_chart(
        px.bar(rating, x="overall", y="pct_reviews", labels={"overall": "Rating", "pct_reviews": "Share"}),
        use_container_width=True,
    )
    right.plotly_chart(
        px.line(yearly, x="review_year", y="avg_rating", markers=True, labels={"review_year": "Year", "avg_rating": "Avg rating"}),
        use_container_width=True,
    )
    st.plotly_chart(px.line(monthly, x="review_ym", y="n_reviews", labels={"review_ym": "Month", "n_reviews": "Reviews"}), use_container_width=True)
    st.plotly_chart(px.bar(verified, x="verified", y="avg_rating", labels={"verified": "Verified", "avg_rating": "Avg rating"}), use_container_width=True)


def product_page() -> None:
    st.title("Product Explorer")
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

    st.plotly_chart(
        px.bar(filtered.head(20).sort_values("n_reviews"), x="n_reviews", y="title", orientation="h", labels={"n_reviews": "Reviews", "title": ""}),
        use_container_width=True,
    )
    display_cols = ["asin", "title", "brand", "primary_category", "n_reviews", "avg_rating", "price_num"]
    st.dataframe(filtered[[col for col in display_cols if col in filtered.columns]].head(200), use_container_width=True, hide_index=True)

    brands = load_csv("top_brands.csv").head(25)
    st.plotly_chart(px.bar(brands.sort_values("n_reviews"), x="n_reviews", y="brand", orientation="h", labels={"n_reviews": "Reviews", "brand": ""}), use_container_width=True)


def recommender_page() -> None:
    st.title("Recommender")
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
    st.dataframe(selected[["rank", "asin", "title", "predicted_rating"]], use_container_width=True, hide_index=True)
    st.subheader("Popularity/High-Rating Fallback")
    st.dataframe(fallback[["asin", "title", "n_reviews", "avg_rating"]], use_container_width=True, hide_index=True)


def sentiment_page() -> None:
    st.title("Sentiment Analyzer")
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
        prob = model.predict_proba([text])[0]
        pred = int(prob[1] >= 0.5)
        st.metric("Prediction", "Positive" if pred == 1 else "Negative", f"{prob[1]:.1%} positive")

    words = load_csv("sentiment_top_words.csv")
    words = pd.concat([words[words["direction"] == "negative"].head(12), words[words["direction"] == "positive"].head(12)])
    st.plotly_chart(px.bar(words, x="coefficient", y="word", color="direction", orientation="h"), use_container_width=True)

    mismatches = load_csv("sentiment_mismatch_examples.csv")
    st.dataframe(mismatches[["asin", "title", "overall", "prediction", "probability_positive", "reviewText", "summary"]].head(50), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Amazon Movies & TV Dashboard", layout="wide")
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
