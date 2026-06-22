# Amazon Movies & TV Dashboard

Streamlit dashboard for the Amazon Reviews 2018 `Movies_and_TV` 5-core big data course project.

## Live App

[https://amazon-movies-tv-dashboard.streamlit.app/](https://amazon-movies-tv-dashboard.streamlit.app/)

## Deployment Boundary

The deployed app reads only prepared small artifacts under `dashboard/data` and `dashboard/models`. It does not start Spark, connect to HDFS, or require the raw Amazon review files at runtime.

This keeps the public dashboard lightweight: the Spark/HDFS pipeline produces the exported CSV/PKL files offline, and Streamlit only serves the prepared analytical results.

## Included Artifacts

- `dashboard/data/*.csv`: overview metrics, rating trends, product summaries, ALS recommendation samples, and sentiment outputs.
- `dashboard/models/sentiment_model.pkl`: the trained sentiment model used by the Sentiment Analyzer page.
- `dashboard/app.py`: the Streamlit application.

Large raw Amazon files, HDFS directories, Spark intermediate outputs, and local environment files are intentionally excluded from this repository.

## Pages

- Overview
- Product Explorer
- Recommender
- Sentiment Analyzer

## Run Locally

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Streamlit Community Cloud

Use these settings when deploying:

- Repository: `zkzkzzz/amazon-movies-tv-dashboard`
- Branch: `main`
- Main file path: `dashboard/app.py`

Public app URL:

```text
https://amazon-movies-tv-dashboard.streamlit.app/
```
