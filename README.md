# Amazon Movies & TV Dashboard

Streamlit dashboard for the Amazon Reviews 2018 `Movies_and_TV` 5-core big data course project.

The deployed app reads only prepared small artifacts under `dashboard/data` and `dashboard/models`. It does not start Spark, connect to HDFS, or require the raw Amazon review files.

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

After deployment, the expected public app URL is:

```text
https://amazon-movies-tv-dashboard.streamlit.app
```

## Pages

- Overview
- Product Explorer
- Recommender
- Sentiment Analyzer

