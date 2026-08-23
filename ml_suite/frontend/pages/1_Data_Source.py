"""Ingest data from a file, a SQL database, or MongoDB."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import api_client as api  # noqa: E402

st.set_page_config(page_title="Data Source · ML Suite", page_icon="📁", layout="wide")
st.title("Data source")
st.caption("Everything lands in the same table format, whatever it came from.")

tab_file, tab_sql, tab_mongo, tab_existing = st.tabs(
    ["File upload", "SQL database", "MongoDB", "Already loaded"])


def _register(meta: dict) -> None:
    st.session_state["dataset_id"] = meta["dataset_id"]
    st.success(f"Loaded **{meta['name']}** — {meta['rows']:,} rows × "
               f"{meta['columns']} columns  \n`{meta['dataset_id']}`")
    st.page_link("pages/2_Configure.py", label="Next: configure the model →",
                 icon="➡️")


with tab_file:
    uploaded = st.file_uploader(
        "CSV, TSV, Excel, JSON, or Parquet",
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet"])
    name = st.text_input("Name for this dataset (optional)")
    if uploaded is not None:
        st.caption(f"{uploaded.name} · {uploaded.size / 1024:.0f} KB")
        if st.button("Load file", type="primary"):
            with st.spinner("Reading and cleaning…"):
                try:
                    _register(api.upload_dataset(
                        uploaded.getvalue(), uploaded.name, name or None))
                except api.ApiError as exc:
                    st.error(exc.message)
                    if exc.detail:
                        st.caption(str(exc.detail))

    sample_dir = Path(__file__).resolve().parents[2] / "sample_data"
    samples = sorted(sample_dir.glob("*.csv")) if sample_dir.exists() else []
    if samples:
        st.divider()
        st.caption("Or try a bundled sample:")
        for path in samples:
            if st.button(f"Load {path.name}", key=f"sample_{path.name}"):
                try:
                    _register(api.upload_dataset(path.read_bytes(), path.name, path.stem))
                except api.ApiError as exc:
                    st.error(exc.message)

with tab_sql:
    st.markdown("Runs a **read-only** query. Write statements are rejected.")
    uri = st.text_input(
        "Connection URI",
        placeholder="postgresql+psycopg2://user:password@host/dbname?sslmode=require",
        type="password",
        help="Neon, Supabase, RDS, local Postgres/MySQL, or sqlite:///file.db")
    query = st.text_area("Query", value="SELECT * FROM your_table", height=120)
    c1, c2 = st.columns(2)
    limit = c1.number_input("Row limit", 100, 1_000_000, 50_000, step=1000)
    sql_name = c2.text_input("Dataset name", key="sql_name")
    if st.button("Run query", type="primary", disabled=not (uri and query)):
        with st.spinner("Querying…"):
            try:
                _register(api.dataset_from_sql(uri, query, sql_name or None, int(limit)))
            except api.ApiError as exc:
                st.error(exc.message)
                if exc.detail:
                    st.code(str(exc.detail)[:1500])
    st.caption("The connection URI is used for this query only and is never stored.")

with tab_mongo:
    mongo_uri = st.text_input("MongoDB URI",
                              placeholder="mongodb+srv://user:password@cluster.mongodb.net",
                              type="password")
    c1, c2 = st.columns(2)
    database = c1.text_input("Database")
    collection = c2.text_input("Collection")
    filter_text = st.text_area("Filter (JSON)", value="{}", height=90)
    c3, c4 = st.columns(2)
    mongo_limit = c3.number_input("Document limit", 100, 1_000_000, 50_000,
                                  step=1000, key="mongo_limit")
    mongo_name = c4.text_input("Dataset name", key="mongo_name")
    ready = all([mongo_uri, database, collection])
    if st.button("Fetch documents", type="primary", disabled=not ready):
        import json
        try:
            filter_doc = json.loads(filter_text or "{}")
        except json.JSONDecodeError as exc:
            st.error(f"Filter is not valid JSON: {exc}")
            filter_doc = None
        if filter_doc is not None:
            with st.spinner("Fetching…"):
                try:
                    _register(api.dataset_from_mongo(
                        mongo_uri, database, collection, filter_doc,
                        int(mongo_limit), mongo_name or None))
                except api.ApiError as exc:
                    st.error(exc.message)
                    if exc.detail:
                        st.code(str(exc.detail)[:1500])
    st.caption("Nested documents are flattened into columns automatically.")

with tab_existing:
    try:
        datasets = api.list_datasets()
    except api.ApiError as exc:
        st.error(exc.message)
        datasets = []
    if not datasets:
        st.info("No datasets loaded yet.")
    else:
        frame = pd.DataFrame([{
            "name": d["name"], "rows": d["rows"], "columns": d["columns"],
            "source": d["source"], "created": d["created_at"],
            "dataset_id": d["dataset_id"],
        } for d in datasets])
        st.dataframe(frame, width="stretch", hide_index=True)
        choice = st.selectbox(
            "Select a dataset to work with",
            [d["dataset_id"] for d in datasets],
            format_func=lambda i: next(
                f"{d['name']} ({d['rows']:,} rows)" for d in datasets
                if d["dataset_id"] == i))
        c1, c2 = st.columns([1, 4])
        if c1.button("Use this", type="primary"):
            st.session_state["dataset_id"] = choice
            st.success(f"Selected `{choice}`")
            st.page_link("pages/2_Configure.py", label="Next: configure →", icon="➡️")
        if c2.button("Delete", key="del_ds"):
            try:
                api.delete_dataset(choice)
                st.rerun()
            except api.ApiError as exc:
                st.error(exc.message)
