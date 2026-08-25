"""Reuse a saved model: score typed rows or a whole file."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import api_client as api  # noqa: E402

st.set_page_config(page_title="Predict · ML Suite", page_icon="🔮", layout="wide")
st.title("Predict")
st.caption("Pass raw rows in the original schema — the model carries its own "
           "preprocessing.")

try:
    models = api.list_models()
except api.ApiError as exc:
    st.error(exc.message)
    st.stop()

if not models:
    st.info("No trained models yet.")
    st.page_link("pages/2_Configure.py", label="Train one", icon="⚙️")
    st.stop()

current = st.session_state.get("model_id")
ids = [m["model_id"] for m in models]
model_id = st.selectbox(
    "Model", ids, index=ids.index(current) if current in ids else 0,
    format_func=lambda i: next(
        f"{m['algorithm']} · {m['dataset_name']} → {m['target_column']} · "
        f"{m['primary_metric']}={m['metrics'].get(m['primary_metric'], 0):.4f}"
        for m in models if m["model_id"] == i))
meta = next(m for m in models if m["model_id"] == model_id)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Algorithm", meta["algorithm"])
c2.metric("Task", meta["task"].title())
c3.metric(meta["primary_metric"], f"{meta['metrics'].get(meta['primary_metric'], 0):.4f}")
c4.metric("Size", f"{meta['size_kb']} KB")
st.caption(f"Trained on **{meta['dataset_name']}** to predict "
           f"**{meta['target_column']}** · saved {meta['created_at']}")

required = meta["feature_columns"]
with st.expander(f"Required input columns ({len(required)})"):
    st.code(", ".join(required))
    if meta.get("dropped_features"):
        st.caption(f"Still required in the input, but ignored by the model: "
                   f"{', '.join(meta['dropped_features'])}")

tab_form, tab_file, tab_manage = st.tabs(["Single row", "Batch file", "Manage"])

with tab_form:
    st.markdown("Enter values for one row:")
    values: dict = {}
    columns = st.columns(3)
    for i, col in enumerate(required):
        values[col] = columns[i % 3].text_input(col, key=f"in_{col}")
    if st.button("Predict", type="primary"):
        row = {}
        for key, raw in values.items():
            if raw == "":
                row[key] = None
                continue
            try:  # send numbers as numbers so dtypes match training
                row[key] = float(raw) if "." in raw else int(raw)
            except ValueError:
                row[key] = raw
        try:
            result = api.predict_rows(model_id, [row])
            prediction = result["predictions"][0]
            st.success(f"### {meta['target_column']} = **{prediction}**")
            if result.get("probabilities"):
                probs = result["probabilities"][0]
                st.dataframe(pd.DataFrame([
                    {"class": k, "probability": round(v, 4)}
                    for k, v in sorted(probs.items(), key=lambda kv: -kv[1])
                ]), width="stretch", hide_index=True)
        except api.ApiError as exc:
            st.error(exc.message)
            if exc.detail:
                st.caption(str(exc.detail))

with tab_file:
    st.markdown("Upload rows to score in bulk. Extra columns are ignored; the "
                "required ones must be present.")
    uploaded = st.file_uploader("CSV, Excel, JSON, or Parquet",
                                type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
                                key="predict_upload")
    if uploaded is not None and st.button("Score file", type="primary"):
        with st.spinner("Scoring…"):
            try:
                csv_bytes = api.predict_file_csv(model_id, uploaded.getvalue(),
                                                 uploaded.name)
                scored = pd.read_csv(io.BytesIO(csv_bytes))
                st.success(f"Scored {len(scored):,} rows.")
                st.dataframe(scored.head(200), width="stretch")
                st.download_button("Download predictions", csv_bytes,
                                   file_name=f"predictions_{model_id}.csv",
                                   mime="text/csv", type="primary")
                target_col = f"predicted_{meta['target_column']}"
                if meta["task"] == "classification" and target_col in scored:
                    from lib import charts
                    counts = scored[target_col].value_counts().to_dict()
                    st.plotly_chart(
                        charts.class_distribution({str(k): int(v)
                                                   for k, v in counts.items()}),
                        width="stretch")
            except api.ApiError as exc:
                st.error(exc.message)
                if exc.detail:
                    st.caption(str(exc.detail))

with tab_manage:
    frame = pd.DataFrame([{
        "algorithm": m["algorithm"], "task": m["task"],
        "dataset": m["dataset_name"], "target": m["target_column"],
        "metric": m["primary_metric"],
        "score": round(m["metrics"].get(m["primary_metric"], 0), 4),
        "size KB": m["size_kb"], "created": m["created_at"],
        "model_id": m["model_id"],
    } for m in models])
    st.dataframe(frame, width="stretch", hide_index=True)
    st.markdown("**Use this model from your own code:**")
    sample_fields = ", ".join(f'"{c}": ...' for c in required[:3])
    if api.mode() == "http":
        st.code(f'''import requests

response = requests.post(
    "{api.API_BASE}/api/predict/{model_id}",
    json={{"rows": [{{{sample_fields}, ...}}]}},
)
print(response.json()["predictions"])''', language="python")
    else:
        # No API is being served in embedded mode; load the pipeline directly.
        st.code(f'''import joblib, pandas as pd

pipeline = joblib.load("storage/models/{model_id}.joblib")
rows = pd.DataFrame([{{{sample_fields}, ...}}])
print(pipeline.predict(rows))''', language="python")
        st.caption("Running in-process, so there is no REST endpoint. Start the "
                   "API with `./run.sh api` if you want one.")
    if st.button("Delete this model"):
        try:
            api.delete_model(model_id)
            st.rerun()
        except api.ApiError as exc:
            st.error(exc.message)
