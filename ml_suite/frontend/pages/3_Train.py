"""Watch a training run: live progress and a leaderboard that fills in."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import api_client as api  # noqa: E402
from lib import charts  # noqa: E402

st.set_page_config(page_title="Train · ML Suite", page_icon="🚀", layout="wide")
st.title("Training")

runs = []
try:
    runs = api.list_runs(limit=25)
except api.ApiError as exc:
    st.error(exc.message)
    st.stop()

run_id = st.session_state.get("run_id")
if runs:
    ids = [r["run_id"] for r in runs]
    index = ids.index(run_id) if run_id in ids else 0
    run_id = st.selectbox(
        "Run", ids, index=index,
        format_func=lambda i: next(
            f"{r['dataset_name']} → {r['target_column']}  ·  {r['status']}"
            f"  ·  {r['created_at']}" for r in runs if r["run_id"] == i))
    st.session_state["run_id"] = run_id

if not run_id:
    st.info("No runs yet.")
    st.page_link("pages/2_Configure.py", label="Configure a model", icon="⚙️")
    st.stop()

try:
    run = api.get_run(run_id)
except api.ApiError as exc:
    st.error(exc.message)
    st.stop()

status = run["status"]
badge = {"succeeded": "✅ Succeeded", "failed": "❌ Failed",
         "running": "⏳ Running", "queued": "• Queued"}.get(status, status)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"### {badge}")
c2.metric("Dataset", run.get("dataset_name", "—"))
c3.metric("Target", run.get("target_column", "—"))
c4.metric("Task", (run.get("task") or run.get("requested_task") or "auto").title())

if status in ("running", "queued"):
    st.progress(min(run.get("progress", 0.0), 1.0),
                text=f"{run.get('stage', 'working')} — "
                     f"{run.get('progress', 0) * 100:.0f}%")

if status == "failed":
    st.error(f"**{run.get('error', 'Training failed')}**")
    if run.get("error_detail"):
        st.code(str(run["error_detail"])[:2000])

leaderboard = run.get("leaderboard") or []
if leaderboard:
    metric = run.get("primary_metric") or run.get("requested_metric") or "score"
    done = [e for e in leaderboard if e["status"] == "ok"]
    failed = [e for e in leaderboard if e["status"] != "ok"]

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        if done:
            st.plotly_chart(charts.leaderboard(leaderboard, metric),
                            width="stretch")
    with right:
        st.subheader("Results so far")
        rows = []
        for entry in leaderboard:
            row = {"rank": entry.get("rank") or "—", "model": entry["model"],
                   "status": entry["status"]}
            row[metric] = (round(entry["primary_score"], 4)
                           if entry.get("primary_score") is not None else None)
            row["cv"] = entry.get("cv_score")
            row["seconds"] = entry.get("fit_seconds")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if failed:
        with st.expander(f"{len(failed)} model(s) failed"):
            for entry in failed:
                st.markdown(f"**{entry['model']}** — `{entry.get('error', '')}`")

if status == "succeeded":
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", run.get("best_model", "—"))
    c2.metric(run.get("primary_metric", "score"),
              f"{run.get('best_score', 0):.4f}" if run.get("best_score") is not None else "—")
    c3.metric("Elapsed", f"{run.get('elapsed_seconds', 0):.1f}s")
    st.session_state["model_id"] = run.get("model_id")
    a, b = st.columns(2)
    a.page_link("pages/4_Report.py", label="See the full report →", icon="📊")
    b.page_link("pages/5_Predict.py", label="Predict with this model →", icon="🔮")

# Poll while work is outstanding. Streamlit has no push channel, so the page
# reruns itself on a timer rather than holding the connection open.
if status in ("running", "queued"):
    time.sleep(2)
    st.rerun()
