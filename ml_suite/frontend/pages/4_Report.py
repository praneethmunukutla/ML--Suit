"""The visual report for a finished run: ranking, diagnostics, and weights."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import api_client as api  # noqa: E402
from lib import charts  # noqa: E402

st.set_page_config(page_title="Report · ML Suite", page_icon="📊", layout="wide")
st.title("Model report")

try:
    runs = [r for r in api.list_runs(limit=50) if r["status"] == "succeeded"]
except api.ApiError as exc:
    st.error(exc.message)
    st.stop()

if not runs:
    st.info("No completed runs yet.")
    st.page_link("pages/2_Configure.py", label="Train a model", icon="⚙️")
    st.stop()

current = st.session_state.get("run_id")
ids = [r["run_id"] for r in runs]
run_id = st.selectbox(
    "Run", ids, index=ids.index(current) if current in ids else 0,
    format_func=lambda i: next(
        f"{r['dataset_name']} → {r['target_column']}  ·  {r['best_model']}"
        f"  ·  {r['created_at']}" for r in runs if r["run_id"] == i))

run = api.get_run(run_id)
metric = run["primary_metric"]
leaderboard = run["leaderboard"]
best = next(e for e in leaderboard if e["model"] == run["best_model"])
task = run["task"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Best model", run["best_model"])
c2.metric(metric, f"{run['best_score']:.4f}")
c3.metric("Train rows", f"{run['n_train']:,}")
c4.metric("Test rows", f"{run['n_test']:,}")
c5.metric("Total time", f"{run['elapsed_seconds']:.1f}s")
st.caption(f"Task inferred as **{task}** — {run['task_reason']}. "
           f"Scores below are on the held-out test split, never seen during tuning.")

st.divider()
left, right = st.columns([3, 2])
with left:
    st.plotly_chart(charts.leaderboard(leaderboard, metric), width="stretch")
with right:
    st.subheader("Winning configuration")
    st.json(best["best_params"], expanded=True)
    st.caption(f"Chosen from {best.get('n_candidates', '?')} sampled combinations "
               f"by {run['cv_folds']}-fold cross-validation.")

metric_names = (["accuracy", "f1", "precision", "recall"]
                if task == "classification"
                else ["r2", "explained_variance"])
st.plotly_chart(charts.metric_comparison(leaderboard, metric_names), width="stretch")

st.divider()
st.subheader("Diagnostics for the winning model")
diagnostics = best.get("diagnostics", {})
d1, d2 = st.columns(2)

if task == "classification":
    with d1:
        if diagnostics.get("confusion_matrix"):
            st.plotly_chart(
                charts.confusion_matrix(diagnostics["confusion_matrix"],
                                        diagnostics["class_labels"]),
                width="stretch")
    with d2:
        st.markdown("**All metrics**")
        st.dataframe(pd.DataFrame([
            {"metric": k, "value": round(v, 4)}
            for k, v in best["metrics"].items() if v is not None
        ]), width="stretch", hide_index=True)
        matrix = diagnostics.get("confusion_matrix")
        if matrix and len(matrix) == 2:
            (tn, fp), (fn, tp) = matrix
            st.caption(f"Of {tp + fn} actual positives the model caught {tp}; "
                       f"it raised {fp} false alarms out of {tn + fp} negatives.")
else:
    with d1:
        if diagnostics.get("actual"):
            st.plotly_chart(
                charts.predicted_vs_actual(diagnostics["actual"],
                                           diagnostics["predicted"]),
                width="stretch")
    with d2:
        if diagnostics.get("residuals"):
            st.plotly_chart(
                charts.residuals(diagnostics["predicted"],
                                 diagnostics["residuals"]),
                width="stretch")
    st.dataframe(pd.DataFrame([
        {"metric": k, "value": round(v, 4)}
        for k, v in best["metrics"].items()
        if v is not None and not k.startswith("neg_")
    ]), width="stretch", hide_index=True)

st.divider()
f1, f2 = st.columns([3, 2])
with f1:
    st.plotly_chart(charts.feature_importance(best.get("feature_importance", [])),
                    width="stretch")
with f2:
    st.subheader("Preprocessing applied")
    prep = run["preprocessing"]
    for label, key in [("Numeric — imputed & scaled", "numeric"),
                       ("Categorical — one-hot encoded", "categorical"),
                       ("High cardinality — ordinal encoded", "high_cardinality"),
                       ("Datetime — expanded into parts", "datetime"),
                       ("Dropped as identifiers", "dropped")]:
        if prep.get(key):
            st.markdown(f"**{label}**  \n"
                        f"<span style='font-size:0.85em;color:#52514e'>"
                        f"{', '.join(prep[key])}</span>", unsafe_allow_html=True)

st.divider()
st.subheader("Full leaderboard")
rows = []
for entry in leaderboard:
    row = {"rank": entry.get("rank") or "—", "model": entry["model"],
           "status": entry["status"]}
    row.update({k: (round(v, 4) if v is not None else None)
                for k, v in entry.get("metrics", {}).items()
                if not k.startswith("neg_")})
    row["cv score"] = entry.get("cv_score")
    row["fit seconds"] = entry.get("fit_seconds")
    rows.append(row)
table = pd.DataFrame(rows)
st.dataframe(table, width="stretch", hide_index=True)
st.download_button("Download leaderboard as CSV",
                   table.to_csv(index=False).encode(),
                   file_name=f"leaderboard_{run_id}.csv", mime="text/csv")

st.info(f"The winning model is saved as `{run['model_id']}` and is ready to use "
        f"on the Predict page.", icon="💾")
