"""Review the column profile and choose features, target, and search settings."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import api_client as api  # noqa: E402
from lib import charts  # noqa: E402

st.set_page_config(page_title="Configure · ML Suite", page_icon="⚙️", layout="wide")
st.title("Configure")

dataset_id = st.session_state.get("dataset_id")
if not dataset_id:
    st.warning("No dataset selected yet.")
    st.page_link("pages/1_Data_Source.py", label="Go to Data Source", icon="📁")
    st.stop()

try:
    profile = api.get_profile(dataset_id)
except api.ApiError as exc:
    st.error(exc.message)
    st.stop()

st.caption(f"**{profile['name']}** · `{dataset_id}`")
cols = st.columns(5)
cols[0].metric("Rows", f"{profile['rows']:,}")
cols[1].metric("Columns", profile["columns"])
cols[2].metric("Missing cells", f"{profile['total_missing']:,}")
cols[3].metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
cols[4].metric("In memory", f"{profile['memory_mb']} MB")

profiles = profile["column_profiles"]

tab_cols, tab_preview, tab_quality = st.tabs(
    ["Columns", "Preview", "Data quality"])

# How each column kind will be handled, shown so the choice is not a black box.
KIND_HELP = {
    "numeric": "median-imputed, then standardised",
    "categorical": "mode-imputed, then one-hot encoded",
    "high_cardinality": "mode-imputed, then ordinal encoded + scaled",
    "boolean": "treated as a two-level category",
    "datetime": "expanded into year/month/day/weekday/hour",
    "identifier": "dropped — too many distinct values to carry signal",
    "empty": "dropped — no values",
}

with tab_cols:
    frame = pd.DataFrame([{
        "column": p["name"], "detected as": p["kind"], "dtype": p["dtype"],
        "missing %": p["missing_pct"], "unique": p["unique"],
        "will be": KIND_HELP.get(p["kind"], ""),
        "example": ", ".join(str(s) for s in p["sample"][:2]),
    } for p in profiles])
    st.dataframe(frame, width="stretch", hide_index=True)

with tab_preview:
    st.dataframe(pd.DataFrame(profile["preview"]), width="stretch")

with tab_quality:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.missing_values(profiles), width="stretch")
    with c2:
        dropped = [p["name"] for p in profiles
                   if p["kind"] in ("identifier", "empty")]
        if dropped:
            st.warning(f"**Will be dropped as identifiers:** {', '.join(dropped)}",
                       icon="⚠️")
        high = [p["name"] for p in profiles if p["missing_pct"] > 40]
        if high:
            st.warning(f"**Over 40% missing:** {', '.join(high)} — consider "
                       "excluding these.", icon="⚠️")
        if profile["duplicate_rows"]:
            st.info(f"{profile['duplicate_rows']:,} duplicate rows are present. "
                    "They are kept as-is.", icon="ℹ️")
        if not dropped and not high and not profile["duplicate_rows"]:
            st.success("No quality problems detected.", icon="✅")

st.divider()
st.subheader("Select target and features")

names = [p["name"] for p in profiles]
suggested = profile["suggested_targets"]

c1, c2 = st.columns([1, 2])
with c1:
    default_target = names.index(suggested[-1]) if suggested else 0
    target = st.selectbox("Target column (y)", names, index=default_target,
                          help="What you want the model to predict")
    try:
        preview = api.task_preview(dataset_id, target)
        st.success(f"**{preview['task'].title()}** — {preview['reason']}", icon="🎯")
        task_override = st.radio(
            "Task type", ["auto", "classification", "regression"],
            horizontal=True,
            help="Override only if the inference above is wrong")
        metric = st.selectbox(
            "Optimise for", preview["metrics"],
            index=preview["metrics"].index(preview["default_metric"]))
        available_models = preview["models"]
    except api.ApiError as exc:
        st.error(exc.message)
        st.stop()

with c2:
    usable = [n for n in names if n != target]
    auto_exclude = {p["name"] for p in profiles
                    if p["kind"] in ("identifier", "empty")}
    features = st.multiselect(
        "Feature columns (X)", usable,
        default=[n for n in usable if n not in auto_exclude],
        help="Identifier-like columns are excluded by default")
    if not features:
        st.error("Select at least one feature.")
    excluded = set(usable) - set(features)
    if excluded:
        st.caption(f"Excluded: {', '.join(sorted(excluded))}")

    target_profile = next(p for p in profiles if p["name"] == target)
    if target_profile.get("top_values"):
        st.plotly_chart(charts.class_distribution(target_profile["top_values"]),
                        width="stretch")

with st.expander("Search settings"):
    s1, s2, s3, s4 = st.columns(4)
    test_size = s1.slider("Test split", 0.1, 0.4, 0.2, 0.05,
                          help="Held out and never seen during tuning")
    cv_folds = s2.slider("CV folds", 2, 10, 3,
                         help="More folds is more reliable but slower")
    search_iter = s3.slider("Search iterations", 3, 60, 12,
                            help="Hyperparameter combinations tried per model")
    s4.metric("Approx. fits", f"{len(available_models) * cv_folds * search_iter:,}")
    chosen_models = st.multiselect(
        "Models to compare", available_models, default=available_models,
        help="Fewer models finishes sooner")

st.divider()
ready = bool(features and chosen_models)
if st.button("Start training", type="primary", disabled=not ready, width="stretch"):
    payload = {
        "dataset_id": dataset_id,
        "feature_columns": features,
        "target_column": target,
        "task": None if task_override == "auto" else task_override,
        "metric": metric,
        "models": chosen_models,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "search_iter": search_iter,
    }
    try:
        run = api.start_training(payload)
        st.session_state["run_id"] = run["run_id"]
        st.switch_page("pages/3_Train.py")
    except api.ApiError as exc:
        st.error(exc.message)
        if exc.detail:
            st.caption(str(exc.detail))
