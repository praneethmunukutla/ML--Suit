"""ML Suite — home page. Shows system state and recent activity."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import api_client as api  # noqa: E402

st.set_page_config(page_title="ML Suite", page_icon="🧪", layout="wide")

st.title("ML Suite")
st.caption("Bring any dataset. Pick X and y. Get a tuned, compared, reusable model.")

try:
    api.health()
    info = api.system_info()
    online = True
except api.ApiError as exc:
    online = False
    st.error(f"**{exc.message}**")
    if exc.detail:
        st.caption(str(exc.detail))
    st.code("cd ml_suite && ./run.sh api", language="bash")
    st.stop()

storage = info["storage"]
cols = st.columns(4)
cols[0].metric("Datasets", storage["datasets"])
cols[1].metric("Models trained", storage["models"])
cols[2].metric("Training runs", storage["runs"])
cols[3].metric("Storage used", f"{storage['disk_mb']} MB")

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("How it works")
    st.markdown(
        """
        1. **Data source** — upload a file, or pull from Postgres/Neon or MongoDB.
        2. **Configure** — review the column profile, pick features and a target.
           The task type is inferred for you.
        3. **Train** — every suitable model is tuned and compared on a held-out
           split.
        4. **Report** — leaderboard, diagnostics, and feature weights.
        5. **Predict** — reuse the winning model on new rows, any time.

        The saved model contains its own imputers, scalers, and encoders, so
        prediction takes raw rows in the original schema — no repeat setup.
        """
    )
    st.info("New here? Load `sample_data/customers_classification.csv` on the "
            "**Data Source** page and target the `churned` column.", icon="💡")

with right:
    st.subheader("Recent runs")
    try:
        runs = api.list_runs(limit=6)
    except api.ApiError:
        runs = []
    if not runs:
        st.caption("Nothing trained yet.")
    for run in runs:
        icon = {"succeeded": "✅", "failed": "❌",
                "running": "⏳", "queued": "•"}.get(run["status"], "•")
        best = run.get("best_model")
        score = run.get("best_score")
        detail = (f"{best} · {score:.4f}" if best and score is not None
                  else run.get("stage", run["status"]))
        st.markdown(f"{icon} **{run.get('dataset_name', '?')}** → "
                    f"`{run.get('target_column', '?')}`  \n"
                    f"<span style='color:#52514e;font-size:0.85em'>{detail}</span>",
                    unsafe_allow_html=True)

st.divider()
with st.expander("System details"):
    c1, c2 = st.columns(2)
    c1.markdown(f"**Python** {info['python']}  \n"
                f"**XGBoost** {'available' if info['xgboost_available'] else 'not installed'}  \n"
                f"**Storage backend** {info['storage_backend']}")
    limits = info["limits"]
    c2.markdown(f"**Max upload** {limits['max_upload_mb']} MB  \n"
                f"**Max training rows** {limits['max_train_rows']:,}  \n"
                f"**Concurrent runs** {limits['train_workers']}")
    st.caption(f"Classification models: {', '.join(info['models']['classification'])}")
    st.caption(f"Regression models: {', '.join(info['models']['regression'])}")
