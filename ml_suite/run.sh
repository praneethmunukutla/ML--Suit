#!/usr/bin/env bash
# Launcher for the ML Suite. Usage:
#   ./run.sh          start both API and dashboard
#   ./run.sh api      API only        (http://127.0.0.1:8000/docs)
#   ./run.sh ui       dashboard only  (http://127.0.0.1:8501)
#   ./run.sh stop     stop both
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
API_PORT="${MLSUITE_PORT:-8000}"
UI_PORT="${MLSUITE_UI_PORT:-8501}"

if [ ! -x "$PY" ]; then
  echo "No virtualenv found. Create one with:"
  echo "  python3 -m venv --system-site-packages .venv"
  echo "  .venv/bin/pip install -r requirements.txt"
  exit 1
fi

stop_port() {
  local pids
  pids=$(lsof -ti:"$1" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "Stopping process on port $1"
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

start_api() {
  stop_port "$API_PORT"
  mkdir -p logs
  nohup "$PY" -m uvicorn backend.main:app \
    --host 127.0.0.1 --port "$API_PORT" > logs/api.log 2>&1 &
  echo "API      → http://127.0.0.1:$API_PORT/docs   (logs/api.log)"
}

start_ui() {
  stop_port "$UI_PORT"
  mkdir -p logs
  nohup "$PY" -m streamlit run frontend/app.py \
    --server.port "$UI_PORT" --server.headless true \
    --browser.gatherUsageStats false > logs/ui.log 2>&1 &
  echo "Dashboard → http://127.0.0.1:$UI_PORT              (logs/ui.log)"
}

case "${1:-all}" in
  api) start_api ;;
  ui)  start_ui ;;
  stop) stop_port "$API_PORT"; stop_port "$UI_PORT"; echo "Stopped." ;;
  all)
    start_api
    # Give uvicorn a moment so the dashboard's first health check succeeds.
    sleep 4
    start_ui
    echo
    echo "Open http://127.0.0.1:$UI_PORT to begin.  Stop with ./run.sh stop"
    ;;
  *) echo "Usage: ./run.sh [api|ui|stop|all]"; exit 1 ;;
esac
