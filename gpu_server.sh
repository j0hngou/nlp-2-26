#!/usr/bin/env bash
# gpu_server.sh - SSH tunnel for remote GPU server access
#
# Usage:
#   1. Edit the variables below with your server details.
#   2. Run: bash gpu_server.sh
#   3. Open the printed URL in your browser to access Jupyter.
#
# If you are using Google Colab, you do NOT need this script.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
REMOTE_USER="${REMOTE_USER:-your_username}"
REMOTE_HOST="${REMOTE_HOST:-gpu-server.example.com}"
REMOTE_PORT="${REMOTE_PORT:-22}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
LOCAL_PORT="${LOCAL_PORT:-8888}"

# ── Start SSH tunnel + Jupyter ───────────────────────────────
echo "Connecting to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT} ..."
echo "Jupyter will be available at http://localhost:${LOCAL_PORT}"
echo "Press Ctrl+C to stop."

ssh -N -L "${LOCAL_PORT}:localhost:${JUPYTER_PORT}" \
    -p "${REMOTE_PORT}" \
    "${REMOTE_USER}@${REMOTE_HOST}"
