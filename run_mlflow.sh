#!/usr/bin/env bash

set -euo pipefail

# === CONFIG ===
DB_USER="mlflow_user"
DB_NAME="mlflow_db"
DB_HOST="localhost"
DB_PORT="5432"
PASS_FILE=".mlflow_password"
VENV_PATH="venv"
MLFLOW_PORT="5000"
MLFLOW_HOST="127.0.0.1"

# === CHECKS ===
if [[ ! -f "$PASS_FILE" ]]; then
  echo "❌ Password file '$PASS_FILE' not found. Run setup_postgresql.sh first." >&2
  exit 1
fi

if [[ ! -d "$VENV_PATH" ]]; then
  echo "❌ Virtualenv '$VENV_PATH' not found. Please create it first." >&2
  exit 1
fi

# === READ PASSWORD ===
MLFLOW_PASS=$(<"$PASS_FILE")

# === Construct DATABASE URI ===
export MLFLOW_TRACKING_URI="postgresql://${DB_USER}:${MLFLOW_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

# === Activate environment and launch MLflow ===
source "${VENV_PATH}/bin/activate"
echo "🚀 Starting MLflow server on ${MLFLOW_HOST}:${MLFLOW_PORT}"
echo "📡 Using database: ${MLFLOW_TRACKING_URI}"

exec mlflow server \
  --backend-store-uri "${MLFLOW_TRACKING_URI}" \
  --default-artifact-root "./mlruns" \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}"
