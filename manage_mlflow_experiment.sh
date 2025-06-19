#!/bin/bash

# 🧪 Uruchamia skrypt do zarządzania eksperymentami MLflow

# Ścieżka do virtualnego środowiska
VENV_PATH="./venv"

# Ścieżka do katalogu ze skryptem Pythona
PY_SCRIPT="claim_modelling_kedro/src/claim_modelling_kedro/experiments/manage_mlflow_experiment.py"

# Sprawdź, czy istnieje venv
if [ ! -d "$VENV_PATH" ]; then
  echo "❌ Venv not found at $VENV_PATH. Activate it or create one first."
  exit 1
fi

# Aktywuj środowisko i uruchom skrypt
source "$VENV_PATH/bin/activate"
python "$PY_SCRIPT" "$@"
