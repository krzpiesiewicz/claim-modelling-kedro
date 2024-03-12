#!/bin/bash
export PYTHONPATH=/home/krzysiek/Development/solvency-modelling/solvency_models:$PYTHONPATH
export PYTHONPATH=/home/krzysiek/Development/solvency-modelling/solvency_models/src:$PYTHONPATH
source venv/bin/activate && cd solvency_models && pytest "$@"
