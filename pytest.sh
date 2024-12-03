#!/bin/bash
export PYTHONPATH=/home/krzysiek/Development/claim-modelling-kedro/claim_modelling_kedro:$PYTHONPATH
export PYTHONPATH=/home/krzysiek/Development/claim-modelling-kedro/claim_modelling_kedro/src:$PYTHONPATH
source venv/bin/activate && cd claim_modelling_kedro && pytest "$@"
