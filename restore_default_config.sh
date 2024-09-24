#!/bin/bash
KEDRO_DIR="solvency_models"

#echo -e "> source venv/bin/activate"
source venv/bin/activate
#echo  -e "> cd $KEDRO_DIR"
cd $KEDRO_DIR

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "$(pwd)"


echo "> python3 src/solvency_models/experiments/restore_default_config.py"

shopt -s expand_aliases
alias decolor.styles='sed -E "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})*)?[m,K,H,f,J]//gm"'
alias decolor.reset='sed -E "s/\x1B\([A-Z]{1}(\x1B\[[m,K,H,f,J])?//gm"'
alias decolor='decolor.styles | decolor.reset'


# script --flush --command "python3 src/solvency_models/experiments/restore_default_config.py"
script --return --flush --quiet --command "python3 src/solvency_models/experiments/restore_default_config.py" | tee /dev/tty > /dev/null
