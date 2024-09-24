#!/bin/bash
KEDRO_DIR="solvency_models"
LOGS_DIR="logs/run_experiment"

#echo -e "> source venv/bin/activate"
source venv/bin/activate
#echo  -e "> cd $KEDRO_DIR"
cd $KEDRO_DIR

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

ARGS="$@"

echo "$(pwd)"


echo "> python3 src/solvency_models/experiments/run_experiment.py $ARGS"

shopt -s expand_aliases
alias decolor.styles='sed -E "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})*)?[m,K,H,f,J]//gm"'
alias decolor.reset='sed -E "s/\x1B\([A-Z]{1}(\x1B\[[m,K,H,f,J])?//gm"'
alias decolor='decolor.styles | decolor.reset'

DATETIME=$(date '+%F %T')
ESC_DATETIME="${DATETIME//:/}"
ESC_DATETIME="${ESC_DATETIME// /_}"

LOG_NAME="log_${ESC_DATETIME}"
LOG_DIR="$LOGS_DIR/$LOG_NAME"

echo -e "Logging to: $LOG_DIR\n"
mkdir $LOG_DIR

STATUS_FILE=$LOG_DIR/status.txt

START_TIME=$(date +%s)

script --return --flush --quiet --command "python3 src/solvency_models/experiments/run_experiment.py $ARGS && (echo \"succeeded\" > $STATUS_FILE) || (echo \"failed\" > $STATUS_FILE)" | tee /dev/tty > $LOG_DIR/$LOG_NAME.log
if [ -f $STATUS_FILE ] && grep -Fxq "succeeded" $STATUS_FILE; then
    status=0
else
    status=1
fi
END_TIME=$(date +%s)
ELAPSED_TIME=$(($END_TIME-$START_TIME))
ELAPSED_HOURS=$(($ELAPSED_TIME/3600))
ELAPSED_MINUTES=$(($ELAPSED_TIME/60%60))
ELAPSED_SECONDS=$(($ELAPSED_TIME%60))

ELAPSED_TIME_STR=""
if [ $ELAPSED_HOURS -gt 0 ]; then
  ELAPSED_TIME_STR="$ELAPSED_HOURS hours"
fi
if [ $ELAPSED_MINUTES -gt 0 ] || [ $ELAPSED_HOURS -gt 0 ]; then
  [ -n "$ELAPSED_TIME_STR" ] && ELAPSED_TIME_STR="$ELAPSED_TIME_STR, "
  ELAPSED_TIME_STR="${ELAPSED_TIME_STR}$ELAPSED_MINUTES minutes"
fi
if [ $ELAPSED_SECONDS -gt 0 ] || [ $ELAPSED_MINUTES -gt 0 ] || [ $ELAPSED_HOURS -gt 0 ]; then
  [ -n "$ELAPSED_TIME_STR" ] && ELAPSED_TIME_STR="$ELAPSED_TIME_STR, "
  ELAPSED_TIME_STR="${ELAPSED_TIME_STR}$ELAPSED_SECONDS seconds"
fi

if [ $status -eq 0 ]; then
  echo -e "\n(${ESC_DATETIME}) run_experiment.py $ARGS completed successfully."
  echo -e "\n(${ESC_DATETIME}) run_experiment.py $ARGS completed successfully." >> $LOG_DIR/$LOG_NAME.log
else
  echo -e "\n(${ESC_DATETIME}) run_experiment.py $ARGS failed."
  echo -e "\n(${ESC_DATETIME}) run_experiment.py $ARGS failed." >> $LOG_DIR/$LOG_NAME.log
fi
echo -e "(${ESC_DATETIME}) run_experiment.py execution time: $ELAPSED_TIME_STR."
echo -e "(${ESC_DATETIME}) run_experiment.py execution time: $ELAPSED_TIME_STR." >> $LOG_DIR/$LOG_NAME.log

cat $LOG_DIR/$LOG_NAME.log | aha > $LOG_DIR/$LOG_NAME.html

cat $LOG_DIR/$LOG_NAME.log | tr -cd '\11\12\15\40-\176' | sed 's/]8.*$//' | sed 's/\[[0-9]*m//g' | sed 's/\[[0-9]*;[0-9]*m//g' | sed 's/\[[0-9]*;[0-9]*;[0-9]*m//g' > $LOG_DIR/$LOG_NAME.clean.log
