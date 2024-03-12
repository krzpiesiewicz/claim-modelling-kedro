#!/bin/bash
KEDRO_DIR="solvency_models"

LOGS_DIR="logs"
#echo -e "> source venv/bin/activate"
source venv/bin/activate
#echo  -e "> cd $KEDRO_DIR"
cd $KEDRO_DIR

export KEDRO_LOGGING_CONFIG=$(pwd)/conf/logging.yml

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

INDEX_FILE="$LOGS_DIR/index.txt"
exec 200<>$INDEX_FILE
if [ $? -ne 0 ]; then
  echo "Error: Unable to open file descriptor for $INDEX_FILE"
  exit 1
fi

flock 200
if [ $? -ne 0 ]; then
  echo "Error: Unable to lock file descriptor for $INDEX_FILE"
  exit 1
fi

# Create a temporary file
temp_file=$(mktemp)

# Copy the content of the file descriptor to the temporary file
cat <&200 >$temp_file

if [ -f $temp_file ]; then
  index_line=$( tail -n 1 $temp_file )
else
  index_line=""
fi

echo index_line: $index_line

if [ -z "$index_line" ]; then
  NUMBER=0
else
  NUMBER=$(echo $index_line | grep -Eo '^run: [0-9]+' /dev/stdin)
  NUMBER="${NUMBER//run:/}"
fi

NUMBER=$(($NUMBER+1))

DATETIME=$(date '+%F %T')
ESC_DATETIME="${DATETIME//:/}"
ESC_DATETIME="${ESC_DATETIME// /_}"

LOG_NAME="log_${ESC_DATETIME}_run_$NUMBER"
LOG_DIR="$LOGS_DIR/$LOG_NAME"
FULL_LOG_DIR_PATH=$(pwd)"/$LOG_DIR"

KEDRO_ARGS="$@"

#if ! [ -z $1 ]; then
#  PIPELINE=$1
#  PIPELINE_OPT="--pipeline $PIPELINE"
#else
#  PIPELINE="__default__"
#  PIPELINE_OPT=""
#fi

echo "Kedro run: $NUMBER"
echo "Time: $DATETIME"
echo -e "Logging to: $LOG_DIR\n"
mkdir $LOG_DIR
# save new index line to the file descriptor
echo -e "run: $NUMBER  ($DATETIME),  log-dir: $LOG_NAME" >&200
# relase the lock of the file descriptor
flock -u 200

#echo "> kedro run $KEDRO_ARGS"
echo "> kedro run $KEDRO_ARGS --kedro-run-no $NUMBER --kedro-runtime '$DATETIME' --kedro-logdir-path $FULL_LOG_DIR_PATH --kedro-log-html-file $FULL_LOG_DIR_PATH/$LOG_NAME.html --kedro-logs-index-file $INDEX_FILE"

shopt -s expand_aliases
alias decolor.styles='sed -E "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})*)?[m,K,H,f,J]//gm"'
alias decolor.reset='sed -E "s/\x1B\([A-Z]{1}(\x1B\[[m,K,H,f,J])?//gm"'
alias decolor='decolor.styles | decolor.reset'


STATUS_FILE=$LOG_DIR/status.txt
START_TIME=$(date +%s)
#script --flush --quiet --command "kedro run $KEDRO_ARGS --kedro-run-no $NUMBER --kedro-runtime '$DATETIME' --kedro-logdir-path $FULL_LOG_DIR_PATH --kedro-log-html-file $FULL_LOG_DIR_PATH/$LOG_NAME.html --kedro-logs-index-file $INDEX_FILE" | tee /dev/tty $LOG_DIR/$LOG_NAME.log | aha > $LOG_DIR/$LOG_NAME.html
script --return --flush --quiet --command "kedro run $KEDRO_ARGS --kedro-run-no $NUMBER --kedro-runtime '$DATETIME' --kedro-logdir-path $FULL_LOG_DIR_PATH --kedro-log-html-file $FULL_LOG_DIR_PATH/$LOG_NAME.html --kedro-logs-index-file $INDEX_FILE && (echo \"succeeded\" > $STATUS_FILE) || (echo \"failed\" > $STATUS_FILE)" | tee /dev/tty > $LOG_DIR/$LOG_NAME.log
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
  echo -e "\nKedro run $NUMBER completed successfully."
  echo -e "\nKedro run $NUMBER completed successfully." >> $LOG_DIR/$LOG_NAME.log
else
  echo -e "\nKedro run $NUMBER failed."
  echo -e "\nKedro run $NUMBER failed." >> $LOG_DIR/$LOG_NAME.log
fi
echo -e "Kedro execution time: $ELAPSED_TIME_STR."
echo -e "Kedro execution time: $ELAPSED_TIME_STR." >> $LOG_DIR/$LOG_NAME.log

cat $LOG_DIR/$LOG_NAME.log | aha > $LOG_DIR/$LOG_NAME.html

cat $LOG_DIR/$LOG_NAME.log | tr -cd '\11\12\15\40-\176' | sed 's/]8.*$//' | sed 's/\[[0-9]*m//g' | sed 's/\[[0-9]*;[0-9]*m//g' | sed 's/\[[0-9]*;[0-9]*;[0-9]*m//g' > $LOG_DIR/$LOG_NAME.clean.log

LOGFILE_KEDRO_LOG_TO_MLFLOW=$LOG_DIR/$LOG_NAME--kedro-log-to-mlflow
HTML_LOG_FILE=$FULL_LOG_DIR_PATH/$LOG_NAME.html
status=&(script --return --flush --quiet --command "kedro run --pipeline dummy --kedro-log-to-mlflow --kedro-run-no $NUMBER --kedro-run-status $status --kedro-elapsed-time $ELAPSED_TIME --kedro-logdir-path $FULL_LOG_DIR_PATH --kedro-log-html-file $HTML_LOG_FILE" > $LOGFILE_KEDRO_LOG_TO_MLFLOW.log)
status=$?
cat $LOGFILE_KEDRO_LOG_TO_MLFLOW.log | tr -cd '\11\12\15\40-\176' | sed 's/]8.*$//' | sed 's/\[[0-9]*m//g' | sed 's/\[[0-9]*;[0-9]*m//g' | sed 's/\[[0-9]*;[0-9]*;[0-9]*m//g' > $LOGFILE_KEDRO_LOG_TO_MLFLOW.clean.log

if [ $status -eq 0 ]; then
  echo -e "\nSaved kedro log to mlflow ($(basename $HTML_LOG_FILE))."
else
  echo -e "\nFailed to save kedro log to mlflow => See file://$(pwd)/$LOGFILE_KEDRO_LOG_TO_MLFLOW.clean.log for more details."
fi