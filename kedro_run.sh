#!/bin/bash
KEDRO_DIR="claim_modelling_kedro"
LOGS_DIR="logs/kedro"

# === Show help if no arguments or --help ===
if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
  cat <<EOF
Usage: ./kedro_run.sh [KEDRO_ARGS...] [--no-notify]

Runs a Kedro pipeline and logs output to a timestamped log directory.

Arguments:
  [KEDRO_ARGS...]       Arguments passed directly to 'kedro run'
                        (e.g. --pipeline ds --mlflow-run-id 42424242424242424)

Options:
  --no-notify           Suppress desktop notifications and sound at the end.
                        This flag can appear anywhere and is not passed to Kedro.

Examples:
  ./kedro_run.sh --pipeline all_to_test
  ./kedro_run.sh --pipeline all_to_test --no-notify
  ./kedro_run.sh --no-notify --pipeline all_to_test

------------------------------------------------------------
Kedro CLI help (from \`kedro run --help\`):
------------------------------------------------------------
EOF

  source venv/bin/activate
  export PYTHONPATH="${PYTHONPATH}:$(pwd)/claim_modelling_kedro/src"
  cd claim_modelling_kedro || exit 1
  kedro run --help || echo "⚠️ Could not display Kedro help. Is your project configured correctly?"
  exit 0
fi

# === PARSE ARGS ===
NO_NOTIFY=false
KEDRO_ARGS=""

for arg in "$@"; do
  if [[ "$arg" == "--no-notify" ]]; then
    NO_NOTIFY=true
  else
    # Escape and append argument
    KEDRO_ARGS="$KEDRO_ARGS $(printf "%q" "$arg")"
  fi
done

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
script --return --flush --quiet --command '
  # --- trap Ctrl+C inside the “script” session ---
  trap "echo 130 > "'"$STATUS_FILE"'" ; exit 130" INT

  # --- run Kedro ---
  kedro run \
    '"$KEDRO_ARGS"' \
    --kedro-run-no '"$NUMBER"' \
    --kedro-runtime "'"$DATETIME"'" \
    --kedro-logdir-path "'"$FULL_LOG_DIR_PATH"'" \
    --kedro-log-html-file "'"$FULL_LOG_DIR_PATH/$LOG_NAME.html"'" \
    --kedro-logs-index-file "'"$INDEX_FILE"'"

  # --- capture normal exit code ---
  code=$?
  echo $code > "'"$STATUS_FILE"'"
  exit $code
' | tee /dev/tty > "$LOG_DIR/$LOG_NAME.log"

if [[ -f "$STATUS_FILE" ]]; then
  status=$(cat "$STATUS_FILE")
  echo -e "Kedro run $NUMBER completed with status: $status."
else
  echo -e "❌ Status file '$STATUS_FILE' not found."
  echo -e "Kedro run $NUMBER completed but status could not be determined."
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
status2=&(script --return --flush --quiet --command "kedro run --pipeline dummy --kedro-log-to-mlflow --kedro-run-no $NUMBER --kedro-run-status $status --kedro-elapsed-time $ELAPSED_TIME --kedro-logdir-path $FULL_LOG_DIR_PATH --kedro-log-html-file $HTML_LOG_FILE" > $LOGFILE_KEDRO_LOG_TO_MLFLOW.log)
status2=$?
cat $LOGFILE_KEDRO_LOG_TO_MLFLOW.log | tr -cd '\11\12\15\40-\176' | sed 's/]8.*$//' | sed 's/\[[0-9]*m//g' | sed 's/\[[0-9]*;[0-9]*m//g' | sed 's/\[[0-9]*;[0-9]*;[0-9]*m//g' > $LOGFILE_KEDRO_LOG_TO_MLFLOW.clean.log

if [ $status2 -eq 0 ]; then
  msg="\nSaved kedro log to mlflow ($(basename $HTML_LOG_FILE))."
else
  msg="\nFailed to save kedro log to mlflow => See file://$(pwd)/$LOGFILE_KEDRO_LOG_TO_MLFLOW.clean.log for more details."
fi
echo -e "$msg"

if [ $status -ne 0 ]; then
  RUN_ID_FILE="$LOG_DIR/mlflow_run_id.txt"
  RUN_ID=$(cat "$RUN_ID_FILE")
  MLFLOW_SERVER_CONFIG_PATH="conf/local/mlflow.yml"
  python3 - <<EOF
import yaml
import sys
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# Ścieżka do konfiguracji
config_path = Path("$MLFLOW_SERVER_CONFIG_PATH")
if not config_path.exists():
    print(f"Config file not found: {config_path}")
    sys.exit(1)

# Wczytaj konfigurację YAML
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

try:
    tracking_uri = config["server"]["mlflow_tracking_uri"]
except KeyError:
    print("mlflow_tracking_uri not found in config")
    sys.exit(1)

# Ustaw i oznacz run jako FAILED
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

try:
    client.set_terminated("$RUN_ID", status="FAILED")
    print(f"Run $RUN_ID has been marked as FAILED on {tracking_uri}")
except Exception as e:
    print(f"Failed to set status for run $RUN_ID: {e}")
    sys.exit(2)
EOF
fi

if ! $NO_NOTIFY; then
  # Notification parameters
  TITLE="Finished"
  MESSAGE="Kedro run completed with status: $status.\n$msg"
  APP_NAME="Claim Modelling Kedro"
  SOUND_FILE="/usr/share/sounds/freedesktop/stereo/complete.oga"

  # Send notification
  notify-send -a "$APP_NAME" -u normal -t 0 "$TITLE" "$MESSAGE"

  # Play sound with fallback if the file doesn't exist
  if [[ -f "$SOUND_FILE" ]]; then
      paplay "$SOUND_FILE"
  else
      echo "⚠️ Sound file not found: $SOUND_FILE" >&2
  fi
fi

exit "$status"