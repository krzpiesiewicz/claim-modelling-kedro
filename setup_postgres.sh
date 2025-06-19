#!/usr/bin/env bash

set -euo pipefail

# Colors
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
CYAN=$(tput setaf 6)
YELLOW=$(tput setaf 3)
RESET=$(tput sgr0)
BOLD=$(tput bold)

# Config
DB_SUPERUSER="postgres"
PGDATA="/var/lib/postgres/data"
MLFLOW_USER="mlflow_user"
MLFLOW_DB="mlflow_db"
PASS_FILE="$(pwd)/.mlflow_password"

# Logging helpers
info()   { echo -e "${CYAN}ℹ️  $*${RESET}"; }
warn()   { echo -e "${YELLOW}⚠️  $*${RESET}"; }
error()  { echo -e "${RED}❌ $*${RESET}" >&2; }
success(){ echo -e "${GREEN}✅ $*${RESET}"; }

# Root check
if [[ "$EUID" -ne 0 ]]; then
  error "Run this script as root (use sudo)."
  exit 1
fi

# PostgreSQL check
if ! command -v postgres &>/dev/null; then
  error "PostgreSQL is not installed or not in PATH."
  exit 1
fi

# Ask for password
read -s -p "🔑 Enter new password for '${MLFLOW_USER}': " pw1; echo
read -s -p "🔑 Re-enter password: " pw2; echo
if [[ "$pw1" != "$pw2" || -z "$pw1" ]]; then
  error "Passwords do not match or are empty."
  exit 1
fi
MLFLOW_PASS="$pw1"

# Init DB if needed
if [[ ! -d "$PGDATA" || -z "$(ls -A "$PGDATA")" ]]; then
  info "Initializing PostgreSQL at $PGDATA ..."
  sudo -u "$DB_SUPERUSER" initdb --locale="$LANG" -E UTF8 -D "$PGDATA"
else
  info "PostgreSQL data directory already initialized."
fi

# Start + enable service
if command -v systemctl &>/dev/null; then
  systemctl enable --now postgresql
  success "PostgreSQL service started."
else
  warn "systemctl not found. Ensure PostgreSQL is running manually."
fi

# Create role
info "Creating user '${MLFLOW_USER}' if it doesn't exist..."
sudo -u "$DB_SUPERUSER" psql <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${MLFLOW_USER}')
  THEN CREATE ROLE ${MLFLOW_USER} LOGIN PASSWORD '${MLFLOW_PASS}'; END IF;
END\$\$;
SQL

# Create DB
info "Creating database '${MLFLOW_DB}' if it doesn't exist..."
sudo -u "$DB_SUPERUSER" psql -tAc "SELECT 1 FROM pg_database WHERE datname='${MLFLOW_DB}'" | grep -q 1 || \
  sudo -u "$DB_SUPERUSER" createdb -O "${MLFLOW_USER}" "${MLFLOW_DB}"

# Grant privileges
sudo -u "$DB_SUPERUSER" psql -c "GRANT ALL PRIVILEGES ON DATABASE ${MLFLOW_DB} TO ${MLFLOW_USER};"

# Save password
info "Saving password to ${PASS_FILE} ..."
echo "$MLFLOW_PASS" > "$PASS_FILE"
chmod 600 "$PASS_FILE"
chown "$(logname)":"$(logname)" "$PASS_FILE"

success "Setup complete!"
echo -e "${BOLD}Connection string:${RESET} postgresql://${MLFLOW_USER}:${MLFLOW_PASS}@localhost/${MLFLOW_DB}"
