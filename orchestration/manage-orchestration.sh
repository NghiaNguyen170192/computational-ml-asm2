#!/bin/bash

# Orchestration (Airflow + Postgres + Redis + MinIO + Nginx) Management Script
# Usage: ./manage-orchestration.sh [start|stop|restart|status|logs|build|reload|trigger <dag_id>|ps]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yaml"

print_status(){ echo -e "${BLUE}[INFO]${NC} $1"; }
print_success(){ echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warn(){ echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error(){ echo -e "${RED}[ERROR]${NC} $1"; }

dc() {
  docker compose -f "$COMPOSE_FILE" "$@"
}

start_stack(){
  print_status "Starting orchestration stack..."
  dc up -d --build postgres redis minio minio-init postgres-init airflow-init \
    airflow-webserver airflow-scheduler airflow-worker airflow-triggerer pgadmin nginx
  print_success "Orchestration started. Web UI: Airflow http://localhost:8080 (via Nginx)"
}

stop_stack(){
  print_status "Stopping orchestration stack..."
  dc stop airflow-webserver airflow-scheduler airflow-worker airflow-triggerer nginx pgadmin minio redis postgres
  print_success "Stopped."
}

restart_stack(){
  stop_stack
  start_stack
}

status_stack(){
  dc ps
}

logs_stack(){
  dc logs -f "$@"
}

build_stack(){
  print_status "Rebuilding Airflow images and restarting core services..."
  dc build airflow-webserver airflow-scheduler airflow-worker airflow-triggerer airflow-cli || true
  dc up -d airflow-webserver airflow-scheduler airflow-worker airflow-triggerer
  print_success "Rebuilt and restarted Airflow services."
}

reload_dags(){
  print_status "Reloading DAGs (scheduler + webserver)..."
  dc exec airflow-scheduler airflow dags reserialize || true
  # Nudge webserver to reload
  dc exec airflow-webserver bash -lc "pkill -f gunicorn || true"
  print_success "DAGs reloaded."
}

trigger_dag(){
  local dag_id="$1"
  if [ -z "$dag_id" ]; then
    print_error "Please provide a DAG id. Example: $0 trigger dag_binance_daily"
    exit 1
  fi
  print_status "Triggering DAG: ${dag_id}"
  dc exec airflow-webserver airflow dags trigger "$dag_id"
  print_success "Triggered ${dag_id}. Use '$0 logs airflow-worker' to watch."
}

usage(){
  cat <<EOF
Usage: $0 <command> [args]

Commands:
  start                 Start the full orchestration stack
  stop                  Stop the stack
  restart               Restart the stack
  status                Show container status
  ps                    Alias for status
  logs [service]        Tail logs (e.g. airflow-scheduler, airflow-worker)
  build                 Rebuild Airflow images and restart core services
  reload                Reload DAGs (scheduler + webserver)
  trigger <dag_id>      Trigger a DAG run

Examples:
  $0 start
  $0 reload
  $0 trigger dag_binance_daily
  $0 logs airflow-worker
EOF
}

cmd="$1"; shift || true

case "$cmd" in
  start) start_stack ;;
  stop) stop_stack ;;
  restart) restart_stack ;;
  status|ps) status_stack ;;
  logs) logs_stack "$@" ;;
  build) build_stack ;;
  reload) reload_dags ;;
  trigger) trigger_dag "$1" ;;
  ""|help|-h|--help) usage ;;
  *) print_error "Unknown command: $cmd"; usage; exit 1 ;;
esac


