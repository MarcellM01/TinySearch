#!/usr/bin/env sh
set -eu

# Glama wraps the configured command with mcp-proxy. This script only prepares
# the bundled local SearXNG process, then execs the raw TinySearch stdio MCP
# server. Do not run mcp-proxy inside this script.

: "${SEARXNG_URL:=http://127.0.0.1:8080/search}"
: "${TINYSEARCH_MODELS_DIR:=/data/models}"
: "${SEARXNG_SETTINGS_PATH:=/etc/searxng/settings.yml}"
: "${SEARXNG_SRC_DIR:=/opt/searxng}"
: "${MCP_TRANSPORT:=stdio}"
: "${MCP_HOST:=0.0.0.0}"
: "${SEARXNG_STARTUP_WAIT_SECONDS:=5}"

export SEARXNG_URL
export TINYSEARCH_MODELS_DIR
export SEARXNG_SETTINGS_PATH
export MCP_TRANSPORT
export MCP_HOST

# Run SearXNG directly from the source checkout instead of installing it as a
# wheel/package. This avoids SearXNG's isolated Python build step, which can
# fail inside generated container builders even after requirements are installed.
export PYTHONPATH="${SEARXNG_SRC_DIR}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$TINYSEARCH_MODELS_DIR"

echo "Starting bundled SearXNG on 127.0.0.1:8080..." >&2
(
  cd "$SEARXNG_SRC_DIR"
  /opt/searxng-venv/bin/python -m searx.webapp
) >&2 2>&2 &
SEARXNG_PID="$!"

cleanup() {
  if kill -0 "$SEARXNG_PID" 2>/dev/null; then
    kill "$SEARXNG_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

echo "Waiting ${SEARXNG_STARTUP_WAIT_SECONDS}s for SearXNG to initialize..." >&2
sleep "$SEARXNG_STARTUP_WAIT_SECONDS"

echo "Starting TinySearch stdio MCP server..." >&2
exec /opt/tinysearch-venv/bin/python servers/mcp_server.py
