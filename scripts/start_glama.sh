#!/usr/bin/env sh
set -eu

# Glama builds TinySearch into a single container. This script is used when the
# container also includes a local SearXNG source checkout, so Glama can run
# TinySearch as one MCP server image while TinySearch still uses a
# SearXNG-compatible backend internally.

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

echo "Starting bundled SearXNG on 127.0.0.1:8080..."
(
  cd "$SEARXNG_SRC_DIR"
  /opt/searxng-venv/bin/python -m searx.webapp
) &
SEARXNG_PID="$!"

cleanup() {
  if kill -0 "$SEARXNG_PID" 2>/dev/null; then
    kill "$SEARXNG_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

echo "Waiting ${SEARXNG_STARTUP_WAIT_SECONDS}s for SearXNG to initialize..."
sleep "$SEARXNG_STARTUP_WAIT_SECONDS"

echo "Starting TinySearch MCP server through mcp-proxy..."
exec mcp-proxy -- /opt/tinysearch-venv/bin/python servers/mcp_server.py
