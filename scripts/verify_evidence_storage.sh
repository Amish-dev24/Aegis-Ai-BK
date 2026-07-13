#!/usr/bin/env bash
# Quick check: evidence files on disk vs what the API can serve.
# Usage: ./scripts/verify_evidence_storage.sh [api_base_url]
set -euo pipefail

API_BASE="${1:-http://localhost:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVIDENCE_DIR="$PROJECT_ROOT/evidence"

echo "=== Aegis evidence storage check ==="
echo "Project:  $PROJECT_ROOT"
echo "Evidence: $EVIDENCE_DIR"
echo "API:      $API_BASE"
echo ""

if [ ! -d "$EVIDENCE_DIR" ]; then
  echo "ERROR: evidence/ directory missing — run: ./scripts/setup_ec2_storage.sh"
  exit 1
fi

ON_DISK="$(find "$EVIDENCE_DIR" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)"
echo "JPEG files on disk: $ON_DISK"

if [ "$ON_DISK" -eq 0 ]; then
  echo "WARN: No .jpg files in evidence/ — images will 404 even if RDS has records."
  exit 1
fi

SAMPLE="$(find "$EVIDENCE_DIR" -maxdepth 1 -name '*.jpg' | head -1)"
SAMPLE_NAME="$(basename "$SAMPLE")"
URL="$API_BASE/evidence/$SAMPLE_NAME"

HTTP_CODE="$(curl -s -o /dev/null -w '%{http_code}' "$URL" || echo "000")"
echo "Sample file: $SAMPLE_NAME"
echo "HTTP $HTTP_CODE for $URL"

if [ "$HTTP_CODE" = "200" ]; then
  echo "OK — storage and API serving are working."
  exit 0
fi

echo "FAIL — file exists on disk but API returned $HTTP_CODE."
echo "Check: docker compose ps, volume mounts, and that API is running."
exit 1
