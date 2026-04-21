#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Aegis AI — Deployment script
#
# Usage:
#   ./deploy.sh                  # deploy latest commit on current branch
#   ./deploy.sh <commit-id>      # deploy a specific git commit (full or short hash)
#   ./deploy.sh main             # deploy latest on branch 'main'
#
# The script auto-detects whether an NVIDIA GPU is present and uses:
#   GPU  → Dockerfile.gpu + docker-compose.gpu.yml
#   CPU  → Dockerfile     + docker-compose.yml
#
# Override GPU detection manually:
#   FORCE_GPU=true  ./deploy.sh
#   FORCE_GPU=false ./deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()  { echo -e "${YELLOW}[deploy]${NC} $*"; }
error() { echo -e "${RED}[deploy]${NC} $*" >&2; exit 1; }

# ── 1. Resolve commit ID ──────────────────────────────────────────────────────
TARGET=${1:-}

if [ -z "$TARGET" ]; then
  # No argument: use current HEAD
  git fetch --quiet origin
  COMMIT_ID=$(git rev-parse HEAD)
  info "No commit specified — using current HEAD: ${COMMIT_ID:0:12}"
else
  # Argument given: could be a full hash, short hash, or branch name
  git fetch --quiet origin
  COMMIT_ID=$(git rev-parse "$TARGET" 2>/dev/null) \
    || error "Cannot resolve '$TARGET' to a commit. Run 'git log --oneline' to list commits."
  info "Deploying commit: ${TARGET} → ${COMMIT_ID:0:12}"
fi

SHORT_ID="${COMMIT_ID:0:12}"

# ── 2. Check out the target commit ───────────────────────────────────────────
CURRENT=$(git rev-parse HEAD)
if [ "$COMMIT_ID" != "$CURRENT" ]; then
  warn "Checking out ${SHORT_ID} (detached HEAD) …"
  git checkout "$COMMIT_ID"
else
  info "Already at ${SHORT_ID}, no checkout needed."
fi

# ── 3. Auto-detect GPU ───────────────────────────────────────────────────────
if [ "${FORCE_GPU:-}" = "true" ]; then
  USE_GPU=true
elif [ "${FORCE_GPU:-}" = "false" ]; then
  USE_GPU=false
elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  USE_GPU=true
else
  USE_GPU=false
fi

if $USE_GPU; then
  COMPOSE_FILE="docker-compose.gpu.yml"
  info "GPU detected (Tesla P100) — using ${COMPOSE_FILE}"
else
  COMPOSE_FILE="docker-compose.yml"
  warn "No GPU detected — using CPU compose (${COMPOSE_FILE})"
fi

# ── 4. Build image tagged with commit hash ───────────────────────────────────
# Use --no-cache only when explicitly requested (FRESH_BUILD=true ./deploy.sh)
info "Building image aegis-bk:${SHORT_ID} …"
BUILD_FLAGS="--build-arg COMMIT_ID=$COMMIT_ID"
if [ "${FRESH_BUILD:-}" = "true" ]; then
  warn "FRESH_BUILD=true — ignoring Docker cache (slow, use only when requirements.txt changed)"
  BUILD_FLAGS="$BUILD_FLAGS --no-cache"
fi
COMMIT_ID="$COMMIT_ID" docker compose -f "$COMPOSE_FILE" build $BUILD_FLAGS

# Tag with short hash for easy rollback
docker tag "$(docker compose -f "$COMPOSE_FILE" images -q api 2>/dev/null | head -1)" \
  "aegis-bk:${SHORT_ID}" 2>/dev/null || true

# ── 5. Roll over (stop old, start new) ───────────────────────────────────────
info "Rolling over to ${SHORT_ID} …"
COMMIT_ID="$COMMIT_ID" docker compose -f "$COMPOSE_FILE" up -d --remove-orphans

# ── 6. Wait for health check ──────────────────────────────────────────────────
info "Waiting for API to become healthy …"
MAX_WAIT=120
ELAPSED=0
until curl -sf http://localhost:8000/health &>/dev/null; do
  sleep 3
  ELAPSED=$((ELAPSED + 3))
  if [ $ELAPSED -ge $MAX_WAIT ]; then
    error "API did not become healthy within ${MAX_WAIT}s. Check logs: docker compose -f ${COMPOSE_FILE} logs api"
  fi
done

info "✓ Deployment complete — commit ${SHORT_ID} is live."
info "  Logs:     docker compose -f ${COMPOSE_FILE} logs -f api"
info "  Rollback: ./deploy.sh <previous-commit-id>"
