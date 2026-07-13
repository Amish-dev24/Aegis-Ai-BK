#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Aegis AI — EC2 evidence persistence setup (Option A: local disk)
#
# Run once on your AWS GPU/CPU instance after cloning the repo:
#   chmod +x scripts/setup_ec2_storage.sh
#   ./scripts/setup_ec2_storage.sh
#
# Optional — mount a dedicated EBS volume for evidence (recommended):
#   sudo EBS_DEVICE=/dev/nvme1n1 ./scripts/setup_ec2_storage.sh --mount-ebs
#
# IMPORTANT:
#   • Use EC2 **Stop/Start** — NOT **Terminate** — to keep the same disk.
#   • Always deploy from the SAME project directory so Docker bind mounts
#     (./evidence → /app/evidence) point at the folder that has your files.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[storage]${NC} $*"; }
warn()  { echo -e "${YELLOW}[storage]${NC} $*"; }
error() { echo -e "${RED}[storage]${NC} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MOUNT_EBS=false

for arg in "$@"; do
  case "$arg" in
    --mount-ebs) MOUNT_EBS=true ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *) error "Unknown argument: $arg (try --help)" ;;
  esac
done

STORAGE_DIRS=(evidence uploads processed_videos models)

# ── 1. Create local storage directories ───────────────────────────────────────
info "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

for dir in "${STORAGE_DIRS[@]}"; do
  mkdir -p "$dir"
  chmod 755 "$dir"
  info "✓ $dir/ exists ($(du -sh "$dir" 2>/dev/null | cut -f1 || echo '0'))"
done

# ── 2. Optional dedicated EBS volume ─────────────────────────────────────────
if $MOUNT_EBS; then
  EBS_DEVICE="${EBS_DEVICE:-/dev/nvme1n1}"
  MOUNT_POINT="${MOUNT_POINT:-/data/aegis-evidence}"

  if [ ! -b "$EBS_DEVICE" ]; then
    error "Block device not found: $EBS_DEVICE — attach an EBS volume in AWS console first."
  fi

  info "Setting up EBS volume $EBS_DEVICE → $MOUNT_POINT"

  if ! sudo blkid "$EBS_DEVICE" &>/dev/null; then
    warn "No filesystem on $EBS_DEVICE — formatting as ext4 (THIS ERASES THE VOLUME)"
    read -r -p "Type YES to format: " confirm
    [ "$confirm" = "YES" ] || error "Aborted."
    sudo mkfs -t ext4 "$EBS_DEVICE"
  fi

  sudo mkdir -p "$MOUNT_POINT"
  if ! mountpoint -q "$MOUNT_POINT"; then
    sudo mount "$EBS_DEVICE" "$MOUNT_POINT"
  fi

  UUID="$(sudo blkid -s UUID -o value "$EBS_DEVICE")"
  FSTAB_LINE="UUID=$UUID  $MOUNT_POINT  ext4  defaults,nofail  0  2"
  if ! grep -q "$MOUNT_POINT" /etc/fstab 2>/dev/null; then
    echo "$FSTAB_LINE" | sudo tee -a /etc/fstab >/dev/null
    info "Added fstab entry for auto-mount on reboot"
  fi

  # Symlink evidence (and optionally other dirs) to the persistent volume
  for dir in evidence processed_videos uploads; do
    target="$MOUNT_POINT/$dir"
    link="$PROJECT_ROOT/$dir"
    if [ -L "$link" ]; then
      info "$dir already symlinked → $(readlink "$link")"
      continue
    fi
    if [ -d "$link" ] && [ "$(ls -A "$link" 2>/dev/null | wc -l)" -gt 0 ]; then
      warn "Moving existing $dir/ contents to $target"
      sudo mkdir -p "$target"
      sudo mv "$link"/* "$target"/ 2>/dev/null || true
      rmdir "$link" 2>/dev/null || rm -rf "$link"
    fi
    sudo mkdir -p "$target"
    sudo chown -R "$(whoami):$(whoami)" "$MOUNT_POINT"
    ln -sfn "$target" "$link"
    info "✓ $dir → $target (symlink)"
  done
fi

# ── 3. Verify Docker bind-mount paths ─────────────────────────────────────────
info "Verifying docker-compose volume paths …"
for compose in docker-compose.yml docker-compose.gpu.yml; do
  if [ -f "$compose" ]; then
    grep -E '^\s+- \./(evidence|uploads|processed_videos)' "$compose" || true
  fi
done

# ── 4. Summary ────────────────────────────────────────────────────────────────
EVIDENCE_COUNT="$(find "$PROJECT_ROOT/evidence" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)"
info "Evidence snapshots on disk: $EVIDENCE_COUNT"

echo ""
info "Setup complete. Next steps:"
echo "  1. Deploy:  ./deploy.sh"
echo "  2. After detections, confirm files exist:  ls evidence/ | head"
echo "  3. Stop/start EC2 (do NOT terminate), then re-check:  ls evidence/ | head"
echo "  4. Test one image:  curl -I http://localhost:8000/evidence/<filename>.jpg"
echo ""
warn "If you TERMINATE the instance, this disk is destroyed unless you snapshot the EBS volume."
