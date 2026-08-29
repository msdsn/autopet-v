#!/bin/bash
# Sync the working tree to a remote GPU box: scripts/env/sync.sh <hostname|alias> [--git]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
NAME=${1:?usage: sync.sh <host-alias-or-hostname> [--git]}
HOST=$(grep -E "^${NAME}=" "$ROOT/.colab_hosts" 2>/dev/null | cut -d= -f2); HOST=${HOST:-$NAME}
PASS=${SSHPASS:?set SSHPASS in .env}
SSH="sshpass -p $PASS ssh -o ConnectTimeout=40 -o StrictHostKeyChecking=no"
REMOTE=/content/autopet
if [ "${2:-}" = "--git" ]; then
  git -C "$ROOT" push origin main
  $SSH "$HOST" "cd $REMOTE && git pull --ff-only && git rev-parse --short HEAD"
else
  $SSH "$HOST" "mkdir -p $REMOTE"
  rsync -az --delete --exclude '__pycache__' --exclude '.git' --exclude 'autoPETV' --exclude 'internal' \
        -e "$SSH" "$ROOT/" "$HOST:$REMOTE/"
fi
echo "synced -> $HOST:$REMOTE"
