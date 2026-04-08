#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p \
  artifacts/metrics \
  artifacts/models \
  artifacts/reports \
  artifacts/data \
  artifacts/predictions \
  artifacts/shap \
  data/raw \
  data/processed \
  logs

sudo chown -R "$(id -u)":"$(id -g)" artifacts data logs
chmod -R u+rwX artifacts data logs

find artifacts data logs -type d -exec chmod 775 {} \;
find artifacts data logs -type f -exec chmod 664 {} \;

cat > .env.docker.local <<ENVEOF
UID=$(id -u)
GID=$(id -g)
ENVEOF

echo "done"
