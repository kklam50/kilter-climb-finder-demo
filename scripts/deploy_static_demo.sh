#!/usr/bin/env bash
# Copies the build output (static_demo/) into a separate clone of the Hugging
# Face *static* Space repo, then commits and pushes. The Space's git history is
# kept separate from this repo's.
#
# Usage: ./scripts/deploy_static_demo.sh <space-git-url> [deploy-dir]
# Run scripts/build_static_demo.py first. Authenticate with a write-scoped HF
# token as the git password.

set -euo pipefail

SPACE_URL="${1:?Usage: $0 <space-git-url> [deploy-dir]}"
DEPLOY_DIR="${2:-../kilter-climb-finder-demo-space}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/static_demo"

[ -d "$BUILD_DIR" ] || { echo "Missing $BUILD_DIR -- run scripts/build_static_demo.py first"; exit 1; }

if [ ! -d "$DEPLOY_DIR/.git" ]; then
  git clone "$SPACE_URL" "$DEPLOY_DIR"
else
  git -C "$DEPLOY_DIR" pull --ff-only
fi

find "$DEPLOY_DIR" -mindepth 1 -maxdepth 1 ! -name .git ! -name .gitattributes -exec rm -rf {} +
cp -r "$BUILD_DIR/." "$DEPLOY_DIR/"

cd "$DEPLOY_DIR"
git add -A
if git diff --cached --quiet; then
  echo "Nothing changed, skipping commit/push."
  exit 0
fi
git commit -m "Deploy $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push
echo "Pushed. Check build progress at $SPACE_URL"
