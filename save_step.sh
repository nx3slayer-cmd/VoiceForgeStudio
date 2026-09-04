#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
source venv/bin/activate

COMMIT_MSG="${1:-Update VoiceForge Studio}"
TAG_NAME="${2:-}"

echo "======================================================"
echo "  [1/2] Running Voice Model & Engine Verification...  "
echo "======================================================"
pytest test_multi_voice.py -v

echo ""
echo "======================================================"
echo "  [2/2] ✓ Tests Passed! Committing to GitHub...       "
echo "======================================================"

git add -A
git commit -m "$COMMIT_MSG"

if [ -n "$TAG_NAME" ]; then
    git tag -a "$TAG_NAME" -m "$COMMIT_MSG"
    git push origin main --tags
else
    git push origin main
fi

echo "======================================================"
echo "  🎉 Successfully Verified, Committed & Pushed!       "
echo "======================================================"
