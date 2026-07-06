#!/usr/bin/env bash
# Point git at the tracked hooks dir. Run once after cloning.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath scripts/hooks
echo "git hooks enabled: core.hooksPath=scripts/hooks"
