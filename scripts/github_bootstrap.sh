#!/usr/bin/env bash
# Authenticate once and push this project's reviewed main branch to the user-designated repository.
# This script never creates a different repository, changes visibility, or pushes legacy branches.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
repository_url="https://github.com/yuanshu777/eco_prepro.git"

if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "Switch to the reviewed main branch before publishing." >&2
  exit 1
fi
if ! gh auth status --hostname github.com >/dev/null 2>&1; then
  gh auth login --hostname github.com --git-protocol https --web
fi
gh auth setup-git --hostname github.com
if git remote get-url origin >/dev/null 2>&1; then
  if [[ "$(git remote get-url origin)" != "$repository_url" ]]; then
    echo "origin does not match the user-designated repository; inspect the remote before publishing." >&2
    exit 1
  fi
else
  git remote add origin "$repository_url"
fi
git push -u origin main:main
echo "Uploaded to https://github.com/yuanshu777/eco_prepro"
