#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

#Verify the frozen records before changing source formatting
python -m scripts.verify_frozen_artifacts | tee logs/15_repository_integrity.txt
python -m scripts.check_checkpoint_loading | tee logs/17_checkpoint_loading.txt
python -m scripts.check_comment_style

#Use the Ruff formatter and verify that all Python syntax trees are unchanged
if ! python -m pip show ruff >/dev/null 2>&1; then
    echo "Installing Ruff for source formatting and PEP 8 checks"
    python -m pip install ruff
fi
python -m ruff check --select E,F,W --ignore E501 src scripts
python -m compileall -q src scripts
python -m scripts.verify_frozen_artifacts >/dev/null
python -m scripts.check_checkpoint_loading >/dev/null
printf '%s\n' "Artifact hashes, checkpoint loading, comments, and lint: PASSED"

#Stage repository files using the explicit exclusions in .gitignore
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git init -b main
fi
git branch -M main
git add -A
python -m scripts.check_git_stage
git diff --cached --stat

#Require a real human to approve public publication
printf '\nPublish these staged files to PUBLIC GitHub repository AfonsoAzevedo/ds312-adversarial-ml? [y/N] '
read -r response
if [[ "$response" != "y" && "$response" != "Y" ]]; then
    echo "Nothing committed or published. All changes remain staged."
    exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
    echo "GitHub CLI is missing. Install it with: brew install gh"
    exit 2
fi
if ! gh auth status >/dev/null 2>&1; then
    echo "GitHub CLI is not authenticated. Run: gh auth login"
    exit 2
fi
account=$(gh api user --jq .login)
if [[ "$account" != "AfonsoAzevedo" ]]; then
    echo "GitHub CLI is authenticated as $account, expected AfonsoAzevedo"
    exit 2
fi
if ! git var GIT_AUTHOR_IDENT >/dev/null 2>&1; then
    echo "Git user.name or user.email is missing. Configure identity locally and rerun."
    exit 2
fi

git commit -m "Add reproducible DS 312 adversarial ML project"
repo="AfonsoAzevedo/ds312-adversarial-ml"
if git remote get-url origin >/dev/null 2>&1; then
    git push -u origin main
elif gh repo view "$repo" >/dev/null 2>&1; then
    git remote add origin "https://github.com/$repo.git"
    git push -u origin main
else
    gh repo create "$repo" --public --source=. --remote=origin --push
fi

printf '\nGITHUB REPOSITORY\n'
gh repo view "$repo" --json url,isPrivate --jq '.url + " | private=" + (.isPrivate|tostring)'
printf '\nLOCAL GIT STATUS\n'
git status --short
