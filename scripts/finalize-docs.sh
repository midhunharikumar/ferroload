#!/usr/bin/env bash
#
# Finalize the docs move: relocate the canonical markdown files into docs/,
# replacing the temporary include-markdown stubs created during setup.
#
# Why this is a separate step: the docs site was scaffolded in an environment
# without a working shell, so the original files could not be deleted there. The
# site already renders these files via the include-markdown plugin; running this
# on a machine with git + a shell turns the staged includes into a real move.
#
# DESIGN.md is intentionally NOT moved — it stays at the repo root and out of the
# docs site. CHANGELOG.md is also kept at the root (release tooling expects it
# there); its docs page renders from the root file via include permanently.
#
# Usage:  bash scripts/finalize-docs.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# source (repo-relative)         ->  destination in docs/
moves=(
  "PYTHON_API.md|docs/python/api.md"
  "python/README_HF.md|docs/python/quickstart.md"
  "USAGE.md|docs/rust/usage.md"
  "EXAMPLES.md|docs/rust/examples.md"
  "BENCHMARKS.md|docs/benchmarks.md"
)

use_git=0
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  use_git=1
fi

for pair in "${moves[@]}"; do
  src="${pair%%|*}"
  dst="${pair##*|}"
  if [[ ! -f "$src" ]]; then
    echo "skip: $src not found (already moved?)"
    continue
  fi
  rm -f "$dst"                       # drop the include stub
  mkdir -p "$(dirname "$dst")"
  if [[ "$use_git" == "1" ]]; then
    git rm -q --cached "$dst" 2>/dev/null || true
    git mv "$src" "$dst"
  else
    mv "$src" "$dst"
  fi
  echo "moved: $src -> $dst"
done

cat <<'EOF'

Done. These docs pages now contain the real content (no include stubs).
Note: docs/changelog.md still includes the root CHANGELOG.md by design, so keep
`mkdocs-include-markdown-plugin` installed.
Follow-ups (optional):
  - Trim the now-moved root-file paths from .github/workflows/docs.yml `on.push.paths`
    (leave CHANGELOG.md, which the changelog page still includes).
  - Re-check intra-page links with: mkdocs build --strict
EOF
