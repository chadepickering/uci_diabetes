#!/usr/bin/env bash
# render_combined.sh
# Renders the combined Part 1 + Part 2 Quarto book to chadepickering.github.io/uci-diabetes/
# Run from anywhere: bash scripts/render_combined.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMBINED="$REPO_ROOT/reports/combined"
PART1="$REPO_ROOT/reports/part1"
PART2="$REPO_ROOT/reports/part2"
OUT="$REPO_ROOT/../chadepickering.github.io/uci-diabetes"
FREEZE="$PART1/.quarto/_freeze"

echo "==> Disabling individual book configs..."
mv "$PART1/_quarto.yml" "$PART1/_quarto.yml.bak"
mv "$PART2/_quarto.yml" "$PART2/_quarto.yml.bak"

# Restore on exit (success or failure)
restore() {
  echo "==> Restoring individual book configs..."
  [ -f "$PART1/_quarto.yml.bak" ] && mv "$PART1/_quarto.yml.bak" "$PART1/_quarto.yml"
  [ -f "$PART2/_quarto.yml.bak" ] && mv "$PART2/_quarto.yml.bak" "$PART2/_quarto.yml"
}
trap restore EXIT

echo "==> Rendering combined book..."
cd "$COMBINED"
quarto render

echo "==> Copying Part 1 figures from freeze cache..."
for chapter in 01_eda 02_causal_inference 03_experimental_design; do
  SRC="$FREEZE/$chapter/figure-html"
  DEST="$OUT/part1/${chapter}_files/figure-html"
  if [ -d "$SRC" ]; then
    mkdir -p "$DEST"
    cp "$SRC/"*.png "$DEST/" 2>/dev/null && echo "    Copied $chapter figures" || echo "    No figures for $chapter"
  fi
done

echo "==> Done. Output at: $OUT"
