#!/usr/bin/env bash
# Rows 28 to 35 in adoption order (worker R3, 2026-10-21): C69 -> C70 -> ... -> C80, every piece from saved weights / fitted files,
# every chain file checked against its stored sha256 before the next piece runs (lib/r3.sh check; exit 1 on a mismatch).
# Usage: bash run_28_35.sh [C66.csv C69.csv] [--from NN] [--to NN]
#   C66.csv / C69.csv = the outputs of rows 26 / 27 (default: the stored development files, sha checked).
#   R3_S2B_SERVE = row 26's served S2b 48k arm folder (default the stored artifacts/claude_ref/l_b6/s2b/arm48k).
#   R3_BUILDS=stored -> glue only (stored build parquets), see lib/r3.sh. R3_WITH_36=1 adds slot 36 (S7C-FOLD, only if adopted).
# Output: $OUTD/C80.csv (= CH/C80.csv; sha 3e27d66c...), parity lines in parity/r3_results.tsv.
# Runtime from saved weights (laptop CPU, 4 threads): about 8.5 h (11 state builds ~33 min each, 8 hook builds ~7 min each, 8 serves
# ~3 min each, gap3 ~10 min, Q-ACCUM ~30 min, ~14 stack / transfer steps ~3 min each). Peak disk in WORK about 3 GB.
set -euo pipefail; D=$(cd "$(dirname "$0")" && pwd -W); source "$D/lib/common.sh"; source "$D/lib/r3.sh"
FROM=28; TO=36; A=()
while [ $# -gt 0 ]; do case $1 in --from) FROM=$2; shift 2;; --to) TO=$2; shift 2;; *) A+=("$1"); shift;; esac; done
link_in() { local n=$1 f=$2; [ "$(sha "$f")" = "$(xsha "$n")" ] || r3fail "$n input $f sha differs from $(xsha "$n")"
  [ -f "$CH/$n.csv" ] && [ "$(sha "$CH/$n.csv")" = "$(xsha "$n")" ] && return 0
  ln "$f" "$CH/$n.csv" 2>/dev/null || cp "$f" "$CH/$n.csv"; }
if [ "$FROM" -le 29 ]; then
  if [ ${#A[@]} -ge 2 ]; then link_in C66 "${A[0]}"; link_in C69 "${A[1]}"
  else link_in C66 "$TFB_ROOT/$(xcol C66 3)"; link_in C69 "$TFB_ROOT/$(xcol C69 3)"; fi
fi
T0=$(date +%s)
for p in 28_fvft_mar 29_bundle19 30_fv2 31_fvinel 32_gapfv 33_qaccum 34_fvinel_w10 35_s7c_inel 36_s7c_fold; do
  n=${p%%_*}; [ "$n" -ge "$FROM" ] && [ "$n" -le "$TO" ] || continue
  if [ "$n" = 36 ] && [ "${R3_WITH_36:-0}" != 1 ]; then say "slot 36 (S7C-FOLD) skipped (R3_WITH_36 != 1)"; continue; fi
  say "piece $p start"; bash "$D/$p.sh" || r3fail "piece $p"
done
if [ "$TO" -ge 35 ]; then
  F=$CH/C80.csv; [ "${R3_WITH_36:-0}" = 1 ] && F=$CH/C81.csv
  mkdir -p "$OUTD"; cp "$F" "$OUTD/$(basename "$F")"
  say "R3_CHAIN_DONE $(basename "$F") $(sha "$F") in $(( $(date +%s) - T0 )) s -> $OUTD/$(basename "$F")"
fi
