# Row 26: S2b B6 imputer members (48k iterations), stored outputs and serve recipe

Default build path (pipeline/pieces/26_bundle18.sh step 1): the 60 stored member arrays `26_bundle18/arm48k/imp_*.npz`
plus their receipt (61 files, sha256-pinned in `weights/rows/26.csv`; old path artifacts/claude_ref/l_b6/s2b/arm48k). They are
copied into `$WORK/26_bundle18/arm48k` and the rest of the piece (hooked S2b state build, gap post, S7b stage 3, gap2 step)
runs from code.

Verified 2026-11-07 on the laptop (Git Bash, Windows venv): piece 26 alone on the stored C65 (c_gapw_on_c64, sha 7b2a2daa)
with the stored arrays wrote C66 with sha256 f1a5278f5f17335445ccd4fdc6852ecb760ff41e3dd5282040094846b05e5108, equal to the
submitted development file artifacts/claude_ref/l_b6/bundle18/c_bundle18_s2b_s7b.csv (SHA_EQUAL, 74 min; hook, POST_CHECK2 and
check b EXACT on all 128,465 gap rows).

## Why stored outputs
The arrays were served once during development. Re-serving the same 60 saved members (`sd_*.pt`, `run_*.json`, it48000_w32,
also in weights/rows/26.csv) on the laptop CPU gives arrays that differ in the last float bits, and the hooked build turns that
into state differences of up to 2.33 km/h on a few cells (chain run 2026-10-21: row 26 DIFF, then row 29 failed on a sha
check). The difference does not depend on thread count or CPU instruction set; it comes from the serve path itself. So the
default path ships the arrays as a stored-prediction exception, like rows 21 to 23 and 33.

## Serve from the saved members (a sibling, not C66)
`R26_SERVE=1 bash pipeline/pieces/26_bundle18.sh PARENT OUT` runs `pipeline/pieces/src/26/s2b_serve.py $WORK/26_bundle18/arm48k
--root <weights 26_bundle18/s2b>`: 60 members on CPU (OMP/MKL 3 threads), about 5 min, peak about 3 GB. The result is a sibling
of C66 within noise (board effect far below the .00001 state read noise); later pieces that check parent sha256 need
SKIP_PARENT_SHA=1.

## Retrain
The members come from exp/L_b6 (s2b_b6r.py: B6R masked U-Net, ramp and occupancy channels, 48k iterations, width 32, three
seeds per corridor and month, trained only on the visible cells of the scored month). Recipe and receipts: exp/L_b6/S2B_RESULT.md
and BUNDLE18.md in the development repository; the trained state dicts are the saved weights above.
