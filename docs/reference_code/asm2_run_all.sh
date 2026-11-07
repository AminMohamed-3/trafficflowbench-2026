#!/bin/bash
# asm2: single entry for the final pair A/B, from the released data and shipped fitted inputs.
# Every queue/ODME row is regenerated (no prior submission file); Task 1 uses the frozen fitted artifacts listed in
# docs/report/ASM2_DEPENDENCIES.md (refit switches below). Run from the repo root on Linux, Python 3.12 + requirements pins.
#   bash docs/reference_code/asm2_run_all.sh OUT.csv [--retrain-cnn] [--refit-gap] [--refit-b6] [--no-gap] [--no-selcond] [--tune=CFG.json]
#        [--no-s0631 | --s0631] [--bbmember=CFG.json | --no-bbmember] [--adapt-recipe=EPOCHS,LR,CAP] [--mix-a=A]
#        [--cold-xsd | --no-cold-xsd] [--s08=L,R | --s08=shipped] [--s08-from=DIR] [--hedge A|B] [--hedge-rows=FILE]
#        [--b6=ITERS | --b6=shipped] [--b6-panel-it=PANEL:ITERS,...] [--b6-root=DIR[,DIR]] [--b6-from=DIR] [--b6-weights=FILE]
# Final pair (C_pair 2026-10-12, exp/C_pair/PAIR.md; ADOPTED row 14): --hedge A (default) = the expected-best file, --hedge B =
#   file B. --hedge-rows=FILE = the rows file of the file being built (csv submission_id, queue_pred, queue_pred_A, window_id,
#   written by c_hedge_b.py build on the finished A; private onset T+30 rows only, asserted equal to A's onset rows before the
#   change). Without it: A has no hedge rows, B takes the onset9 as-built rows (4 windows) as before. The split pair gives
#   BOTH files a rows file (A' = --hedge A --hedge-rows=..._pairsplit_A.hedge_B_rows.csv, B = --hedge B --hedge-rows=..._pairsplit_B...).
#   Validation rows are never touched, so every pair has public B = public A, PROVIDED S0631 is the same in both files (below).
# S08 stack input (--s08=L,R, default 127,4800 = C43 .89703, ADOPTED row 15; --s08=shipped = the 0398 files, 31 x 600, C36
#   form): official split-local refit at L leaves x R rounds, seeds 4701-4703, same recipe as the ship path
#   (c_s08_ship.py data/fit/export per split -> $A/s08/l<L>_r<R>/official_<split>.parquet; needs the F4 parent
#   artifacts/lane_s/0182/production273/model), read by the Task 1 chain in place of the 0398 parquets (TFB_ASM2_ART).
#   --s08-from=DIR (repo-relative) takes fitted exports official_{validation,private}.parquet from DIR, no refit.
#   Threads TFB_ASM2_S08_THREADS (default OMP_NUM_THREADS); TFB_ASM2_S08_SPLITS limits the refit to one split (staged runs).
#   ~3 h at 10 threads (data 15 + fit 25 + export 40 min per split), 2.6 GB of training matrices under $A/s08.
# B6 imputer members (--b6=ITERS, default 24000 = C47 .89776, ADOPTED row 16, KB2 2026-10-13; --b6=12000 = C44 .89744;
#   --b6=shipped = the 0399 files at 6000 iterations with asm/b6w_s08.json, C43 form): U-Net width 32, seeds 0,1,2,
#   trained per panel on the split's OWN layer.
#   Default route, no GPU: the trained weights are shipped (artifacts/claude_ref/b6l/it<ITERS>_w32/sd_*.pt with their
#   training receipts run_*.json; both the 24000 and the 12000 set) and asm2_b6l.py serve applies them to the official
#   view on CPU (-> $A/b6l/imp_*.npz, 60 files, ~6 min at 6 threads); the Task 1 chain reads them through the shipped
#   tune file of the set, artifacts/claude_ref/asm/tune_b6l24k.json or tune_b6l12k.json (TFB_ASM2_ART maps its nominal
#   arm dir b6l/official to $A/b6l), with that set's blend weights artifacts/claude_ref/asm/b6w_24k.json or b6w_12k.json
#   (refit on March I-S-OWN). The C47 file was made this way (members served on CPU); the 12000 members of the C44 probe
#   were GPU-made files, which CPU inference repeats up to float32 noise (max 4e-4 km/h, 2e-2 vph). --b6-from=DIR takes
#   fitted member files imp_{valown,privown}_<panel>_s{0,1,2}.npz from DIR instead (no inference; the C44 files give
#   the C44 rows exactly). --b6-panel-it=PANEL:ITERS,... other iterations for some panels (weights looked up as
#   it<ITERS>_w32 under the --b6-root dirs, first hit); any set other than 24000 or 12000 everywhere needs its own
#   refit blend weights --b6-weights=FILE (c_b6long_probe.py weights). --refit-b6 retrains the members on one GPU first (c_b6long_full_step.py trainf,
#   deterministic kernels, ~10 T4-hours at 12000) and serves from those. A --tune file with its own b6 block wins
#   (members and weights as written there; --refit-b6 then rebuilds its tune/b6det arms as before).
# Ongoing CNN members (C 2026-10-09, ledger 0771; probes C31-C33): base CNN (C03 recipe) and sccnn, seeds 0-4, 16 epochs;
#   per-date adaptation recipe (--adapt-recipe EPOCHS,LR,CAP, default 5,6e-4,none = mh5, adopted 2026-10-10, C35 +.0007,
#   I-Q-SPLIT both months; was 3,3e-4,none (C33); G = 3,1e-4,1500 rejected) through c_e16_adapt.py serve (production mains;
#   the recipe is recorded in every per-date receipt and must match the shipped weights, package/e16mh5);
#   each member file is mixed p = A*adapted + (1-A)*unadapted (--mix-a, default .75, C29 form) before the ensemble.
# Cold-start pools XSD (default on, --no-cold-xsd = C35 form; ADOPTED row 13, C36 +.00036, I-Q-SPLIT both months): for the
#   official ongoing windows on cold dates (same-type split-day pool <= 2 days) every member is adapted on ALL earlier dates
#   (3 ep, lr 3e-4), then a same-day pre-T hourly checkpoint stage (c_cold_split.py official: 10 members, CPU, 2 threads,
#   TFB_ASM2_DET=1, seeds = date index, poison + ft parity checks); c_cold_ship.py build puts these rows into the served
#   on/ files before the mix. ~65 min per member; TFB_ASM2_COLD_NPAR members at a time (default 6, 2 threads each).
#   Not applied with --frozen-adapted-probs (stored C33 mixed probabilities).
# Ongoing backbone member (default docs/reference_code/asm2_bbmember_c28.json: C28 ConvNeXt-Tiny, drop-in .2): blended at
#   probability level with the shipped chain and decoded by the C26 decoder (asm2_bbmember.py); config = tuning parameter.
# S 0631 (user-approved 2026-10-08; user 2026-10-10 after board C37 -.00003: hedge file B only, default off for A;
#   C 2026-10-12 after C_pair: DEFAULT OFF FOR BOTH FILES, --s0631 turns it on. Reason: it changes 45,695 speed rows in both
#   splits, so "B on, A off" made public B differ from public A by the C37 read and gave B a TabPFN/GPU dependency, for at
#   most +-.00002 of E[max]): TabPFN nongap switch-speed rows on top of the final state (asm2_s0631.py).
#   Needs one CUDA GPU (original CUDA inference; CPU drifts up to .5 km/h) and, on first use, the pinned TabPFN runtime and
#   public checkpoint (official PyPI/Hugging Face downloads, sha-verified). --no-s0631 builds the C26 file exactly.
# --tune=CFG (tune, ledgers 0479-0481): knob config for the Task 1 chain (per-split B6 imputer arms and blend weights, stage 3/4
#   bags and scales); with --refit-b6 the B6 imputers of the config (3 seeds x validation/private, deterministic kernels) are rebuilt.
# CPU build box (KB2 2026-10-13, exp/C_kbuild/KBUILD.md section 8): the default build needs no GPU. With the CPU wheel
#   torch==2.7.1+cpu (no nvidia-* packages, no triton) set TFB_ASM2_CPU_ONLY=1 for the whole run: asm2_inputs.py --environment
#   then expects that wheel, and the ongoing serving step (c_e16_adapt.py serve, serve_torch_rule) accepts the receipts of the
#   shipped per-date weights (they store the pinned 2.7.1+cu118) for CPU serving by the same release. Shipped inputs: the
#   archives of artifacts/claude_ref/asm2/package/serving_archives.json, unpacked and checked by python docs/reference_code/asm2_unpack.py.
#   Final file (C47): TFB_ASM2_CPU_ONLY=1 bash docs/reference_code/asm2_run_all.sh   (-> artifacts/claude_ref/asm2/final_A.csv)
# Prerequisites: release in data/raw/kaggle_public (Kaggle download), caches: python scripts/build_cache.py --panel all --split all
# Runtime (VM 5, 3 CPU threads): Task 1 ~45-60 min, ODME+onset ~5 min, CNN adaptation ~40 min per seed (CPU), CNN retrain
# (e16: ctl ~80 min + sccnn ~110 min per seed on one GPU), gap refit ~?, B6 refit ~30 min (GPU). e16 adaptation + unadapted
# pass ~25 min per base seed and ~60 min for the 5-seed sccnn set (CPU, 2 threads each, run in parallel below).
set -euo pipefail
SECONDS=0
OUT=""; S0631=""; BB_CFG=docs/reference_code/asm2_bbmember_c28.json; RETRAIN=0; GAP=1; REFIT_GAP=0; REFIT_B6=0; TUNE=""; HEDGE=A; CHECK=""; STATE_ATOL=1e-8; FROZEN_PROBS=0; REFIT_ADAPT=0
ADAPT_RECIPE=5,6e-4,none; MIX_A=0.75; COLD_XSD=1; S08=127,4800; S08_FROM=""; HEDGE_ROWS=""
B6=24000; B6_PANEL_IT=""; B6_ROOT=artifacts/claude_ref/b6l; B6_FROM=""; B6_WEIGHTS=""
NOT=0; NOT_ROWS=artifacts/T_noT/delivery/task2_noT.parquet; NOT_RECEIPT=artifacts/T_noT/delivery/rows_receipt.json
if [ $# -gt 0 ] && [[ $1 != --* ]]; then OUT=$1; shift; fi
while [ $# -gt 0 ]; do f=$1; shift; case $f in
  --retrain-cnn) RETRAIN=1; REFIT_ADAPT=1;; --refit-gap) REFIT_GAP=1;; --refit-b6) REFIT_B6=1;; --no-gap) GAP=0;; --no-selcond) :;;
  --refit-adaptation) REFIT_ADAPT=1;;
  --no-s0631) S0631=0;; --s0631) S0631=1;;
  --adapt-recipe=*) ADAPT_RECIPE=${f#--adapt-recipe=};; --mix-a=*) MIX_A=${f#--mix-a=};;
  --cold-xsd) COLD_XSD=1;; --no-cold-xsd) COLD_XSD=0;;
  --s08=*) S08=${f#--s08=};; --s08-from=*) S08_FROM=${f#--s08-from=}; S08_FROM=${S08_FROM%/};;
  --b6=*) B6=${f#--b6=};; --b6-panel-it=*) B6_PANEL_IT=${f#--b6-panel-it=};; --b6-root=*) B6_ROOT=${f#--b6-root=};;
  --b6-from=*) B6_FROM=${f#--b6-from=}; B6_FROM=${B6_FROM%/};; --b6-weights=*) B6_WEIGHTS=${f#--b6-weights=};;
  --no-bbmember) BB_CFG="";; --bbmember=*) BB_CFG=${f#--bbmember=};;
  --frozen-adapted-probs) FROZEN_PROBS=1;;
  --noT) NOT=1;;
  --noT-rows=*) NOT_ROWS=${f#--noT-rows=};;
  --noT-receipt=*) NOT_RECEIPT=${f#--noT-receipt=};;
  --tune=*) TUNE=${f#--tune=};;
  --hedge) HEDGE=${1:?--hedge requires A or B}; shift;; --hedge=*) HEDGE=${f#--hedge=};;
  --hedge-rows=*) HEDGE_ROWS=${f#--hedge-rows=};;
  --check) CHECK=${1:?--check requires reference spec}; shift;; --check=*) CHECK=${f#--check=};;
  --state-atol=*) STATE_ATOL=${f#--state-atol=};;
  *) echo "unknown flag $f"; exit 2;; esac; done
[[ $HEDGE == A || $HEDGE == B ]] || { echo "--hedge requires A or B"; exit 2; }
[ -n "$S0631" ] || S0631=0   # off for A and B unless --s0631 (C 2026-10-12; was: B on, A off, user 2026-10-10)
[[ -z "$HEDGE_ROWS" || -f "$HEDGE_ROWS" ]] || { echo "--hedge-rows file not found: $HEDGE_ROWS"; exit 2; }
[[ $S08 == shipped || $S08 =~ ^[0-9]+,[0-9]+$ ]] || { echo "--s08 requires LEAVES,ROUNDS or shipped"; exit 2; }
[[ $S08 != shipped || -z "$S08_FROM" ]] || { echo "--s08-from requires --s08=L,R"; exit 2; }
S08_L=${S08%,*}; S08_R=${S08#*,}
[[ $B6 == shipped || $B6 =~ ^[0-9]+$ ]] || { echo "--b6 requires ITERATIONS or shipped"; exit 2; }
[[ $B6 != shipped || -z "$B6_FROM$B6_PANEL_IT$B6_WEIGHTS" ]] || { echo "--b6-from, --b6-panel-it and --b6-weights require --b6=ITERATIONS"; exit 2; }
[[ $B6 == shipped || -n "$B6_WEIGHTS" || ( ( $B6 == 24000 || $B6 == 12000 ) && -z "$B6_PANEL_IT" ) ]] || { echo "B6 members other than 24000 or 12000 iterations everywhere need their own blend weights: --b6-weights=FILE"; exit 2; }
[[ -z "$B6_WEIGHTS" || -f "$B6_WEIGHTS" ]] || { echo "--b6-weights file not found: $B6_WEIGHTS"; exit 2; }
[[ -z "$B6_FROM" || $REFIT_B6 == 0 ]] || { echo "--b6-from and --refit-b6 exclude each other"; exit 2; }
[[ $RETRAIN == 0 || $FROZEN_PROBS == 0 ]] || { echo "--retrain-cnn requires adaptation replay"; exit 2; }
[[ $NOT == 0 || $HEDGE == A ]] || { echo "--noT requires hedge A until a no-T hedge is qualified"; exit 2; }
[[ $NOT == 0 || -z "$HEDGE_ROWS" ]] || { echo "--noT cannot take --hedge-rows"; exit 2; }
SUFFIX=""; if [ $NOT = 1 ]; then SUFFIX=_noT; fi
OUT=${OUT:-artifacts/claude_ref/asm2/final_${HEDGE}${SUFFIX}.csv}
R=docs/reference_code; A=artifacts/claude_ref/asm2/run; mkdir -p $A
# Every step runs through asm2_step.py: identical to `python SCRIPT ARGS`, but a finished step leaves with os._exit(0)
# after Python-level cleanup (pyarrow 23 / pandas 3 can abort at native teardown, exit 134, after the work is done).
PY="python $R/asm2_step.py"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-3} TFB_ASM2_DET=1
export TFB_ROOT=${TFB_ROOT:-$(pwd)}
if [ $NOT = 1 ]; then
  $PY $R/asm2_noT_rows.py --rows "$NOT_ROWS" --receipt "$NOT_RECEIPT"
fi
if [ "${TFB_ASM2_USE_ADAPTED:-1}" = 0 ]; then REFIT_ADAPT=1; fi
if [ $REFIT_ADAPT = 1 ]; then
  export TFB_ASM2_ADAPT_ROOT=${TFB_ASM2_ADAPT_ROOT:-$A/refit_adapted_models}
  export TFB_ASM2_USE_ADAPTED=0
else
  export TFB_ASM2_ADAPT_ROOT=${TFB_ASM2_ADAPT_ROOT:-artifacts/claude_ref/asm2/fitted_adapted_models}
  export TFB_ASM2_USE_ADAPTED=${TFB_ASM2_USE_ADAPTED:-1}
fi
$PY $R/asm2_inputs.py --environment --environment-out $A/environment.json
$PY scripts/build_cache.py --panel all --split all

# Task 2 onset (Q03 refit from raw, asserted equal to the reviewed model) and Task 4 ODME (NNLS .05)
$PY $R/asm2_qo.py $A/qo.parquet --threads 3
$PY scripts/q0436_official.py build --output-dir $A/q0436 --serving-only
$PY $R/asm2_onset_rows.py --windows $A/q0436/official_windows.csv --base $A/qo_onset.parquet \
  --out $A/onset.parquet --delta $A/onset_delta.csv

# Task 2 ongoing (C33): e16 base CNN seeds 0-4 + e16 sccnn seeds 0-4, per-date adaptation recipe $ADAPT_RECIPE (CPU,
# 2 threads, seed = date index), unadapted predictions of the same members, mix p = MIX_A*adapted + (1-MIX_A)*unadapted
# per member file (c_noadapt_blend.py), then ens + eiou, sccnn type A (ong2_probe build --noT), band raise.
RUN=artifacts/claude_ref/qcnn; SEEDS="0 1 2 3 4"; export E16_RECIPE=$ADAPT_RECIPE
SC_RUNS=$(for s in $SEEDS; do printf "artifacts/claude_ref/ong2/run_sc_s%s_e16," $s; done); SC_RUNS=${SC_RUNS%,}
RECEIPT_RUNS=""   # served run names (adapted weights checked per run in the build receipt)
$PY $R/ong2_official_types.py
if [ $FROZEN_PROBS = 1 ]; then   # stored mixed probabilities of the C33 serving pass (c33_e16s5_on_c31)
  BASE_PROBS=artifacts/claude_ref/asm2/e16/frozen/cnn_ens2.parquet
  SC_PROBS=artifacts/claude_ref/asm2/e16/frozen/sccnn_official.parquet
  QUEUE_MODE=fitted-probabilities
else
if [ $RETRAIN = 1 ]; then
  $PY $R/qcnn_data.py
  for s in $SEEDS; do $PY $R/c_e16_train.py ctl $s 16 --replace; $PY $R/c_e16_train.py sc $s 16 --replace; done   # GPU
fi
mkdir -p $A/on $A/off $A/mix; pids=()
CPU2=(env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2)   # 2 threads per process, as fitted
for s in $SEEDS; do
  ( "${CPU2[@]}" $PY $R/c_e16_adapt.py serve qcnn $RUN/run_full_s${s}_npt_e16 $A/on/cnn_s$s.csv
    "${CPU2[@]}" $PY $R/c_noadapt_ong.py off qcnn $RUN/run_full_s${s}_npt_e16 $A/off/cnn_s$s.csv ) > $A/ong_cnn_s$s.log 2>&1 & pids+=($!)
done
( "${CPU2[@]}" $PY $R/c_e16_adapt.py serve sccnn $SC_RUNS $A/on/sccnn_official.parquet
  "${CPU2[@]}" $PY $R/c_noadapt_ong.py off sccnn $SC_RUNS $A/off/sccnn_official.parquet ) > $A/ong_sccnn.log 2>&1 & pids+=($!)
for q in "${pids[@]}"; do wait $q || { tail -n 20 $A/ong_*.log; exit 1; }; done
ON_DIR=$A/on
if [ $COLD_XSD = 1 ]; then   # cold-start pools XSD (C36): official ongoing windows on cold dates, 10 members, CPU
  CX=$A/cold_xsd; rm -rf $CX $A/on_xsd; mkdir -p $CX/logs; cpids=(); cnames=()
  for t in "sc_s4|$SC_RUNS|4" "sc_s3|$SC_RUNS|3" "sc_s2|$SC_RUNS|2" "sc_s1|$SC_RUNS|1" "sc_s0|$SC_RUNS|0" \
           $(for s in $SEEDS; do echo "ctl_s$s|$RUN/run_full_s${s}_npt_e16|0"; done); do
    IFS='|' read -r cn cr cj <<< "$t"
    while [ $(jobs -rp | wc -l) -ge ${TFB_ASM2_COLD_NPAR:-6} ]; do sleep 15; done
    "${CPU2[@]}" $PY $R/c_cold_split.py official $cr $CX/$cn.parquet --member $cj > $CX/logs/$cn.log 2>&1 & cpids+=($!); cnames+=($cn)
  done
  for i in "${!cpids[@]}"; do
    wait ${cpids[$i]} && grep -q COLD_OFFICIAL_DONE $CX/logs/${cnames[$i]}.log || { echo "cold XSD ${cnames[$i]} failed"; tail -n 20 $CX/logs/${cnames[$i]}.log; exit 1; }
  done
  $PY $R/c_cold_ship.py build $CX $A/on $A/off $A/on_xsd | tee $CX/build.log
  grep -q COLD_XSD_BUILD_OK $CX/build.log || { echo "cold XSD build checks failed"; exit 1; }
  ON_DIR=$A/on_xsd
fi
$PY $R/c_noadapt_blend.py $ON_DIR $A/off $MIX_A $A/mix
$PY $R/qcnn_avg_probs.py $A/mix/cnn_ens2.csv $(for s in $SEEDS; do echo $A/mix/cnn_s$s.parquet; done)
  BASE_PROBS=$A/mix/cnn_ens2.parquet
  RECEIPT_RUNS=$(for s in $SEEDS; do printf "run_full_s%s_npt_e16,run_sc_s%s_e16," $s $s; done); RECEIPT_RUNS=${RECEIPT_RUNS%,}
  SC_PROBS=$A/mix/sccnn_official.parquet
  if [ "$TFB_ASM2_USE_ADAPTED" = 1 ]; then QUEUE_MODE=adapted-weights; else QUEUE_MODE=adaptation-replay; fi
fi
$PY $R/ong2_probe.py build --sc="$SC_PROBS" --noT --base-probs="$BASE_PROBS" --ongoing-out=$A/ongoing.parquet
SEL="--no-selcond"
ONGOING_ROWS=$A/ongoing.parquet
if [ -n "$BB_CFG" ]; then   # pretrained-backbone member: serve (GPU for source=weights), blend, C26 decoder (a=0 gate inside)
  $PY $R/asm2_bbmember.py serve --cfg "$BB_CFG" --out $A/bb_member.parquet
  $PY $R/asm2_bbmember.py decode --cfg "$BB_CFG" --member $A/bb_member.parquet --base "$BASE_PROBS" --sc "$SC_PROBS"     --parent-ongoing $A/ongoing.parquet --out $A/ongoing_bb.parquet
  ONGOING_ROWS=$A/ongoing_bb.parquet
fi

# S08 stack input at L x R (row 15): official refit per split with the ship-path recipe (CPU, deterministic LightGBM)
if [ $S08 != shipped ] && [ -z "$S08_FROM" ]; then
  mkdir -p $A/s08
  for sp in ${TFB_ASM2_S08_SPLITS:-validation private}; do
    for st in "data $sp" "fit $sp $S08_L $S08_R" "export $sp $S08_L $S08_R"; do
      C_S08_OUT=$A/s08 C_THREADS=${TFB_ASM2_S08_THREADS:-$OMP_NUM_THREADS} $PY $R/c_s08_ship.py $st | tee -a $A/s08/ship_$sp.log
    done
    grep -q "^wrote .*l${S08_L}_r${S08_R}.official_$sp.parquet" $A/s08/ship_$sp.log || { echo "S08 export $sp failed"; exit 1; }
  done
fi

# optional Task 1 refits (frozen files otherwise)
ART='{"_": "_"'
if [ $S08 != shipped ]; then   # S08 exports into the Task 1 chain (asm2_build Stages._in override, as the C38-C43 probes)
  S08_DIR=${S08_FROM:-$A/s08/l${S08_L}_r${S08_R}}
  for sp in validation private; do
    [ -f $S08_DIR/official_$sp.parquet ] || { echo "S08 input missing: $S08_DIR/official_$sp.parquet"; exit 1; }
    ART="$ART, \"artifacts/lane_s/0398_s08/official_$sp.parquet\": \"$S08_DIR/official_$sp.parquet\""
  done
fi
# B6 members (row 16): official-view member files from the shipped OWN-trained weights (CPU), --b6-from, or --refit-b6 (GPU)
B6_ACTIVE=0; TUNE_EFF=$TUNE
if [ $B6 != shipped ]; then
  mkdir -p $A/b6l; B6_SET=12k; if [ $B6 = 24000 ] && [ -z "$B6_PANEL_IT" ]; then B6_SET=24k; fi   # shipped sets: 24k (C47), 12k (C44)
  B6_TUNE=artifacts/claude_ref/asm/tune_b6l$B6_SET.json; B6_WNOM=artifacts/claude_ref/asm/b6w_$B6_SET.json   # other sets: the 12k block + --b6-weights
  if [ -n "$TUNE" ]; then   # user knobs + the shipped b6 block; a user b6 block wins (nothing written)
    rm -f $A/b6l/tune.json; $PY $R/asm2_b6l.py tune --shipped $B6_TUNE --base "$TUNE" --out $A/b6l/tune.json
    if [ -f $A/b6l/tune.json ]; then TUNE_EFF=$A/b6l/tune.json; B6_ACTIVE=1; else echo "--tune carries its own b6 block: used as it is, --b6 flags ignored"; fi
  else TUNE_EFF=$B6_TUNE; B6_ACTIVE=1; fi
fi
if [ $B6_ACTIVE = 1 ]; then
  B6_SRC=${B6_FROM:-$A/b6l}
  if [ -z "$B6_FROM" ]; then
    B6_ARGS=(--it "$B6" --roots "$B6_ROOT" --out $A/b6l); if [ -n "$B6_PANEL_IT" ]; then B6_ARGS+=(--panel-it "$B6_PANEL_IT"); fi
    if [ $REFIT_B6 = 1 ]; then $PY $R/asm2_b6l.py serve "${B6_ARGS[@]}" --refit | tee $A/b6l/serve.log   # one CUDA GPU, deterministic kernels
      grep -q "^B6L_SERVE_OK 60 " $A/b6l/serve.log || { echo "B6 member refit failed"; exit 1; }
    else CUDA_VISIBLE_DEVICES=-1 $PY $R/asm2_b6l.py serve "${B6_ARGS[@]}" | tee $A/b6l/serve.log
      grep -q "^B6L_SERVE_OK 60 .* device cpu" $A/b6l/serve.log || { echo "B6 member step failed"; exit 1; }
    fi
  fi
  for pn in D12_I405_N D12_I405_S D12_I5_N D12_I5_S D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S; do for s in 0 1 2; do for m in valown privown; do
    [ -f $B6_SRC/imp_${m}_${pn}_s${s}.npz ] || { echo "B6 member missing: $B6_SRC/imp_${m}_${pn}_s${s}.npz"; exit 1; }
    ART="$ART, \"artifacts/claude_ref/b6l/official/imp_${m}_${pn}_s${s}.npz\": \"$B6_SRC/imp_${m}_${pn}_s${s}.npz\""; done; done; done
  if [ -n "$B6_WEIGHTS" ]; then ART="$ART, \"$B6_WNOM\": \"$B6_WEIGHTS\""; fi
fi
if [ $REFIT_GAP = 1 ]; then $PY $R/asm2_gaprefit.py
  ART="$ART, \"artifacts/claude_ref/gap/offpred_v3.npz\": \"artifacts/claude_ref/asm2/gaprefit/offpred_v3.npz\""; fi
if [ $B6_ACTIVE = 0 ] && [ $REFIT_B6 = 1 ] && [ -n "$TUNE" ]; then   # tune config arms: tune/b6det imp_{val,priv}_<pn>_s{0,1,2}, rebuilt into asm2/run/b6det
  for s in 0 1 2; do for m in val priv; do TFB_ASM2_DET=1 TFB_ASM2_B6OUT=$A/b6det $PY $R/asm2_b6imp.py $m all $s; done; done
  for pn in D12_I405_N D12_I405_S D12_I5_N D12_I5_S D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S; do for s in 0 1 2; do for m in val priv; do
    ART="$ART, \"artifacts/claude_ref/tune/b6det/imp_${m}_${pn}_s${s}.npz\": \"$A/b6det/imp_${m}_${pn}_s${s}.npz\""; done; done; done
elif [ $B6_ACTIVE = 0 ] && [ $REFIT_B6 = 1 ]; then TFB_ASM2_DET=1 $PY $R/asm2_b6imp.py priv all 0   # deterministic kernels: bitwise repeatable on one GPU
  for pn in D12_I405_N D12_I405_S D12_I5_N D12_I5_S D7_I10_E D7_I10_W D7_I210_E D7_I210_W D7_I405_N D7_I405_S; do
    ART="$ART, \"artifacts/claude_ref/b6full/imp_priv_${pn}_s0.npz\": \"artifacts/claude_ref/asm2/b6refit/imp_priv_${pn}_s0.npz\""; done; fi
export TFB_ASM2_ART="$ART}"
FINAL_ARGS=(--hedge "$HEDGE" --state-atol "$STATE_ATOL")
if [ -n "$HEDGE_ROWS" ]; then if [ $HEDGE == B ]; then FINAL_ARGS+=(--hedge-rows "$HEDGE_ROWS"); else FINAL_ARGS+=(--hedge-rows-A "$HEDGE_ROWS"); fi; fi
if [ $GAP = 0 ]; then FINAL_ARGS+=(--no-gap); fi
if [ -n "$TUNE_EFF" ]; then FINAL_ARGS+=(--tune "$TUNE_EFF"); fi
if [ -n "$CHECK" ]; then FINAL_ARGS+=(--check "$CHECK"); fi
if [ $NOT = 1 ]; then FINAL_ARGS+=(--noT --noT-rows "$NOT_ROWS" --noT-receipt "$NOT_RECEIPT"); fi

# Task 1 chain + S lane + phys5, then assembly of all rows
T1_ARGS=()
if [ $S0631 = 1 ]; then
  # S 0631: state chain on CPU (writes the query rows), TabPFN heads on the GPU, then assembly from the overlaid state
  CUDA_VISIBLE_DEVICES="" $PY $R/asm2_final.py --out $A/t1.csv --queue-base none --state-only --s0631-query $A/s0631_query.npz "${FINAL_ARGS[@]}"
  $PY $R/asm2_s0631.py predict --query $A/s0631_query.npz --out $A/s0631_rows.npz --parity $A/s0631_parity.json
  T1_ARGS=(--t1-from $A/t1.csv.state.parquet --state-rows $A/s0631_rows.npz)
fi
CUDA_VISIBLE_DEVICES="" $PY $R/asm2_final.py --out "$OUT" --queue-base none --onset-rows $A/onset.parquet --onset-delta "" \
  --odme-rows $A/qo_odme.parquet --ongoing-rows $ONGOING_ROWS $SEL "${FINAL_ARGS[@]}" "${T1_ARGS[@]}"
if [ $NOT = 1 ]; then QUEUE_MODE=strict-noT-rows; fi
$PY $R/asm2_build_receipt.py "$OUT" --seconds "$SECONDS" --adapt-root "$TFB_ASM2_ADAPT_ROOT" \
  --queue-mode "$QUEUE_MODE" --base-probs "$BASE_PROBS" --selection-probs "$SC_PROBS" --runs "$RECEIPT_RUNS"
echo "done: $OUT (manifest $OUT.manifest.json)"
