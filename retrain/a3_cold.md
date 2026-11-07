# Row 22: L_dose A3 cold-date members (speed term inside both XSD stages), rerun and re-serve recipe

Default build path (pipeline/pieces/22_a3cold.sh): the 10 A3 member files (`weights/rows/22.csv`,
`22_a3cold/members/{ctl,sc}_s{0..4}.parquet` + `.json` receipts; sha list of the development repo
`artifacts/claude_ref/l_dose/a3/members_sha256.txt`). They hold, per cold official ongoing window (56: March 31, April 25),
p_un (unadapted), p_serve (copied from the shipped cold file, info), p_xall (stage 1) and p_xsd (served value), each
computed by exp/L_dose/a3_cold.py official from the RELOADED saved per-date models. The shipped chain then runs unchanged
(c_cold_ship build -> .75 blend -> seed mean -> ong2_probe --noT -> Tiny member decode) and gives C57 exactly (laptop
2026-10-21: every chain file bytewise equal to the stored one).

## Saved weights
Development repo `artifacts/claude_ref/l_dose/a3/weights/<member>/<split>/<date>/{xall.pt, xsd_H###.pt}`: 530 state dicts
of the qcnn Net (cin = CTX*5+16), 1.9 GiB, sha list `a3/weights_sha256.txt`. Not listed in weights/rows/22.csv (the
default path reads the served probabilities); publishing them is the user's call.

## Re-serve from the saved weights (CPU, minutes)
Per member and split: qpa.load_split(split), base net = the member's run model.pt (ctl: artifacts/claude_ref/qcnn/
run_full_s{s}_npt_e16, sc: artifacts/claude_ref/ong2/run_sc_s{j}_e16), for each cold official date and window at t load
`xsd_H<12 floor(t/12)>.pt`, x = qpa.official_x(...) (sccnn: S.fill with the window type flags), p = sigmoid(net(x)) on CPU,
TFB_ASM2_DET=1, 2 threads (the serving loop of a3_cold.py official, lines "for p, v in data.items(): for w, t, x in
qpa.official_x"). Bitwise on the VM CPU that made the files (the VM reproduced VM2's shipped files bitwise); on another CPU
max |dp| about 1.25e-6 (A3_RESULT.md, laptop vs VM), so decoded cells can differ in rare ties: a sibling. No script for
this in final_solution yet (open item).

## Full refit (CPU only, deterministic)
- Code: exp/L_dose/a3_cold.py official (A2 speed head exp/L_dose/a2_speed.py, heads.pt sha b842413a...), driver
  exp/L_dose/a3_vm.sh (20 tasks = 10 members x 2 splits, one process each, `--wdir` saves the models), merge
  (`a3_cold.py merge`), finisher exp/L_dose/a3_finish.sh (sha lists, member receipt checks, chain, delta).
- Recipe (fixed by the advisor before any number, docs/research/advisor_L4/ADVICE.md section 5): stage 1 xall =
  c_cold_split.ft on all earlier dates of the split, 3 epochs, lr 3e-4, seed k, + .3 x speed NLL on 32 origins per step
  from every 15-minute origin of the earlier days; stage 2 per hour checkpoint H = 12 floor(t/12): ft on same-day windows
  t' + 6 <= H (3 ep, 3e-4, seed 1000 k + H) + .3 x speed NLL on same-day origins with t + 6 <= H; speed batches on their
  own RNGs (seed + 50000, flags seed + 60000). Cold dates = same-type split-day pool <= 2.
- Checks inside: weight 0 == c_cold_split.ft bitwise (A3_FTCHECK), future poison per split (later dates and day d after
  H poisoned -> every fit identical), ft parity vs qcnn_predict_adapt.finetune.
- Hardware and time (2026-10-15/16): Vast VM CPU, 12 threads = 6 processes x 2, torch 2.7.1+cu128 wheel on CPU, 249 min
  for the official members (about 100 to 120 core-hours incl. TM-R exports). No GPU.
- Expected result: bitwise on the same CPU type; elsewhere a sibling (~1e-6 forwards).
- Score (ADOPTED.md row 22): board C57 .89907 (+.00057 vs C55); private expected +.0001 (-.0006 to +.0008).
