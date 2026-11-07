# Row 21: ConvNeXt-Tiny ongoing member (VM retrain with saved weights), serve and retrain recipe

Default build path (pipeline/pieces/21_tinyvm.sh): the stored member probabilities `21_tinyvm/bb_member_vm.parquet`
(sha c02523d2, 87,000 rows, 80 windows) = the VM serve of the saved `model.pt` (sha 43da1ffc). Both are in
`weights/rows/21.csv`. With them the shipped decode (asm2_bbmember.py decode, a = .2) gives C55 exactly (laptop
2026-10-21: the new ongoing rows bytewise equal to the stored ones, a13fff47).

## Serve from model.pt
`21_tinyvm.sh PARENT OUT --serve-weights cpu|cuda` writes a config with source = weights; asm2_bbmember.py serve builds
c_bb_arch (convnext_tiny.fb_in1k, adapt up2, pretrained=False) and loads model.pt; inputs qcnn_predict_adapt.load_split
(official_x, the released masked cubes) + selection flags (c_bb_serve.predict), no per-date adaptation.
- Needs timm 1.0.15 + torchvision 0.22.1 (+ huggingface_hub 0.30.2, safetensors 0.5.3) in the build python; the laptop
  .venv_tfb has no timm, so this was not run here.
- CPU: 80 forward passes of a 28.4 M parameter net (one per official window) plus data loading: a few minutes, under
  2 GiB. fp32 (amp only on CUDA) -> probabilities differ from the bf16 GPU serve at about 1e-3: a SIBLING; some of the 65
  decoded cells may differ. Pieces 22 and 23 require the stored member (sha check) and stop on a sibling.
- CUDA: bf16 autocast as the original serve; bitwise only on the VM card (NVIDIA GeForce RTX 5070, torch 2.7.1+cu128).

## Retrain (GPU)
exp/C_backbone/vm_tiny/run.sh (+ vm_tiny.py, which only redirects output paths): steps env -> data -> weights -> train ->
servedigest -> serve. Training = docs/reference_code/c_bb_tune.py --bb convnext_tiny.fb_in1k --adapt up2 --patience 99
--epochs 8 --seed 0 --serving --fit-end 2031-02-28 --accum 1 (bf16 autocast), ImageNet weights convnext_tiny.fb_in1k at the
pinned revision (sha256 08b9dc9c...), qcnn cubes of qcnn_data.py (windows.parquet sha a6efcbc9...). VM RTX 5070 12 GB:
about 35 min training (4.33 min/epoch) + 8 min serve, VRAM about 5.2 GB. Result: a new draw of the member (C55 was +.00034
public vs C54, "mostly March luck"; private expected 0, -.0003 to +.0003, ADOPTED.md row 21).
