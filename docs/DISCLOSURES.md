# Disclosures (DRAFT, 2026-10-21)

## 1. Selection-conditioned inputs (U1): please read

The organiser's ongoing-window selector keeps a window only if persistence IoU over the horizon is <= .9, and skips
origin t-1 only when its horizon matched persistence (> .9). Our ongoing decode and member inputs use what this
published rule implies, from data released at T (window_index rows of earlier origins, the pre-T gate):
- sccnn flags f_hard (always 1) and f_prevA (type A windows) as input channels (also in the ConvNeXt member and the
  cold-start sccnn rows);
- the decode cap IoU(mask, persistence) <= .92 on all ongoing windows and the band raise to >= .8 on type A windows;
- the type A test itself (window_index spacing of earlier origins plus the pre-T gate).
No value stamped after T is read. Value at stake: about +.0045 public (ADOPTED row 4, C19 +.00451, plus part of rows 6
and 13). Rows: 43 type A ongoing windows (validation 23, private 20); the cap on all 80 ongoing windows.
If the organiser rules this out, the minimal fix is: types and flags off (sccnn in "unknown" mode), no cap, no band raise.
Decision recorded: disclose plainly in the report; asking the organiser is the user's decision.
Same family, organiser-defined: onset rows predict only T+30 (the released onset condition says no queue at T+5..T+25).

## 2. Pretrained models (public, open licence; D-0048)

In the submitted files:
- ConvNeXt-Tiny ImageNet-1k, `timm/convnext_tiny.fb_in1k`, revision b43a6303c9fcf176d2d707478a128c2c91e93528,
  Apache-2.0, fine-tuned on train data only (ongoing member, a = .2). timm 1.0.15.

Screened, not in any submitted file (listed for completeness): ResNet-18 (`timm/resnet18.a1_in1k`), MobileNetV3-Large
(`timm/mobilenetv3_large_100.ra_in1k`), EfficientNet-B0 (`timm/efficientnet_b0.ra_in1k`), ConvNeXt-Small and Base
(`.fb_in1k`), ConvNeXt-Tiny in22k_ft_in1k, Big-LaMa (Places2, Apache-2.0), Chronos-Bolt (Amazon, Apache-2.0),
all Apache-2.0; revisions and SHA256 in RULES.md section 3 of the development repository.
TabPFN v2 regressor (`Prior-Labs/TabPFN-v2-reg`, revision 4972a65a1b30806315c6f92499959ffbfc69a673, SHA256
2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736, Prior Labs License 1.1): piece S0631, OFF in the
default build and in file A; if any shipped file uses it, the report must say "Built with PriorLabs-TabPFN".

## 3. External data

None in any submitted file. One screen used public Caltrans PeMS station data (districts 7 and 12, January to June 2024)
to pretrain an ongoing-queue member with the user's approval (2026-10-10); it failed its first gate and was dropped.

## 4. Organiser material in this repository

`ref/trafficflowbench-public/config/task3_lwr_modes.json` is a file of the organisers' public repository
(trafficflowbench-public), read by `src/tfb/metric.py` for the physics term of our local scorer. Its licence is the
organisers' (see their repository).

## 5. Declared methods for April rows

Following the organiser ruling of 2026-10-15 and the team rulings of 2026-10-15/17: several pieces (ADOPTED rows 27-35)
fit or choose their April rows on April's own released view or on train + March (target-month learning on released
data, Task 1 offline), and a March board verdict of a METHOD chose it for April rows. March rows use March-fit or
train-fit choices only. Each is listed with its recipe in `pipeline/pieces/`.

## 6. Development process

Author: Amin Mohamed. Built under written project rules (online rule, board-use rules, no credentials in code).
