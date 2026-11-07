"""C-tune 2026-10-08: tunable variant of c_bb_arch.BBNet for the backbone tuning program (exp/C_backbone/TUNING_PLAN.md).
c_bb_arch.py is unchanged; make(spec) here reproduces c_bb_arch.make(spec) exactly when the spec has no tuning keys
(same module tree, same parameter creation order, so the same seed gives the same init).
Extra spec keys (all optional, default = c_bb_arch recipe):
  adapter : 'lin3'   1x1 conv 21->3 + BN (default)
            'mlp3'   1x1 21->W, BN, GELU, 1x1 W->3, BN (W = adapter_w, default 32)
            'conv3'  3x3 conv 21->3 + BN
            'inchans' no adapter: BN on the 21 channels, the backbone stem takes 21 input channels (timm in_chans=21:
                      pretrained stem weights repeated over channels and rescaled, timm adapt_input_conv)
  D, hp   : FPN width (96) and pooled time rows (6)
  nblocks : residual 1D blocks after the join (3; dilations 1,2,4,8,16,...)
  dropout : dropout on the joined per-link features and before the 1x1 head (0)
  stride1 : True -> stride-1 stem (as c_bb_arch 's1') combined with an upK input (adapt 'upK' only)
Weights come only from the pinned manifest (c_bb_arch.weights_file)."""
import sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "docs/reference_code"))
import c_bb_arch
from c_bb_arch import BBNet, Block, CIN, NIMG, to_img, weights_file, _stride1

TUNE_KEYS = ("adapter", "adapter_w", "D", "hp", "nblocks", "dropout", "stride1")


class TuneNet(BBNet):
    def __init__(self, bb, adapt="up2", pretrained=True, adapter="lin3", adapter_w=32, D=96, hp=6, nblocks=3, dropout=0.,
                 stride1=False):
        nn.Module.__init__(self)
        import timm
        assert adapt != "mid", "mid: use c_bb_arch"
        self.adapt, self.hp, self.bbname = adapt, hp, bb
        kw = dict(pretrained=pretrained)
        if pretrained:
            kw["pretrained_cfg_overlay"] = dict(file=weights_file(bb))
        if adapter == "lin3":
            self.adapter = nn.Sequential(nn.Conv2d(NIMG, 3, 1), nn.BatchNorm2d(3))
        elif adapter == "mlp3":
            self.adapter = nn.Sequential(nn.Conv2d(NIMG, adapter_w, 1), nn.BatchNorm2d(adapter_w), nn.GELU(),
                                         nn.Conv2d(adapter_w, 3, 1), nn.BatchNorm2d(3))
        elif adapter == "conv3":
            self.adapter = nn.Sequential(nn.Conv2d(NIMG, 3, 3, padding=1), nn.BatchNorm2d(3))
        elif adapter == "inchans":
            self.adapter = nn.BatchNorm2d(NIMG); kw["in_chans"] = NIMG
        else:
            raise ValueError(adapter)
        self.enc = timm.create_model(bb, features_only=True, **kw)
        if adapt in ("s1", "s2"):
            _stride1(self.enc, bb, adapt == "s2")
        else:
            assert adapt.startswith("up"); self.up = int(adapt[2:])
            if stride1:
                _stride1(self.enc, bb, False)
        chs = self.enc.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, D, 1) for c in chs])
        self.smooth = nn.Sequential(nn.Conv2d(D, D, 3, padding=1), nn.BatchNorm2d(D), nn.GELU())
        self.skip = nn.Conv1d(CIN, 64, 1)
        self.join = nn.Conv1d(D * hp + 64, 128, 1)
        self.blocks = nn.Sequential(*[Block(128, 2 ** i) for i in range(nblocks)])
        self.out = nn.Conv1d(128, 6, 1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def stage_groups(self):
        """backbone parameters by depth: 0 = stem (stem_*, conv1, bn1, conv_stem), k = k-th stage (stages_i / blocks_i:
        i + 1; layer_k / layerk: k). Returns a list of lists, index = depth."""
        import re
        by = {}
        for n, p in self.enc.named_parameters():
            m = re.match(r"(stages|blocks|layer)_?(\d+)", n)
            d = 0 if m is None else int(m.group(2)) + (0 if m.group(1) == "layer" else 1)
            by.setdefault(d, []).append(p)
        return [by.get(d, []) for d in range(max(by) + 1)]

    def forward(self, x):
        L = x.shape[-1]
        feats = self.encode(to_img(x))
        p = self.lat[-1](feats[-1])
        for f, lat in zip(reversed(feats[:-1]), list(self.lat)[-2::-1]):
            p = F.interpolate(p, size=f.shape[-2:], mode="bilinear", align_corners=False) + lat(f)
        p = self.smooth(p)
        if p.shape[-1] < L:
            p = F.interpolate(p, size=(p.shape[-2], L), mode="bilinear", align_corners=False)
        p = F.adaptive_avg_pool2d(p, (self.hp, L))
        z = self.drop(torch.cat([p.flatten(1, 2), self.skip(x)], 1))
        return self.out(self.drop(F.gelu(self.blocks(self.join(z)))))


def make(spec):
    if spec.get("bb", "cnn") == "cnn" or not any(k in spec for k in TUNE_KEYS) and spec.get("adapt") == "mid":
        return c_bb_arch.make(spec)
    kw = {k: spec[k] for k in TUNE_KEYS if k in spec}
    return TuneNet(spec["bb"], spec.get("adapt", "up2"), bool(spec.get("pretrained", True)), **kw)


nparams = c_bb_arch.nparams
