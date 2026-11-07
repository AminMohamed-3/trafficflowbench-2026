"""C-bb 2026-10-08: ImageNet-pretrained 2D CNN backbones (timm) for the ongoing queue forecaster (Goal 3 phase 1).
Same inputs and outputs as qcnn_train.Net / ong2 sccnn: x (B, 136, L) = 24 context steps x 5 channels (time folded
into channels, links on the last axis) + 16 per-link channels (7 statics, 6 prior slots: 127..129 hold the sccnn
selection flags, tod, ongoing flag); output 6 x L logits (T+5..T+30).
Image: the 24 x 5 context becomes a 5-channel (time x links) map, the 16 per-link channels are broadcast over time:
(B, 21, 24, L). Adaptations (adapt):
  upK : 1x1 adapter 21->3 (+BN), bilinear upscale xK, standard backbone (timm features_only)
  s1  : 1x1 adapter 21->3 (+BN), stride-1 stem and no max-pool (ResNet) / stride-1 conv_stem (EffNet, MobileNetV3) /
        stride-1 'same' patchify stem (ConvNeXt), native resolution
  s2  : as s1 but total stem stride 2 (ConvNeXt: 4x4 patchify stride 2 pad 1; ResNet: stride-1 conv1, keep max-pool)
  mid : 'start from the middle' (user idea): drop stem + stage 1, learned conv adapter 21->C (two 3x3 conv+BN+ReLU)
        into stage 2 (ResNet only: layer2..layer4)
Decoder (FPN): 1x1 laterals to D, top-down bilinear upsampling + sum to the finest level, 3x3 smooth, adaptive average
pool to (hp time rows, L links), flattened to D*hp channels per link, joined with a 1x1 skip of the raw 1D input
(U-Net style), 3 residual dilated 1D blocks (qcnn_train.Block) and a 1x1 head to 6.
Weights: only the pinned files in artifacts/claude_ref/c_bb/weights_manifest.json (c_bb_weights.py, VM download).
make(spec) -> module; spec = dict(bb=<timm name|cnn>, adapt=..., pretrained=bool)."""
import json, sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "docs/reference_code"))
from qcnn_train import Net, Block, CTX

MANIFEST = ROOT / "artifacts/claude_ref/c_bb/weights_manifest.json"
CIN = CTX * 5 + 16
NIMG = 5 + CIN - CTX * 5        # 21 image channels


def weights_file(name):
    m = json.loads(MANIFEST.read_text())["models"][name]
    return m["file"]


def to_img(x):
    B, C, L = x.shape
    t = x[:, :CTX * 5].reshape(B, CTX, 5, L).permute(0, 2, 1, 3)
    s = x[:, CTX * 5:, None, :].expand(B, C - CTX * 5, CTX, L)
    return torch.cat([t, s], 1)


def _stride1(enc, name, s2=False):
    if s2:          # total stem stride 2 (stride-1 patchify does not fit 12 GB for ConvNeXt at 24 x ~190)
        if name.startswith("convnext"):
            st = enc.stem_0 if hasattr(enc, "stem_0") else enc.stem[0]
            st.stride = (2, 2); st.padding = (1, 1)
        elif name.startswith("resnet"):
            enc.conv1.stride = (1, 1)
        else:
            raise ValueError(name)
        return
    if name.startswith("resnet"):
        enc.conv1.stride = (1, 1); enc.maxpool = nn.Identity()
    elif name.startswith(("efficientnet", "mobilenetv3")):
        enc.conv_stem.stride = (1, 1)
    elif name.startswith("convnext"):
        st = enc.stem_0 if hasattr(enc, "stem_0") else enc.stem[0]
        st.stride = (1, 1); st.padding = "same"
    else:
        raise ValueError(name)


class BBNet(nn.Module):
    def __init__(self, bb="resnet18.a1_in1k", adapt="s1", pretrained=True, D=96, hp=6):
        super().__init__()
        import timm
        self.adapt, self.hp = adapt, hp
        kw = dict(pretrained=pretrained)
        if pretrained:
            kw["pretrained_cfg_overlay"] = dict(file=weights_file(bb))
        if adapt == "mid":
            assert bb.startswith("resnet"), "mid adaptation implemented for ResNet only"
            m = timm.create_model(bb, num_classes=0, **kw)
            c2 = m.layer2[0].conv1.in_channels
            self.adapter = nn.Sequential(nn.Conv2d(NIMG, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
                                         nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU())
            self.enc = nn.ModuleList([m.layer2, m.layer3, m.layer4])
            chs = [c2] + [fi["num_chs"] for fi in m.feature_info[2:]]      # layer2..layer4
        else:
            self.adapter = nn.Sequential(nn.Conv2d(NIMG, 3, 1), nn.BatchNorm2d(3))
            self.enc = timm.create_model(bb, features_only=True, **kw)
            if adapt in ("s1", "s2"):
                _stride1(self.enc, bb, adapt == "s2")
            else:
                assert adapt.startswith("up"); self.up = int(adapt[2:])
            chs = self.enc.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, D, 1) for c in chs])
        self.smooth = nn.Sequential(nn.Conv2d(D, D, 3, padding=1), nn.BatchNorm2d(D), nn.GELU())
        self.skip = nn.Conv1d(CIN, 64, 1)
        self.join = nn.Conv1d(D * hp + 64, 128, 1)
        self.blocks = nn.Sequential(Block(128, 1), Block(128, 2), Block(128, 4))
        self.out = nn.Conv1d(128, 6, 1)

    def backbone_params(self):
        return list(self.enc.parameters())

    def encode(self, img):
        a = self.adapter(img)
        if self.adapt == "mid":
            feats = [a]
            for st in self.enc:
                feats.append(st(feats[-1]))
            return feats
        if self.adapt.startswith("up"):
            a = F.interpolate(a, scale_factor=self.up, mode="bilinear", align_corners=False)
        return self.enc(a)

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
        z = torch.cat([p.flatten(1, 2), self.skip(x)], 1)
        return self.out(F.gelu(self.blocks(self.join(z))))


def make(spec):
    if spec.get("bb", "cnn") == "cnn":
        return Net(cin=CIN)
    return BBNet(spec["bb"], spec.get("adapt", "s1"), bool(spec.get("pretrained", True)))


def nparams(m):
    return int(sum(p.numel() for p in m.parameters()))
