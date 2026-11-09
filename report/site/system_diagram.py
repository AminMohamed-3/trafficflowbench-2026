"""System diagram of the final solution (one source for the site and the Kaggle write-up).

  python report/site/system_diagram.py          -> report/site/system_diagram.svg (inlined by build_site.py)
                                                 + report/figures/out/system_diagram.html (light-theme page for the PNG)
Colours are CSS classes (sd-*) defined in the site's stylesheet and in the PNG page, so the figure follows light and dark.
"""
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H = 1000, 712
CHIP = {"train": ("train", "sd-c-train"), "month": ("month", "sd-c-month"), "T": ("≤ T", "sd-c-online"),
        "M4A": ("Mar→Apr", "sd-c-m4a")}
out = []


def t(x, y, s, cls="sd-s", anchor="start", extra=""):
    out.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{extra}>{escape(s)}</text>')


def rect(x, y, w, h, cls, rx=6, dash=False):
    d = ' stroke-dasharray="5 4"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"{d}/>')


def arrow(pts, cls="sd-ln", label=None, lx=None, ly=None, anchor="middle"):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    out.append(f'<path d="{d}" class="{cls}" fill="none" marker-end="url(#sd-arr)"/>')
    if label:
        t(lx, ly, label, "sd-a", anchor)


def chips(right, y, keys):
    x = right
    for k in reversed(keys):
        lab, cls = CHIP[k]
        w = 12 + 6.4 * len(lab)
        x -= w
        out.append(f'<rect x="{x:.0f}" y="{y}" width="{w:.0f}" height="17" rx="3" class="{cls}"/>')
        t(round(x + w / 2), y + 12.5, lab, f"sd-chip {cls}-t", "middle")
        x -= 6


def box_lines(x, y, w, h, title, lines, cls="sd-mini"):
    rect(x, y, w, h, cls, rx=4)
    t(x + 9, y + 18, title, "sd-m")
    for i, s in enumerate(lines):
        t(x + 9, y + 34 + 14 * i, s, "sd-s")


# ---- column heads
t(16, 26, "INPUTS", "sd-h"); t(240, 26, "ENGINES · BUILT ONCE PER SCORED MONTH", "sd-h"); t(800, 26, "OUTPUT", "sd-h")

# ---- inputs
INP = [("Train months", ["June 2030 to Feb 2031", "every cell visible"]),
       ("Scored month", ["released cells only:", "March public, April private"]),
       ("Published constants", ["FD parameters, network,", "OD prior and path matrix"])]
for i, (ti, ls) in enumerate(INP):
    y = 40 + 100 * i
    rect(16, y, 172, 86, "sd-box")
    t(28, y + 24, ti, "sd-t")
    for j, s in enumerate(ls):
        t(28, y + 46 + 16 * j, s, "sd-s")
out.append('<path d="M196,60 L204,60 L204,306 L196,306" class="sd-ln" fill="none"/>')
arrow([(204, 183), (236, 183)])
out.append('<text x="216" y="183" class="sd-a" text-anchor="middle" transform="rotate(-90 216 183)" dy="-4">one month at a time</text>')

# ---- engines
X0, X1 = 240, 764
MX = [252, 428, 604]; MW = 148


def engine(y, h, title, keys, minis, links):
    rect(X0, y, X1 - X0, h, "sd-box")
    t(X0 + 12, y + 21, title, "sd-t")
    chips(X1 - 10, y + 8, keys)
    my, mh = y + 34, h - 44
    for x, (ti, ls) in zip(MX, minis):
        box_lines(x, my, MW, mh, ti, ls)
    for k, lab in enumerate(links):
        a, b = MX[k] + MW, MX[k + 1]
        arrow([(a + 2, my + mh / 2), (b - 2, my + mh / 2)], label=lab, lx=(a + b) / 2, ly=my + mh / 2 - 6)


engine(40, 132, "Task 1 · state · 35%", ["train", "month", "M4A"],
       [("LightGBM stack", ["pooled + split-local,", "physical stages,", "FD projection"]),
        ("B6 masked U-Net", ["3 seeds × 10 corridors,", "trained on the month's", "own released cells"]),
        ("Final state stages", ["gap-row models,", "S7c trees on INEL", "labels, speed clip"])],
       ["rows", "rows"])
rect(X0, 194, X1 - X0, 34, "sd-box", dash=True)
t(X0 + 12, 216, "Task 3 · physics · 15%", "sd-t")
t(X0 + 172, 216, "no model of its own: scored on our Task 1 speed and flow", "sd-s")
arrow([(502, 172), (502, 192)])
t(510, 186, "scored on", "sd-a", "start")
engine(244, 106, "Task 2 · onset · 15%", ["train", "T"],
       [("Q03 site classifier", ["LightGBM on online", "precursors"]),
        ("Q0436 switches", ["calibrated on the", "window's own history"]),
        ("Expected-IoU decode", ["top site, or top two", "if odds ratio ≥ .8"])],
       ["odds", "odds"])
engine(362, 106, "Task 2 · ongoing · 15%", ["train", "T", "M4A"],
       [("Members", ["1-D CNN × 5 seeds,", "ConvNeXt-Tiny at .2"]),
        ("Per-date fine-tune", ["pseudo windows from", "earlier dates"]),
        ("Expected-IoU decode", ["6 steps, persistence", "cap .92"])],
       ["nets", "probs"])
rect(X0, 480, X1 - X0, 60, "sd-box")
t(X0 + 12, 501, "Task 4 · ODME · 20%", "sd-t")
chips(X1 - 10, 488, ["month"])
t(X0 + 12, 526, "the organiser's NNLS: min ‖Af − c‖² + .05 ‖f − b‖², f ≥ 0, on the month's counts c and prior b", "sd-s")

# ---- output
rect(800, 40, 184, 500, "sd-out")
t(892, 70, "Submission", "sd-big", "middle")
t(892, 90, "C79 · the scored file", "sd-s", "middle")
for y, s in ((106, "state rows"), (297, "onset rows"), (415, "ongoing rows"), (510, "ODME rows")):
    arrow([(X1, y), (808, y)])
    t(816, y + 4, s, "sd-m")
t(892, 160, "public .90088", "sd-t", "middle")
t(892, 182, "private .89874", "sd-t", "middle")
t(892, 204, "8th of 802 teams", "sd-s", "middle")
t(892, 236, "March and April rows", "sd-s", "middle")
t(892, 252, "built separately", "sd-s", "middle")

# ---- how a piece got in
t(16, 568, "HOW A PIECE GOT IN", "sd-h")
DEV = [("Idea", "registry: 267 tested"), ("Offline instrument", "hides cells of the month"),
       ("Written forecast", "and band, before upload"), ("Board read", "5 a day, method only"),
       ("Adopt or refute", "45 pieces adopted")]
BX = [16 + 197 * i for i in range(5)]
for x, (a, b) in zip(BX, DEV):
    rect(x, 578, 180, 48, "sd-box")
    t(x + 12, 598, a, "sd-m"); t(x + 12, 615, b, "sd-s")
for k in range(4):
    arrow([(BX[k] + 182, 602), (BX[k + 1] - 2, 602)])
arrow([(894, 578), (894, 560), (502, 560), (502, 542)], label="adds a piece to an engine", lx=698, ly=554)

# ---- rebuild
t(16, 652, "REBUILD FROM THE PUBLIC REPOSITORY (CLEAN FOLDER, 9 NOVEMBER 2026)", "sd-h")
REB = [("Public inputs", "repo, weights-v1, data"), ("Base build C47", "WSL, about 3.5 h"),
       ("17 pieces", "rows 19 to 35, 12 h CPU"), ("Per-step check", "sha256 against development"),
       ("C79 and C80", "byte-equal to stored files")]
for x, (a, b) in zip(BX, REB):
    rect(x, 662, 180, 48, "sd-box")
    t(x + 12, 682, a, "sd-m"); t(x + 12, 699, b, "sd-s")
for k in range(4):
    arrow([(BX[k] + 182, 686), (BX[k + 1] - 2, 686)])

LABEL = ("System diagram of the 8th-place solution: three inputs feed five task engines, built once per scored month; "
         "their rows form the submitted file C79. Each piece entered through an offline instrument, a written forecast and a "
         "board read, and the whole chain rebuilds byte-equal from the public repository.")
svg = (f'<svg class="sd" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(LABEL)}" xmlns="http://www.w3.org/2000/svg">'
       '<defs><marker id="sd-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       '<path d="M0,0 L10,5 L0,10 z" class="sd-head"/></marker></defs>' + "".join(out) + "</svg>")
(HERE / "system_diagram.svg").write_text(svg, encoding="utf-8", newline="\n")

CSS = """
.sd { width: 100%; height: auto; display: block; font-family: var(--body) }
.sd .sd-box { fill: var(--surface); stroke: var(--rule); stroke-width: 1.2 }
.sd .sd-mini { fill: var(--bg); stroke: var(--rule) }
.sd .sd-out { fill: var(--sunk); stroke: var(--ink); stroke-width: 1.5 }
.sd .sd-ln { stroke: var(--ink); stroke-width: 1.4 } .sd .sd-head { fill: var(--ink) }
.sd text { fill: var(--ink) }
.sd .sd-h { font: 700 11px var(--display); letter-spacing: .08em; fill: var(--muted) }
.sd .sd-t { font: 700 13.5px var(--display) } .sd .sd-big { font: 900 18px var(--display) }
.sd .sd-m { font: 700 12px var(--display) } .sd .sd-s { font-size: 11.5px; fill: var(--muted) }
.sd .sd-a { font: 600 10px var(--mono); fill: var(--muted) }
.sd .sd-chip { font: 700 10px var(--mono) }
.sd .sd-c-train { fill: none; stroke: var(--muted) } .sd .sd-c-train-t { fill: var(--muted) }
.sd .sd-c-month { fill: none; stroke: var(--c-state) } .sd .sd-c-month-t { fill: var(--c-state) }
.sd .sd-c-online { fill: none; stroke: var(--pub) } .sd .sd-c-online-t { fill: var(--pub) }
.sd .sd-c-m4a { fill: none; stroke: var(--priv) } .sd .sd-c-m4a-t { fill: var(--priv) }
"""
(HERE / "system_diagram.css").write_text(CSS.strip() + "\n", encoding="utf-8", newline="\n")

# light-theme page for the PNG (tokens copied from the site's light palette)
page = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Overpass:wght@400;600;700;900&family=Overpass+Mono:wght@600;700&family=Atkinson+Hyperlegible:wght@400;700&display=swap">
<style>
:root {{ --bg: #eef1ed; --surface: #f9fbf8; --sunk: #e3e8e2; --ink: #14201a; --muted: #56675d; --rule: #c6d1c9;
  --pub: #2b6cb0; --priv: #c2410c; --c-state: #0f7b6c;
  --display: "Overpass", "Arial Narrow", sans-serif; --body: "Atkinson Hyperlegible", "Segoe UI", sans-serif; --mono: "Overpass Mono", Consolas, monospace }}
html, body {{ margin: 0; background: #ffffff }} .wrap {{ width: 1000px; padding: 16px 20px 18px; background: #ffffff }}
.legend {{ font: 12px var(--body); color: var(--muted); margin-top: 6px }}
{CSS}
</style></head><body><div class="wrap">{svg}
<div class="legend">Tags = what an engine may read: <b style="color:var(--muted)">train</b> train months ·
<b style="color:var(--c-state)">month</b> released cells of the scored month (offline Tasks 1, 3, 4) ·
<b style="color:var(--pub)">≤ T</b> data stamped at or before the forecast origin (online Task 2) ·
<b style="color:var(--priv)">Mar→Apr</b> a March choice carried to April by a declared rule.</div></div></body></html>"""
OUT = HERE.parent / "figures" / "out"
(OUT / "system_diagram.html").write_text(page, encoding="utf-8", newline="\n")
print("system_diagram.svg", len(svg), "bytes")
