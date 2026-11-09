"""Build the solution video: narration (audio/*.mp3) + animated scenes (scenes_template.html) -> out/trafficflowbench_8th.mp4.

  python report/video/build_video.py [--fps 30] [--only-html] [--from-s S --to-s S]

1. Timeline: one scene per narration clip (narration.json order); scene length = lead + clip + tail.
2. build/scenes.html: the template with the data inlined (report/figures/out tables, the registry, the system diagram,
   the clean-folder log lines). Every number on screen comes from those files.
3. capture.mjs (Node 22+, no packages): headless Edge/Chrome over the DevTools protocol renders frame t = i / fps by
   calling window.renderFrame(t) and pipes JPEG frames into ffmpeg.
4. ffmpeg: narration clips placed at their scene start (adelay) and muxed -> H.264 + AAC, 1920x1080.
Needs: ffmpeg/ffprobe on PATH, Node 22+, Microsoft Edge or Chrome. Narration voice: ElevenLabs (credit in the end card).
"""
import argparse, csv, json, os, shutil, subprocess, sys
from pathlib import Path

V = Path(__file__).resolve().parent
REP = V.parent
OUT = REP / "figures" / "out"
FS = REP.parent
LEAD, TAIL, TITLE_LEAD = 0.7, 1.0, 1.4


def dur(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def rows(name, base=OUT, delim=","):
    with open(base / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delim))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def timeline():
    nar = json.loads((V / "narration.json").read_text(encoding="utf-8"))
    clips = sorted((V / "audio").glob("*.mp3"))
    assert len(clips) == len(nar), (len(clips), len(nar))
    t, sc = 0.0, []
    for i, (n, c) in enumerate(zip(nar, clips)):
        assert c.stem.split("_", 1)[1] == n["id"], (c.name, n["id"])
        lead = TITLE_LEAD if i == 0 else LEAD
        d = dur(c)
        tail = 2.2 if i == len(nar) - 1 else TAIL
        sc.append(dict(id=n["id"], text=n["text"], clip=str(c), start=round(t, 3), lead=lead, audio=round(d, 3),
                       dur=round(lead + d + tail, 3)))
        t += lead + d + tail
    return sc, round(t, 3)


def data():
    D = {}
    st = rows("F6_spacetime.csv")
    ns, nt = 1 + max(int(r["station_rank"]) for r in st), 1 + max(int(r["step"]) for r in st)
    mask, err = [[0] * nt for _ in range(ns)], [[None] * nt for _ in range(ns)]
    for r in st:
        i, t = int(r["station_rank"]), int(r["step"])
        mask[i][t] = 2 if r["scored_target"] == "True" else (1 if r["inel"] == "True" else 0)
        if mask[i][t] == 1 and r["err_kmh"]:
            err[i][t] = float(r["err_kmh"])
    D["f6"] = dict(ns=ns, nt=nt, mask=mask, err=err)
    f4 = [r for r in rows("F4_transfer_rule.csv") if not r["excluded_reason"] and r["ratio"]]
    D["f4"] = [dict(g=r["group"], c=r["cnum"], r=float(r["ratio"])) for r in f4]
    f3 = [r for r in rows("F3_forecast_vs_board.csv") if r["d_pred"] and r["d_board"]]
    D["f3"] = [dict(k=r["kind"], x=float(r["d_pred"]), y=float(r["d_board"])) for r in f3]
    p1 = rows("P1_public_private.csv")
    D["p1"] = [dict(f=r["file"], d=r["date"], k=r["kind"], pub=fnum(r["public"]), priv=fnum(r["private"])) for r in p1]
    D["f10"] = [dict(s=r["step"], v=float(r["board_increment"])) for r in rows("F10_label_dose_board.csv")
                if not r["step"].startswith(("INEL-FOLD", "S7C-FOLD"))]
    reg = rows("registry.tsv", REP / "data", "\t")
    cnt = {}
    for r in reg:
        cnt[r["verdict_class"]] = cnt.get(r["verdict_class"], 0) + 1
    D["reg"] = cnt
    D["summary"] = json.loads((OUT / "P_summary.json").read_text(encoding="utf-8"))
    log = (FS / "checks" / "clean_room" / "run_2026-11-09" / "run.log").read_text(encoding="utf-8").splitlines()
    D["log"] = dict(c47=next(l for l in log if "STEP 10 C47" in l), builds=sum(" SHA_EQUAL" in l for l in log),
                    c79=next(l for l in log if "STEP 35 C79 (csv)" in l), c80=next(l for l in log if "STEP 35 C80 (csv)" in l))
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--only-html", action="store_true")
    ap.add_argument("--from-s", type=float, default=0)
    ap.add_argument("--to-s", type=float, default=None)
    ap.add_argument("--stills", default="", help="comma list of times: write PNG stills to build/stills instead of a video")
    a = ap.parse_args()
    sc, total = timeline()
    B = V / "build"
    B.mkdir(exist_ok=True)
    svg = (REP / "site" / "system_diagram.svg").read_text(encoding="utf-8")
    css = (REP / "site" / "system_diagram.css").read_text(encoding="utf-8")
    blob = json.dumps(dict(scenes=sc, total=total, data=data()), separators=(",", ":")).replace("</", "<\\/")
    html = (V / "scenes_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", blob).replace("<!--__SVG__-->", svg).replace("/*__SDCSS__*/", css)
    (B / "scenes.html").write_text(html, encoding="utf-8", newline="\n")
    print(f"timeline {total:.1f} s, {len(sc)} scenes: " + ", ".join(f"{s['id']} {s['dur']:.1f}" for s in sc))
    if a.only_html:
        return
    node = shutil.which("node") or str(Path.home() / "tools/node-v24.21.0-win-x64/node.exe")
    t1 = a.to_s if a.to_s is not None else total
    if a.stills:
        subprocess.run([node, str(V / "capture.mjs"), str(B / "scenes.html"), "stills", a.stills, str(B / "stills")], check=True)
        return
    silent = B / "video_silent.mp4"
    subprocess.run([node, str(V / "capture.mjs"), str(B / "scenes.html"), "video", str(a.fps), str(a.from_s), str(t1), str(silent)], check=True)
    # narration track: each clip delayed to its scene start + lead
    ins, flt = [], []
    for i, s in enumerate(sc):
        ins += ["-i", s["clip"]]
        ms = int(round((s["start"] + s["lead"] - a.from_s) * 1000))
        if ms < 0:
            continue
        flt.append(f"[{i + 1}:a]aresample=48000,adelay={ms}|{ms}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(sc)) if f"[a{i}]" in "".join(flt))
    n = mix.count("[")
    fc = ";".join(flt) + f";{mix}amix=inputs={n}:normalize=0,apad[aout]"
    out = V / "out"
    out.mkdir(exist_ok=True)
    mp4 = out / "trafficflowbench_8th_place.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), *ins, "-filter_complex", fc, "-map", "0:v", "-map", "[aout]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(mp4)], check=True)
    print(f"{mp4} {mp4.stat().st_size / 1e6:.1f} MB, {dur(mp4):.1f} s")


if __name__ == "__main__":
    main()
