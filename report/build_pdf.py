"""Build the PDF report: report/REPORT.md -> report/REPORT.html (print layout) -> report/REPORT.pdf.

  python report/build_pdf.py [--edge PATH]        (needs the `markdown` package and Microsoft Edge or Chrome)

Figures are read from report/figures/out/*.png by relative path, so run figures first (make_figures.py, f6_from_csv.py,
site/system_diagram.py). The PDF is printed by headless Edge/Chrome (no header or footer).
"""
import argparse, html, re, shutil, subprocess, sys
from pathlib import Path

import markdown

REP = Path(__file__).resolve().parent
CANDIDATES = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe", "/usr/bin/google-chrome", "/usr/bin/chromium",
              "/usr/bin/microsoft-edge"]

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #1a1a1a; }
body { margin: 0; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 6pt; color: #0b3d2c; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; padding-bottom: 2pt; border-bottom: 1px solid #c9d1cb; color: #0b3d2c; break-after: avoid; }
h3 { font-size: 10.8pt; margin: 11pt 0 4pt; break-after: avoid; }
h4 { font-size: 9.8pt; margin: 9pt 0 3pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 6pt; }
ul, ol { margin: 0 0 6pt; padding-left: 16pt; }
li { margin: 0 0 2pt; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.4pt; background: #f2f4f2; padding: 0 2px; border-radius: 2px; }
pre { background: #f2f4f2; padding: 6pt 8pt; font-size: 8pt; line-height: 1.35; white-space: pre-wrap; word-break: break-word; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 9pt; font-size: 7.9pt; line-height: 1.3; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border-bottom: 1px solid #dde3de; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }
th { background: #eef2ee; font-weight: 600; border-bottom: 1px solid #9fb0a4; }
img { display: block; width: auto; max-width: 100%; max-height: 112mm; margin: 6pt auto 2pt; break-inside: avoid; }
img.tall { max-height: 235mm; }
p:has(> img) { break-inside: avoid; break-after: avoid; margin-bottom: 0; }
p:has(> img) + p > em:only-child { display: block; font-size: 8.4pt; color: #444; margin-bottom: 8pt; }
hr { border: 0; border-top: 1px solid #c9d1cb; margin: 10pt 0; }
a { color: #0b6644; text-decoration: none; word-break: break-word; }
.meta { color: #444; margin: 0 0 3pt; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default=None)
    a = ap.parse_args()
    md = (REP / "REPORT.md").read_text(encoding="utf-8")
    title = re.match(r"#\s+(.+)", md).group(1).strip()
    # the report hard-wraps lists right under a paragraph line; Python-Markdown needs a blank line before a list
    item = re.compile(r"^(?:\d{1,2}\.|[-*]) \S")
    lines, prev = [], ""
    for ln in md.splitlines():
        if item.match(ln) and prev.strip() and not item.match(prev) and not prev.startswith(("|", "    ", "#")) and not prev.startswith(" "):
            lines.append("")
        lines.append(ln)
        prev = ln if ln.strip() else ""
    md = "\n".join(lines)
    md = re.sub(r"(?<![<(\[\"])(https?://[^\s)\]]+[^\s)\].,])", r"<\1>", md)   # bare URLs become links
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"], output_format="html5")
    body = body.replace('src="figures/out/F2_waterfall.png"', 'class="tall" src="figures/out/F2_waterfall.png"')
    page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    out_html = REP / "REPORT.html"
    out_html.write_text(page, encoding="utf-8", newline="\n")
    exe = a.edge or next((c for c in CANDIDATES if Path(c).exists()), None) or shutil.which("msedge") or shutil.which("chrome")
    if not exe:
        sys.exit("no Edge or Chrome found; open report/REPORT.html in a browser and print to PDF")
    pdf = REP / "REPORT.pdf"
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--generate-pdf-document-outline",
                    f"--print-to-pdf={pdf}", "--virtual-time-budget=10000", out_html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"REPORT.pdf {pdf.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
