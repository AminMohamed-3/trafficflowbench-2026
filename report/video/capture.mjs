// capture.mjs: render build/scenes.html frame by frame in headless Edge/Chrome (DevTools protocol, Node 22+ built-ins only).
//   node capture.mjs SCENES.html video FPS T0 T1 OUT.mp4     frames t = T0 + i/FPS as JPEG files, then ffmpeg (H.264)
//   node capture.mjs SCENES.html stills "t1,t2,..." OUTDIR    PNG stills for review
// The page exposes window.renderFrame(t) (deterministic: the picture depends only on t) and sets window.__ready.
// Headless screenshots occasionally hang; a hung call times out, the browser is restarted and capture resumes at that frame.
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const [, , htmlPath, mode, ...rest] = process.argv;
const BROWSERS = [process.env.BROWSER, "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe", "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/microsoft-edge"];
const exe = BROWSERS.find(p => p && fs.existsSync(p));
if (!exe) { console.error("no Edge or Chrome found (set BROWSER)"); process.exit(2); }
const sleep = ms => new Promise(r => setTimeout(r, ms));
const url = "file:///" + path.resolve(htmlPath).split(path.sep).join("/");

async function launch() {
  const port = 9300 + Math.floor(Math.random() * 600);
  const udd = fs.mkdtempSync(path.join(os.tmpdir(), "tfbcap-"));
  const br = spawn(exe, ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${udd}`, "--hide-scrollbars",
    "--mute-audio", "--force-device-scale-factor=1", "--window-size=1920,1080", "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows", "--disable-features=CalculateNativeWinOcclusion",
    "about:blank"], { stdio: "ignore" });
  let tgt, ws;
  const close = async () => { try { ws?.close(); } catch { } br.kill(); await sleep(700); try { fs.rmSync(udd, { recursive: true, force: true }); } catch { } };
  for (let i = 0; i < 150 && !tgt; i++) {
    try { tgt = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json(); } catch { await sleep(200); }
  }
  if (!tgt) { await close(); throw new Error("browser did not start"); }
  ws = new WebSocket(tgt.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let nid = 0; const pend = new Map();
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id && pend.has(m.id)) { const p = pend.get(m.id); pend.delete(m.id); m.error ? p.rej(new Error(JSON.stringify(m.error))) : p.res(m.result); }
  };
  const send = (method, params = {}, ms = 15000) => new Promise((res, rej) => {
    const id = ++nid, to = setTimeout(() => { pend.delete(id); rej(new Error(`${method} timed out after ${ms} ms`)); }, ms);
    pend.set(id, { res: v => { clearTimeout(to); res(v); }, rej: e => { clearTimeout(to); rej(e); } });
    ws.send(JSON.stringify({ id, method, params }));
  });
  const ev = async expr => {
    const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || JSON.stringify(r.exceptionDetails));
    return r.result.value;
  };
  await send("Page.enable"); await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url });
  let ok = false;
  for (let i = 0; i < 300 && !ok; i++) { try { ok = await ev("window.__ready === true"); } catch { } if (!ok) await sleep(100); }
  if (!ok) { await close(); throw new Error("page not ready"); }
  await ev("document.fonts.ready.then(() => document.fonts.size)");
  const frame = async (t, fmt) => {
    await ev(`window.renderFrame(${t}); 1`);  // captureScreenshot waits for a fresh compositor frame
    const r = await send("Page.captureScreenshot", fmt === "png" ? { format: "png" } : { format: "jpeg", quality: 94, optimizeForSpeed: true });
    return Buffer.from(r.data, "base64");
  };
  return { frame, close };
}

// run job(b, i) for i in [0, n), restarting the browser after a failure (at most 30 restarts)
async function run(n, job) {
  let i = 0, restarts = 0;
  while (i < n) {
    const b = await launch();
    try { for (; i < n; i++) await job(b, i); }
    catch (e) { if (++restarts > 30) { await b.close(); throw e; } console.log(`frame ${i}: ${e.message}; restarting the browser (${restarts})`); }
    await b.close();
  }
}

if (mode === "stills") {
  const [times, outDir] = rest; fs.mkdirSync(outDir, { recursive: true });
  const ts = times.split(",").map(Number);
  await run(ts.length, async (b, i) => fs.writeFileSync(path.join(outDir, `t${ts[i].toFixed(1).padStart(6, "0")}.png`), await b.frame(ts[i], "png")));
  console.log("stills", times);
} else {
  const [fpsS, t0S, t1S, out] = rest; const fps = +fpsS, t0 = +t0S, t1 = +t1S;
  const n = Math.round((t1 - t0) * fps), fdir = path.join(path.dirname(out), "frames");
  fs.rmSync(fdir, { recursive: true, force: true }); fs.mkdirSync(fdir, { recursive: true });
  const T = Date.now();
  await run(n, async (b, i) => {
    fs.writeFileSync(path.join(fdir, `f${String(i).padStart(5, "0")}.jpg`), await b.frame(t0 + i / fps, "jpeg"));
    if (i % (fps * 10) === 0) console.log(`frame ${i}/${n} t=${(t0 + i / fps).toFixed(1)} s, ${((Date.now() - T) / 1000).toFixed(0)} s elapsed`);
  });
  console.log(`captured ${n} frames in ${((Date.now() - T) / 1000).toFixed(0)} s; encoding`);
  const ff = spawn("ffmpeg", ["-y", "-v", "error", "-framerate", String(fps), "-i", path.join(fdir, "f%05d.jpg"),
    "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-r", String(fps), "-movflags", "+faststart", out],
    { stdio: ["ignore", "inherit", "inherit"] });
  const code = await new Promise(r => ff.on("close", r));
  if (code !== 0) throw new Error("ffmpeg exit " + code);
  fs.rmSync(fdir, { recursive: true, force: true });
  console.log(`video ${out}: ${n} frames in ${((Date.now() - T) / 1000).toFixed(0)} s`);
}
