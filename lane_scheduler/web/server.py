"""
Lightweight HTTP dashboard server for the Lane Scheduler.

Serves:
  GET /              → self-contained HTML dashboard (polls /api/snapshot via JS)
  GET /api/snapshot  → JSON snapshot of current queue state

Started as a daemon thread by start_server(); the controller keeps running
if the server crashes.  Per-request access logs are suppressed to reduce noise.
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lane_scheduler.k8s.controller import LaneSchedulerController

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dashboard HTML (self-contained — no external CDN dependencies)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = b"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lane Scheduler &mdash; Queue Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f0f2f5;color:#222;font-size:14px}
header{background:#1a3a5c;color:#fff;padding:12px 20px;display:flex;justify-content:space-between;align-items:center;gap:12px}
header h1{font-size:1rem;font-weight:600;letter-spacing:.02em}
#hdr-status{font-size:.8rem;opacity:.75;white-space:nowrap}
main{max-width:1300px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:16px}
.card{background:#fff;border-radius:6px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.card-title{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#1a3a5c;margin-bottom:12px}
table{width:100%;border-collapse:collapse}
th{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#666;padding:6px 10px;border-bottom:2px solid #e8e8e8;white-space:nowrap;text-align:left}
td{padding:6px 10px;border-bottom:1px solid #f0f0f0;vertical-align:middle}
tr:last-child td{border-bottom:none}
.r{text-align:right;font-variant-numeric:tabular-nums}
.mono{font-family:ui-monospace,monospace}
/* utilization bar */
.bar-wrap{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.bar{width:72px;height:8px;background:#e0e0e0;border-radius:4px;overflow:hidden;flex-shrink:0}
.bar-fill{height:100%;border-radius:4px;background:#4caf50;transition:width .3s}
.bar-fill.warn{background:#ff9800}
.bar-fill.crit{background:#f44336}
/* tier badges */
.badge{display:inline-block;border-radius:3px;padding:1px 6px;font-size:.7rem;font-weight:600}
.t-intro{background:#e3f2fd;color:#1565c0}
.t-upper{background:#e8f5e9;color:#2e7d32}
.t-grad{background:#fce4ec;color:#c62828}
/* wait cells */
.w-med{font-weight:600}
.w-rng{color:#888;font-size:.8rem}
/* health footer */
footer{background:#fff;border-top:1px solid #e0e0e0;padding:8px 20px;display:flex;gap:20px;flex-wrap:wrap;font-size:.78rem;color:#555}
footer span{white-space:nowrap;display:flex;align-items:center;gap:5px}
.dot{width:8px;height:8px;border-radius:50%;background:#4caf50;flex-shrink:0}
.dot.warn{background:#ff9800}
.dot.stale{background:#f44336}
.empty{text-align:center;color:#aaa;padding:20px;font-style:italic}
</style>
</head>
<body>
<header>
  <h1>Lane Scheduler &mdash; Queue Dashboard</h1>
  <span id="hdr-status">Loading…</span>
</header>

<main>
  <div class="card" id="lanes-card">
    <div class="card-title">Lane Status</div>
    <table>
      <thead>
        <tr>
          <th>Lane</th>
          <th class="r">Nodes</th>
          <th class="r">Capacity</th>
          <th>Utilization</th>
          <th class="r">Queued</th>
          <th class="r">Est. drain (P80)</th>
        </tr>
      </thead>
      <tbody id="lane-body"><tr><td colspan="6" class="empty">Loading…</td></tr></tbody>
    </table>
  </div>

  <div class="card" id="courses-card">
    <div class="card-title">Course Queue</div>
    <table>
      <thead>
        <tr>
          <th>Course</th>
          <th class="r">Weight</th>
          <th>Lane</th>
          <th class="r">Running</th>
          <th class="r">Queued</th>
          <th>Top wait (P50)</th>
          <th>Tail wait (P80)</th>
        </tr>
      </thead>
      <tbody id="course-body"><tr><td colspan="7" class="empty">Loading…</td></tr></tbody>
    </table>
  </div>
</main>

<footer id="health-strip"><span>Loading…</span></footer>

<script>
const REFRESH_MS = 10000;
let lastFetchAt = null;

function dur(s) {
  if (s === null || s === undefined) return '–';
  if (s < 60)   return Math.round(s) + 's';
  if (s < 3600) return (s / 60).toFixed(1) + 'm';
  return (s / 3600).toFixed(1) + 'h';
}

function fmtWait(w) {
  if (!w) return '–';
  return '<span class="w-med">' + dur(w.median_s) + '</span>'
       + ' <span class="w-rng">(' + dur(w.p20_s) + '–' + dur(w.p80_s) + ')</span>';
}

function barCls(used, cap) {
  if (cap <= 0) return '';
  const r = used / cap;
  return r >= 0.9 ? 'crit' : r >= 0.7 ? 'warn' : '';
}

function renderLanes(lanes) {
  const tb = document.getElementById('lane-body');
  if (!lanes || !lanes.length) {
    tb.innerHTML = '<tr><td colspan="6" class="empty">No lanes configured</td></tr>';
    return;
  }
  tb.innerHTML = lanes.map(function(ln) {
    const pct = ln.capacity_units > 0
      ? Math.min(100, (ln.running_units / ln.capacity_units * 100)).toFixed(0)
      : 0;
    const cls = barCls(ln.running_units, ln.capacity_units);
    const capLabel = ln.capacity_units + ' GPUs';
    const runLabel = ln.running_units + ' / ' + ln.capacity_units
                   + ' (' + pct + '%)';
    return '<tr>'
      + '<td><strong class="mono">' + ln.name + '</strong></td>'
      + '<td class="r">' + ln.node_count + '</td>'
      + '<td class="r">' + capLabel + '</td>'
      + '<td><div class="bar-wrap"><div class="bar"><div class="bar-fill ' + cls
        + '" style="width:' + pct + '%"></div></div>'
        + '<span>' + runLabel + '</span></div></td>'
      + '<td class="r">' + ln.queued_count + '</td>'
      + '<td class="r">' + dur(ln.drain_p80_s) + '</td>'
      + '</tr>';
  }).join('');
}


function renderCourses(courses) {
  const tb = document.getElementById('course-body');
  if (!courses || !courses.length) {
    tb.innerHTML = '<tr><td colspan="7" class="empty">No courses in queue</td></tr>';
    return;
  }
  const rows = [];
  courses.forEach(function(c) {
    const n = c.lanes.length;
    c.lanes.forEach(function(l, i) {
      const tailCell = l.queued_count > 1
        ? fmtWait(l.tail_wait)
        : '–';
      if (i === 0) {
        rows.push('<tr>'
          + '<td rowspan="' + n + '"><strong class="mono">' + c.course_id + '</strong></td>'
          + '<td class="r" rowspan="' + n + '">' + c.weight.toFixed(3) + '</td>'
          + '<td class="mono">' + l.lane_name + '</td>'
          + '<td class="r">' + l.running_count + '</td>'
          + '<td class="r">' + l.queued_count + '</td>'
          + '<td>' + fmtWait(l.top_wait) + '</td>'
          + '<td>' + tailCell + '</td>'
          + '</tr>');
      } else {
        rows.push('<tr>'
          + '<td class="mono">' + l.lane_name + '</td>'
          + '<td class="r">' + l.running_count + '</td>'
          + '<td class="r">' + l.queued_count + '</td>'
          + '<td>' + fmtWait(l.top_wait) + '</td>'
          + '<td>' + tailCell + '</td>'
          + '</tr>');
      }
    });
  });
  tb.innerHTML = rows.join('');
}

function renderHealth(sys, generatedAt) {
  const age = sys.wait_cache_age_s;
  const dotCls = age === null ? 'stale' : age < 90 ? '' : age < 180 ? 'warn' : 'stale';
  const ageStr = age === null ? 'never' : dur(age) + ' ago';
  document.getElementById('health-strip').innerHTML =
      '<span><span class="dot ' + dotCls + '"></span>'
      + 'Wait cache: ' + ageStr + ' (computed in ' + sys.wait_cache_duration_s + 's)</span>'
    + '<span>Cycle: every ' + sys.cycle_interval_s + 's</span>'
    + '<span>' + sys.course_count + ' courses registered</span>'
    + '<span>Snapshot: ' + new Date(generatedAt).toLocaleTimeString() + '</span>';
}

function updateHeaderStatus(ok) {
  const el = document.getElementById('hdr-status');
  if (!ok) { el.textContent = '⚠️ fetch error'; return; }
  if (!lastFetchAt) return;
  const age = ((Date.now() - lastFetchAt) / 1000).toFixed(0);
  el.textContent = 'Updated ' + age + 's ago · auto-refresh every ' + (REFRESH_MS/1000) + 's';
}

async function fetchAndRender() {
  try {
    const resp = await fetch('/api/snapshot');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    lastFetchAt = Date.now();
    renderLanes(data.lanes);
    renderCourses(data.courses);
    renderHealth(data.system, data.generated_at);
    updateHeaderStatus(true);
  } catch (e) {
    console.error('Dashboard fetch error:', e);
    updateHeaderStatus(false);
  }
}

fetchAndRender();
setInterval(fetchAndRender, REFRESH_MS);
setInterval(function() { if (lastFetchAt) updateHeaderStatus(true); }, 1000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

def _make_handler(controller: "LaneSchedulerController"):
    """Return a BaseHTTPRequestHandler subclass bound to *controller*."""

    class _Handler(BaseHTTPRequestHandler):

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._serve(200, "text/html; charset=utf-8", _DASHBOARD_HTML)
            elif path == "/api/snapshot":
                self._serve_snapshot()
            else:
                self.send_error(404)

        def _serve(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_snapshot(self) -> None:
            try:
                from lane_scheduler.web.snapshot import build_snapshot
                data = build_snapshot(controller)
                body = json.dumps(data, separators=(",", ":")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                logger.error("Snapshot build failed: %s", exc, exc_info=True)
                self.send_error(500)

        def log_message(self, fmt, *args) -> None:  # type: ignore[override]
            pass  # suppress per-request access logs

    return _Handler


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def start_server(controller: "LaneSchedulerController", port: int) -> threading.Thread:
    """
    Start the HTTP dashboard server on *port* in a daemon thread.

    Returns the thread (already started).  The controller keeps running if the
    server encounters an unhandled exception.
    """
    server = HTTPServer(("", port), _make_handler(controller))
    server.allow_reuse_address = True

    def _target() -> None:
        try:
            server.serve_forever()
        except Exception as exc:
            logger.error("Dashboard server crashed: %s", exc, exc_info=True)

    thread = threading.Thread(target=_target, name="web-dashboard", daemon=True)
    thread.start()
    logger.info("Dashboard available at http://0.0.0.0:%d/", port)
    return thread
