"""
Flask web server for browsing and downloading collected market data.
Run with: python app.py
Then open: http://localhost:5050
"""
import os
import io
import zipfile
import json
import threading
from datetime import datetime
import pandas as pd
from flask import Flask, render_template_string, send_file, jsonify, abort
from collector import collect_all, TIMEFRAME_MAP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

app = Flask(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def file_meta(path):
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        rows = len(df)
        start = str(df.index[0])[:19] if rows else "—"
        end   = str(df.index[-1])[:19] if rows else "—"
    except Exception:
        rows, start, end = 0, "—", "—"
    return {
        "path": path,
        "rows": rows,
        "start": start,
        "end": end,
        "size_kb": round(stat.st_size / 1024, 1),
        "updated": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def collect_file_tree():
    tree = {}
    for category in ("live", "historical"):
        cat_dir = os.path.join(DATA_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        tree[category] = {}
        for symbol in sorted(os.listdir(cat_dir)):
            sym_dir = os.path.join(cat_dir, symbol)
            if not os.path.isdir(sym_dir):
                continue
            tree[category][symbol] = {}
            if category == "live":
                for dtype in ("ohlcv", "indicators"):
                    dtype_dir = os.path.join(sym_dir, dtype)
                    if not os.path.isdir(dtype_dir):
                        continue
                    tree[category][symbol][dtype] = {}
                    for fname in sorted(os.listdir(dtype_dir)):
                        if not fname.endswith(".csv"):
                            continue
                        tf = fname[:-4]
                        meta = file_meta(os.path.join(dtype_dir, fname))
                        if meta:
                            tree[category][symbol][dtype][tf] = meta
            else:
                for fname in sorted(os.listdir(sym_dir)):
                    if not fname.endswith(".csv"):
                        continue
                    tf = fname[:-4]
                    meta = file_meta(os.path.join(sym_dir, fname))
                    if meta:
                        tree[category][symbol][tf] = meta
    return tree


INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="60">
<title>PulseVault — Market Data Hub</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', sans-serif;
    background: #f0faf4;
    color: #1a2e1a;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    background: linear-gradient(135deg, #14532d 0%, #166534 60%, #15803d 100%);
    padding: 28px 40px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(20,83,45,0.3);
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .logo-icon {
    width: 46px; height: 46px;
    background: #22c55e;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
  }
  .logo-text h1 {
    font-size: 26px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;
  }
  .logo-text p {
    font-size: 12px; color: #86efac; font-weight: 400; margin-top: 2px;
  }
  .header-right {
    display: flex; align-items: center; gap: 12px;
  }
  .status-dot {
    width: 9px; height: 9px; background: #4ade80;
    border-radius: 50%; animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .status-label { color: #bbf7d0; font-size: 13px; font-weight: 500; }
  .header-link {
    color: #86efac; font-size: 13px; text-decoration: none;
    background: rgba(255,255,255,0.1); padding: 5px 12px;
    border-radius: 6px; border: 1px solid rgba(255,255,255,0.15);
  }
  .header-link:hover { background: rgba(255,255,255,0.18); }
  .btn-collect {
    background: #16a34a; color: #fff; border: none; border-radius: 6px;
    padding: 7px 16px; font-size: 13px; font-weight: 600; cursor: pointer;
    font-family: 'Inter', sans-serif;
  }
  .btn-collect:hover { background: #15803d; }
  .btn-collect:disabled { background: #6b7280; cursor: not-allowed; }
  #collect-msg { color: #86efac; font-size: 12px; margin-left: 8px; }

  /* ── Symbols bar ── */
  .symbols-bar {
    background: #ffffff;
    border-bottom: 1px solid #d1fae5;
    padding: 14px 40px;
    display: flex; align-items: center; gap: 10px;
    font-size: 13px; color: #374151;
  }
  .sym-chip {
    background: #dcfce7; color: #15803d;
    border: 1px solid #bbf7d0;
    border-radius: 20px; padding: 4px 14px;
    font-weight: 600; font-size: 13px;
  }

  /* ── Main layout ── */
  main { max-width: 1200px; margin: 0 auto; padding: 32px 40px; }

  /* ── Download cards ── */
  .download-section { margin-bottom: 36px; }
  .section-title {
    font-size: 11px; font-weight: 600; color: #6b7280;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;
  }
  .download-cards { display: flex; gap: 12px; flex-wrap: wrap; }
  .dl-card {
    display: flex; align-items: center; gap: 10px;
    background: #ffffff; border: 1px solid #bbf7d0;
    border-radius: 10px; padding: 12px 18px;
    text-decoration: none; color: #14532d;
    font-weight: 600; font-size: 14px;
    box-shadow: 0 1px 4px rgba(20,83,45,0.08);
    transition: all 0.15s;
  }
  .dl-card:hover {
    background: #f0fdf4; border-color: #22c55e;
    box-shadow: 0 4px 12px rgba(34,197,94,0.2);
    transform: translateY(-1px);
  }
  .dl-card .icon { font-size: 18px; }
  .dl-card.primary {
    background: linear-gradient(135deg, #16a34a, #15803d);
    color: #ffffff; border-color: transparent;
    box-shadow: 0 2px 8px rgba(22,163,74,0.35);
  }
  .dl-card.primary:hover {
    background: linear-gradient(135deg, #15803d, #14532d);
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(22,163,74,0.4);
  }

  /* ── Category sections ── */
  .category-block { margin-bottom: 40px; }
  .category-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 20px; padding-bottom: 10px;
    border-bottom: 2px solid #d1fae5;
  }
  .category-header h2 {
    font-size: 20px; font-weight: 700; color: #14532d;
  }
  .cat-badge {
    font-size: 11px; font-weight: 600; padding: 3px 10px;
    border-radius: 12px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .cat-badge.live { background: #dcfce7; color: #15803d; }
  .cat-badge.historical { background: #fef9c3; color: #92400e; }

  /* ── Symbol card ── */
  .symbol-card {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 14px; margin-bottom: 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    overflow: hidden;
  }
  .symbol-card-header {
    background: linear-gradient(90deg, #f0fdf4, #ffffff);
    padding: 14px 20px; border-bottom: 1px solid #e5e7eb;
    display: flex; align-items: center; gap: 12px;
  }
  .symbol-name {
    font-size: 17px; font-weight: 700; color: #14532d;
  }
  .symbol-card-body { padding: 16px 20px; }

  /* ── Data type tabs ── */
  .dtype-block { margin-bottom: 20px; }
  .dtype-label {
    font-size: 11px; font-weight: 600; color: #9ca3af;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 8px; padding-left: 4px;
  }

  /* ── Table ── */
  table { width: 100%; border-collapse: collapse; }
  thead tr { background: #f9fafb; }
  th {
    padding: 9px 14px; text-align: left;
    font-size: 11px; font-weight: 600; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 1px solid #e5e7eb;
  }
  td {
    padding: 10px 14px; font-size: 13px; color: #374151;
    border-bottom: 1px solid #f3f4f6;
  }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: #f0fdf4; }

  .tf-badge {
    background: #dcfce7; color: #15803d;
    border: 1px solid #bbf7d0;
    border-radius: 6px; padding: 2px 9px;
    font-size: 12px; font-weight: 600; font-family: monospace;
  }
  .rows-count { font-weight: 600; color: #111827; }
  .date-range { color: #6b7280; font-size: 12px; font-family: monospace; }
  .size-tag { color: #9ca3af; font-size: 12px; }

  .btn-dl {
    display: inline-flex; align-items: center; gap: 5px;
    background: #16a34a; color: #ffffff;
    border-radius: 6px; padding: 5px 12px;
    font-size: 12px; font-weight: 600;
    text-decoration: none; transition: background 0.15s;
  }
  .btn-dl:hover { background: #15803d; }

  /* ── Footer ── */
  footer {
    text-align: center; padding: 24px;
    color: #9ca3af; font-size: 12px;
    border-top: 1px solid #e5e7eb; margin-top: 20px;
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">&#128200;</div>
    <div class="logo-text">
      <h1>PulseVault</h1>
      <p>Futures Market Data Hub &mdash; NQ1! &bull; ES1!</p>
    </div>
  </div>
  <div class="header-right">
    <div class="status-dot"></div>
    <span class="status-label">Connected</span>
    <a href="/status" class="header-link">JSON Status</a>
    <button class="btn-collect" id="collectBtn" onclick="triggerCollect()">Collect Now</button>
    <span id="collect-msg"></span>
  </div>
</header>

<div class="symbols-bar">
  <span style="color:#6b7280; font-weight:500;">Tracking:</span>
  <span class="sym-chip">NQ1!</span>
  <span class="sym-chip">ES1!</span>
  <span style="margin-left:auto; color:#9ca3af;">Timeframes: 1m &bull; 3m &bull; 5m &bull; 15m &bull; 1H &bull; 4H &bull; Daily &bull; Weekly</span>
</div>

<main>

  <div class="download-section">
    <div class="section-title">Bulk Downloads</div>
    <div class="download-cards">
      <a href="/download/zip/all" class="dl-card primary">
        <span class="icon">&#8659;</span> Download Everything (ZIP)
      </a>
      {% for sym in ['NQ1!','ES1!'] %}
      <a href="/download/zip/{{ sym }}" class="dl-card">
        <span class="icon">&#8659;</span> {{ sym }}
      </a>
      {% endfor %}
    </div>
  </div>

  {% for category, symbols in tree.items() %}
  <div class="category-block">
    <div class="category-header">
      <h2>{{ category | capitalize }} Data</h2>
      <span class="cat-badge {{ category }}">{{ category }}</span>
    </div>

    {% for symbol, content in symbols.items() %}
    <div class="symbol-card">
      <div class="symbol-card-header">
        <span class="symbol-name">{{ symbol }}</span>
      </div>
      <div class="symbol-card-body">

        {% if category == 'live' %}
          {% for dtype, timeframes in content.items() %}
          <div class="dtype-block">
            <div class="dtype-label">{{ dtype }}</div>
            <table>
              <thead>
                <tr>
                  <th>Timeframe</th><th>Candles</th><th>Start</th><th>End</th>
                  <th>Size</th><th>Last Updated</th><th>Download</th>
                </tr>
              </thead>
              <tbody>
                {% for tf, meta in timeframes.items() %}
                <tr>
                  <td><span class="tf-badge">{{ tf }}</span></td>
                  <td><span class="rows-count">{{ "{:,}".format(meta.rows) }}</span></td>
                  <td><span class="date-range">{{ meta.start }}</span></td>
                  <td><span class="date-range">{{ meta.end }}</span></td>
                  <td><span class="size-tag">{{ meta.size_kb }} KB</span></td>
                  <td><span class="date-range">{{ meta.updated }}</span></td>
                  <td><a href="/download/{{ symbol }}/{{ dtype }}/{{ tf }}" class="btn-dl">&#8595; CSV</a></td>
                </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
          {% endfor %}

        {% else %}
          <table>
            <thead>
              <tr>
                <th>File</th><th>Candles</th><th>Start</th><th>End</th>
                <th>Size</th><th>Last Updated</th><th>Download</th>
              </tr>
            </thead>
            <tbody>
              {% for tf, meta in content.items() %}
              <tr>
                <td><span class="tf-badge">{{ tf }}</span></td>
                <td><span class="rows-count">{{ "{:,}".format(meta.rows) }}</span></td>
                <td><span class="date-range">{{ meta.start }}</span></td>
                <td><span class="date-range">{{ meta.end }}</span></td>
                <td><span class="size-tag">{{ meta.size_kb }} KB</span></td>
                <td><span class="date-range">{{ meta.updated }}</span></td>
                <td><a href="/download/{{ symbol }}/historical/{{ tf }}" class="btn-dl">&#8595; CSV</a></td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        {% endif %}

      </div>
    </div>
    {% endfor %}
  </div>
  {% endfor %}

</main>

<footer>
  PulseVault &mdash; Futures Market Data Hub &bull; NQ1! &bull; ES1!
</footer>

<script>
function triggerCollect() {
  const btn = document.getElementById('collectBtn');
  const msg = document.getElementById('collect-msg');
  btn.disabled = true; btn.textContent = 'Collecting...'; msg.textContent = '';
  clearTimeout(window._autoRefresh);
  fetch('/collect', {method:'POST'}).then(r=>r.json()).then(d=>{
    if (d.status === 'started') { msg.textContent = 'Running...'; pollCollect(); }
    else { msg.textContent = 'Already running'; btn.disabled = false; btn.textContent = 'Collect Now'; }
  });
}
function pollCollect() {
  fetch('/collect/status').then(r=>r.json()).then(d=>{
    if (d.running) { setTimeout(pollCollect, 2000); return; }
    const btn = document.getElementById('collectBtn'), msg = document.getElementById('collect-msg');
    btn.disabled = false; btn.textContent = 'Collect Now';
    msg.textContent = d.last_error ? 'Error: ' + d.last_error : 'Done — ' + (d.last_run || '');
    setTimeout(() => location.reload(), 1500);
  });
}
window._autoRefresh = setTimeout(() => location.reload(), 60000);
</script>
</body>
</html>
"""


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    tree = collect_file_tree()
    return render_template_string(INDEX_TEMPLATE, tree=tree)


@app.route("/download/<symbol>/<dtype>/<tf>")
def download_file(symbol, dtype, tf):
    if dtype == "historical":
        path = os.path.join(DATA_DIR, "historical", symbol, f"{tf}.csv")
    else:
        path = os.path.join(DATA_DIR, "live", symbol, dtype, f"{tf}.csv")

    if not os.path.exists(path):
        abort(404)

    filename = f"{symbol}_{dtype}_{tf}.csv".replace("!", "")
    return send_file(path, as_attachment=True, download_name=filename, mimetype="text/csv")


@app.route("/download/zip/<symbol>")
def download_zip_symbol(symbol):
    buf = io.BytesIO()
    found = False
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(DATA_DIR):
            for fname in files:
                if not fname.endswith(".csv"):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, DATA_DIR)
                if symbol in rel or symbol.replace("!", "") in rel:
                    zf.write(full, rel)
                    found = True
    if not found:
        abort(404)
    buf.seek(0)
    zipname = f"{symbol}_data.zip".replace("!", "")
    return send_file(buf, as_attachment=True, download_name=zipname, mimetype="application/zip")


@app.route("/download/zip/all")
def download_zip_all():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(DATA_DIR):
            for fname in files:
                if fname.endswith(".csv"):
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, DATA_DIR)
                    zf.write(full, rel)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="market_data_all.zip",
                     mimetype="application/zip")


@app.route("/status")
def status():
    tree = collect_file_tree()
    summary = {}
    for cat, symbols in tree.items():
        summary[cat] = {}
        for sym, content in symbols.items():
            summary[cat][sym] = {}
            if cat == "live":
                for dtype, tfs in content.items():
                    for tf, meta in tfs.items():
                        summary[cat][sym][f"{dtype}/{tf}"] = {
                            "rows": meta["rows"], "updated": meta["updated"]
                        }
            else:
                for tf, meta in content.items():
                    summary[cat][sym][tf] = {"rows": meta["rows"], "updated": meta["updated"]}
    return jsonify(summary)


_collect_lock = threading.Lock()
_collect_status = {"running": False, "last_run": None, "last_error": None}

@app.route("/collect", methods=["POST"])
def trigger_collect():
    if _collect_lock.locked():
        return jsonify({"status": "already_running"}), 409

    def _run():
        _collect_status["running"] = True
        _collect_status["last_error"] = None
        try:
            for tf in TIMEFRAME_MAP:
                collect_all(tf, n_bars=50)
            _collect_status["last_run"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception as e:
            _collect_status["last_error"] = str(e)
        finally:
            _collect_status["running"] = False
            _collect_lock.release()

    _collect_lock.acquire()
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/collect/status")
def collect_status_route():
    return jsonify(_collect_status)


if __name__ == "__main__":
    print("Starting web server at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
