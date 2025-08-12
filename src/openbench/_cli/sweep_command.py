from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import html
import importlib
import io
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from inspect_ai import eval as inspect_eval  # type: ignore
from openbench.config import load_task
from openbench.monkeypatch.display_results_patch import patch_display_results
from openbench.monkeypatch.file_recorder_logfile_patch import patch_file_recorder_logfile

console = Console()


# -----------------------------
# Helpers: basic
# -----------------------------
def _slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _parse_limit(value: Optional[str]) -> Optional[int | tuple[int, int]]:
    if value is None:
        return None
    if "," in value:
        a, b = value.split(",", 1)
        return (int(a.strip()), int(b.strip()))
    return int(value.strip())


def _ensure_matplotlib():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot  # noqa: F401
        return True
    except Exception:
        return False


class _Tee:
    """Write to multiple file-like streams at once (stdout/stderr tee),
    exposing enough of the file API for Rich/typer to detect terminal capabilities.
    """
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        for s in self._streams:
            try:
                if hasattr(s, "isatty") and callable(s.isatty) and s.isatty():
                    return True
            except Exception:
                continue
        return False

    @property
    def encoding(self) -> str:
        for s in self._streams:
            enc = getattr(s, "encoding", None)
            if enc:
                return enc
        return "utf-8"

    @property
    def errors(self) -> str:
        for s in self._streams:
            err = getattr(s, "errors", None)
            if err:
                return err
        return "strict"

    def fileno(self) -> int:
        for s in self._streams:
            try:
                if hasattr(s, "fileno"):
                    return s.fileno()
            except Exception:
                continue
        raise io.UnsupportedOperation("fileno")

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    # Try YAML first if available, else JSON
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        try:
            return json.loads(text)
        except Exception as e:
            raise typer.BadParameter(
                f"Failed to parse config '{path}'. Install 'pyyaml' for YAML or provide valid JSON. Root error: {e}"
            )


def _parse_accuracy_from_text(text: str) -> Optional[float]:
    matches = re.findall(r"accuracy\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r'["\']accuracy["\']\s*:\s*([0-9]*\.?[0-9]+)', text, flags=re.IGNORECASE)
    if matches:
        try:
            return float(matches[-1])
        except Exception:
            return None
    return None


def _write_stdout_capture(log_dir: Path, logfile_key: str, out_text: str, err_text: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / f"{logfile_key}.stdout.txt"
    try:
        p.write_text(out_text + ("\n--- STDERR ---\n" + err_text if err_text else ""), encoding="utf-8")
    except Exception:
        pass
    return p


def _extract_accuracy_from_jsonl(path: Path) -> Optional[float]:
    if not path.exists():
        return None

    last_acc: Optional[float] = None
    total = 0
    correct = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            metrics = obj.get("metrics") if isinstance(obj, dict) else None
            if isinstance(metrics, dict):
                if "accuracy" in metrics:
                    a = metrics["accuracy"]
                    if isinstance(a, (int, float)):
                        last_acc = float(a)
                    elif isinstance(a, dict):
                        for k in ("value", "mean", "score"):
                            v = a.get(k)
                            if isinstance(v, (int, float)):
                                last_acc = float(v)
                                break
                else:
                    for k, v in metrics.items():
                        if "accuracy" in k and isinstance(v, (int, float)):
                            last_acc = float(v)
                        elif "accuracy" in k and isinstance(v, dict):
                            for kk in ("value", "mean", "score"):
                                vv = v.get(kk)
                                if isinstance(vv, (int, float)):
                                    last_acc = float(vv)
                                    break

            score_obj = obj.get("score") if isinstance(obj, dict) else None
            if isinstance(score_obj, dict):
                val = score_obj.get("value")
                if isinstance(val, (int, float)):
                    total += 1
                    if float(val) >= 1.0:
                        correct += 1

    if last_acc is not None:
        return float(last_acc)
    if total > 0:
        return correct / total
    return None


def _extract_accuracy_from_eval(eval_path: Path) -> Optional[float]:
    if not eval_path.exists():
        return None

    try:
        raw = eval_path.read_text(encoding="utf-8")
    except Exception:
        return None

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            if isinstance(obj.get("accuracy"), (int, float)):
                return float(obj["accuracy"])
            metrics = obj.get("metrics")
            if isinstance(metrics, dict):
                a = metrics.get("accuracy")
                if isinstance(a, (int, float)):
                    return float(a)
                if isinstance(a, dict):
                    for k in ("value", "mean", "score"):
                        v = a.get(k)
                        if isinstance(v, (int, float)):
                            return float(v)
                for k, v in metrics.items():
                    if "accuracy" in k and isinstance(v, (int, float)):
                        return float(v)
                    if "accuracy" in k and isinstance(v, dict):
                        for kk in ("value", "mean", "score"):
                            vv = v.get(kk)
                            if isinstance(vv, (int, float)):
                                return float(vv)
    except Exception:
        pass

    try:
        acc = _extract_accuracy_from_jsonl(eval_path)
        if isinstance(acc, (int, float)):
            return float(acc)
    except Exception:
        pass

    return _parse_accuracy_from_text(raw)


def _extract_accuracy_any(log_dir: Path, logfile_key: str) -> tuple[Optional[float], Optional[Path]]:
    candidates = [
        log_dir / f"{logfile_key}.jsonl",
        log_dir / f"{logfile_key}.log.jsonl",
        log_dir / f"{logfile_key}.eval",
    ]
    for p in candidates:
        if not p.exists():
            continue
        acc = _extract_accuracy_from_eval(p) if p.suffix == ".eval" else _extract_accuracy_from_jsonl(p)
        if isinstance(acc, (int, float)):
            return float(acc), p
    return None, None


# -----------------------------
# Snapshot + Asset helpers
# -----------------------------
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_path_like(s: str) -> bool:
    s = os.path.expanduser(s)
    return (
        s.endswith(".py")
        or s.startswith(".")
        or s.startswith("/")
        or os.path.sep in s
        or os.path.isfile(s)
    )


def _resolve_pyfunc_source(spec: str) -> Optional[Path]:
    """
    Given 'module_or_file:function', return the absolute path to the Python source file.
    """
    if ":" not in spec:
        return None
    module_or_path, _func = spec.split(":", 1)
    module_or_path = os.path.expanduser(module_or_path)
    if _is_path_like(module_or_path):
        return Path(module_or_path).resolve()
    # Treat as module
    try:
        mod = importlib.import_module(module_or_path)
        file = getattr(mod, "__file__", None)
        if file:
            return Path(file).resolve()
    except Exception:
        return None
    return None


def _sanitize_for_store(p: Path) -> str:
    parts = list(p.parts)
    clean = "__".join([re.sub(r"[^A-Za-z0-9_.-]", "_", s) for s in parts])
    return clean


def _symlink_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dest.exists() or dest.is_symlink():
            dest.unlink()
    except Exception:
        pass
    try:
        os.symlink(src, dest)
    except Exception:
        try:
            shutil.copy2(src, dest)
        except Exception:
            try:
                data = src.read_bytes()
                dest.write_bytes(data)
            except Exception:
                pass


def _ensure_assets(sweeps_base: Path) -> None:
    """Write (or refresh) shared CSS/JS assets into logs/sweeps/assets."""
    assets = sweeps_base / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    style_css = """
:root{
  --bg:#0b0d10;--card:#111418;--muted:#7a8699;--text:#e6edf3;--border:#1b2026;--accent:#6ea8fe;--accent-2:#7ee787;--warn:#ffc861
}
html[data-theme='light']{
  --bg:#f8fafc;--card:#ffffff;--muted:#4b5563;--text:#0f172a;--border:#e2e8f0;--accent:#2563eb;--accent-2:#16a34a;--warn:#d97706
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;background:var(--bg);color:var(--text);
  font:14.5px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Helvetica,Arial,"Apple Color Emoji","Segoe UI Emoji";
}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.container{max-width:1100px;margin:0 auto;padding:24px}
.header{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:18px}
.breadcrumb{color:var(--muted);font-weight:600;letter-spacing:.2px}
.h1{font-size:26px;font-weight:750;margin:6px 0 2px}
.meta{color:var(--muted);font-size:13px}
.controls{
  display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 10px
}
.input, .select, .btn{
  border:1px solid var(--border);background:var(--card);color:var(--text);
  padding:8px 10px;border-radius:8px;outline:none
}
.input:focus, .select:focus{border-color:var(--accent)}
.btn{cursor:pointer;font-weight:600}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.ghost{background:transparent}
.btn[data-role="theme"]{display:flex;align-items:center;gap:8px}
.grid{
  display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0 18px
}
@media (max-width: 940px){.grid{grid-template-columns:repeat(2,1fr)}}
@media (max-width: 520px){.grid{grid-template-columns:1fr}}
.card{
  background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px
}
.kpi .label{color:var(--muted);font-size:12px;margin-bottom:6px}
.kpi .value{font-size:22px;font-weight:800}
.kpi .sub{color:var(--muted);font-size:12px;margin-top:4px}
.table-wrap{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:auto}
table{width:100%;border-collapse:separate;border-spacing:0;min-width:720px}
thead th{
  position:sticky;top:0;background:var(--card);z-index:2;
  font-weight:700;font-size:13px;color:var(--muted);text-align:left;padding:10px 12px;border-bottom:1px solid var(--border)
}
tbody td{
  padding:10px 12px;border-bottom:1px solid var(--border);vertical-align:top
}
tbody tr:hover{background:rgba(110,168,254,.06)}
th.sortable{cursor:pointer}
th.sortable .arrow{opacity:.45;margin-left:6px;transition:transform .2s}
th.sortable.asc .arrow{transform:rotate(180deg);opacity:.9}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid var(--border);background:transparent;color:var(--muted);font-size:12px}
pre{white-space:pre-wrap;word-wrap:break-word}
.code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
.details{margin:10px 0}
.details summary{cursor:pointer;font-weight:700}
footer{color:var(--muted);font-size:12px;text-align:center;margin:20px 0}
hr.sep{border:0;border-top:1px solid var(--border);margin:18px 0}
.img{max-width:100%;height:auto;border:1px solid var(--border);border-radius:10px}
.path{color:var(--muted)}
.help{color:var(--muted);font-size:12px}
"""
    app_js = """
(function(){
  // theme toggle with localStorage
  const root = document.documentElement;
  const saved = localStorage.getItem('ob-theme');
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  if(!root.getAttribute('data-theme')){
    root.setAttribute('data-theme', saved || (prefersDark ? 'dark' : 'light'));
  }
  function toggleTheme(){
    const cur = root.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('ob-theme', next);
    const label = document.querySelector('[data-role="theme-label"]');
    if(label) label.textContent = next === 'dark' ? 'Dark' : 'Light';
  }
  const themeBtn = document.querySelector('[data-role="theme"]');
  if(themeBtn){ themeBtn.addEventListener('click', toggleTheme); }

  // table sorting
  function sortTable(table, idx, numeric){
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const th = table.tHead.rows[0].cells[idx];
    const asc = !th.classList.contains('asc');
    for(const cell of table.tHead.rows[0].cells){ cell.classList.remove('asc'); }
    th.classList.toggle('asc', asc);
    rows.sort((a,b)=>{
      const ta = a.cells[idx].innerText.trim();
      const tb = b.cells[idx].innerText.trim();
      if(numeric){
        const va = parseFloat(ta.replace('%','')) || -Infinity;
        const vb = parseFloat(tb.replace('%','')) || -Infinity;
        return asc ? (va - vb) : (vb - va);
      }else{
        return asc ? ta.localeCompare(tb) : tb.localeCompare(ta);
      }
    });
    rows.forEach(r=>tbody.appendChild(r));
  }
  document.querySelectorAll('table[data-sortable] thead th').forEach((th, i)=>{
    th.classList.add('sortable');
    const arrow = document.createElement('span');
    arrow.className = 'arrow';
    arrow.textContent = '▲';
    th.appendChild(arrow);
    th.addEventListener('click', ()=>{
      const table = th.closest('table');
      const numeric = th.getAttribute('data-type') === 'num';
      sortTable(table, i, numeric);
    });
  });

  // filtering/search
  const search = document.querySelector('[data-role="search"]');
  const benchSel = document.querySelector('[data-role="bench-filter"]');
  function applyFilter(){
    const q = (search?.value || '').toLowerCase();
    const bench = benchSel?.value || '';
    document.querySelectorAll('[data-row="result"]').forEach(tr=>{
      const rowBench = tr.getAttribute('data-bench') || '';
      const txt = tr.innerText.toLowerCase();
      const okBench = !bench || bench === rowBench;
      const okSearch = !q || txt.includes(q);
      tr.style.display = (okBench && okSearch) ? '' : 'none';
    });
  }
  if(search){ search.addEventListener('input', applyFilter); }
  if(benchSel){ benchSel.addEventListener('change', applyFilter); }

  // copy to clipboard buttons
  document.querySelectorAll('[data-copy]').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      const targetSel = btn.getAttribute('data-copy');
      const el = document.querySelector(targetSel);
      const text = el?.innerText || el?.textContent || '';
      if(!text){ return; }
      navigator.clipboard.writeText(text).then(()=>{
        const old = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(()=>btn.textContent = old, 900);
      });
    });
  });
})();
"""

    (assets / "style.css").write_text(style_css.strip() + "\n", encoding="utf-8")
    (assets / "app.js").write_text(app_js.strip() + "\n", encoding="utf-8")


@dataclass
class RunSpec:
    label: str
    model: str
    model_args: Dict[str, Any]


@dataclass
class BenchSpec:
    name: str
    params: Dict[str, Any]


@dataclass
class SharedSpec:
    epochs: int = 1
    limit: Optional[str] = None
    temperature: float = 0.6
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    timeout: int = 10000
    log_dir: str = "./logs"


def _normalize_runs(raw_runs: List[Dict[str, Any]]) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for r in raw_runs:
        label = r.get("label")
        model = r.get("model")
        if not label or not model:
            raise typer.BadParameter("Each run must include 'label' and 'model'.")
        model_args = dict(r.get("model_args", {}))
        if "model" in model_args and "inner_model" not in model_args:
            model_args["inner_model"] = model_args.pop("model")
        runs.append(RunSpec(label=label, model=model, model_args=model_args))
    return runs


def _normalize_benchmarks(raw_benchmarks: List[Dict[str, Any]]) -> List[BenchSpec]:
    specs: List[BenchSpec] = []
    for b in raw_benchmarks:
        name = b.get("name")
        if not name:
            raise typer.BadParameter("Each benchmark must include 'name'.")
        params = dict(b.get("params", {}))
        specs.append(BenchSpec(name=name, params=params))
    return specs


def _normalize_shared(raw_shared: Dict[str, Any], cfg_log_dir: Optional[str]) -> SharedSpec:
    s = SharedSpec()
    for k in ("epochs", "limit", "temperature", "top_p", "max_tokens", "timeout", "log_dir"):
        if k in raw_shared and raw_shared[k] is not None:
            setattr(s, k, raw_shared[k])
    if cfg_log_dir:
        s.log_dir = cfg_log_dir
    return s


def _snapshot_inputs(
    sweep_dir: Path,
    config_path: Path,
    runs: List[RunSpec],
) -> Dict[str, Any]:
    """
    Save an immutable copy of the sweep config and all pyfunc model files
    as they existed at the START of the sweep.
    """
    inputs_dir = sweep_dir / "inputs"
    models_dir = inputs_dir / "models"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1) Snapshot config file
    cfg_bytes = config_path.read_bytes()
    cfg_hash = _sha256_bytes(cfg_bytes)
    cfg_ext = config_path.suffix or ".yaml"
    cfg_store = inputs_dir / f"sweep_config{cfg_ext}"
    cfg_store.write_bytes(cfg_bytes)

    # 2) Collect unique pyfunc model source files
    model_paths: Dict[str, Path] = {}
    for r in runs:
        if not r.model.startswith("pyfunc/"):
            continue
        spec = r.model.split("pyfunc/", 1)[1]
        src = _resolve_pyfunc_source(spec)
        if src and src.exists():
            model_paths[str(src)] = src

    # 3) Snapshot each model file
    model_entries: List[Dict[str, Any]] = []
    for orig, path in sorted(model_paths.items()):
        data = path.read_bytes()
        h = _sha256_bytes(data)
        sanitized = _sanitize_for_store(path)
        dest = models_dir / sanitized
        dest.write_bytes(data)
        model_entries.append(
            {
                "original_path": str(path),
                "stored_relpath": str(dest.relative_to(sweep_dir)),
                "sha256": h,
            }
        )

    # 4) Write manifest
    manifest = {
        "config": {
            "original_path": str(config_path.resolve()),
            "stored_relpath": str(cfg_store.relative_to(sweep_dir)),
            "sha256": cfg_hash,
        },
        "models": model_entries,
    }
    (inputs_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# -----------------------------
# HTML writers
# -----------------------------
def _best_avg(results: Dict[str, Dict[str, Optional[float]]]) -> tuple[Optional[float], Optional[float]]:
    values: List[float] = []
    for bench, runs_map in results.items():
        for _, v in runs_map.items():
            if isinstance(v, (int, float)):
                values.append(float(v))
    if not values:
        return None, None
    return (max(values), sum(values) / len(values))


def _write_sweep_index_html(
    sweeps_base: Path,
    sweep_dir: Path,
    name: str,
    sweep_tag: str,
    created_at: str,
    benchmarks: List[BenchSpec],
    runs: List[RunSpec],
    results: Dict[str, Dict[str, Optional[float]]],
    log_index: Dict[tuple[str, str], Optional[Path]],
    manifest: Dict[str, Any],
    plot_rel: Optional[str],
) -> None:
    """
    Render a modern, dependency-free HTML page with results, plot, and saved files.
    """
    assets_rel = "../assets"  # per-sweep page is one level deep
    best, avg = _best_avg(results)
    n_bench = len(benchmarks)
    n_cells = n_bench * len(runs)

    # Build options for benchmark filter
    bench_options = "".join(f'<option value="{html.escape(b.name)}">{html.escape(b.name)}</option>' for b in benchmarks)

    # Build table rows
    rows_html = []
    for b in benchmarks:
        for r in runs:
            acc = results.get(b.name, {}).get(r.label)
            acc_str = f"{acc*100:.2f}%" if isinstance(acc, (int, float)) else "—"
            # Prefer the collected link under sweep_dir/logs/
            logp_src = log_index.get((b.name, r.label))
            log_dest = (sweep_dir / "logs" / logp_src.name) if logp_src else None  # type: ignore
            log_link = (
                f'<a href="logs/{html.escape(log_dest.name)}" target="_blank">{html.escape(log_dest.name)}</a>'
                if isinstance(log_dest, Path) and log_dest.exists() else "—"
            )
            rows_html.append(
                f'<tr data-row="result" data-bench="{html.escape(b.name)}">'
                f"<td><span class='badge'>{html.escape(b.name)}</span></td>"
                f"<td>{html.escape(r.label)}</td>"
                f"<td class='code'>{html.escape(r.model)}</td>"
                f"<td>{acc_str}</td>"
                f"<td>{log_link}</td>"
                "</tr>"
            )

    # Inputs (config + models)
    config_rel = manifest["config"]["stored_relpath"]
    config_text = (sweep_dir / config_rel).read_text(encoding="utf-8")
    config_hash = manifest["config"].get("sha256", "")
    config_block = f"""
      <details class="details" open>
        <summary>Config <code class="code">{html.escape(config_rel)}</code> <span class="badge">{html.escape(config_hash)}</span></summary>
        <pre class="code">{html.escape(config_text)}</pre>
      </details>
    """

    model_sections = []
    for m in manifest.get("models", []):
        m_rel = m["stored_relpath"]
        m_text = (sweep_dir / m_rel).read_text(encoding="utf-8")
        model_sections.append(
            f"""
            <details class="details">
              <summary>Model file <code class="code">{html.escape(m_rel)}</code> <span class="badge">{html.escape(m.get('sha256',''))}</span></summary>
              <pre class="code">{html.escape(m_text)}</pre>
            </details>
            """
        )
    models_block = "\n".join(model_sections) if model_sections else "<div class='help'>No pyfunc model files in this sweep.</div>"

    # KPI helpers
    def _k(v: Optional[float]) -> str:
        return f"{v*100:.2f}%" if isinstance(v, (int, float)) else "—"

    plot_tag = f'<img class="img" src="{html.escape(plot_rel)}" alt="Accuracy Plot" />' if plot_rel else "<div class='help'>No plot available.</div>"

    html_text = f"""<!doctype html>
<html data-theme="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(name)} · {html.escape(sweep_tag)} · OpenBench</title>
  <link rel="stylesheet" href="{assets_rel}/style.css" />
  <script defer src="{assets_rel}/app.js"></script>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <div class="breadcrumb"><a href="../index.html">Sweeps</a> / <span>{html.escape(sweep_tag)}</span></div>
        <div class="h1">{html.escape(name)}</div>
        <div class="meta">Created: {html.escape(created_at)} · Folder: <span class="path">{html.escape(str(sweep_dir.resolve()))}</span></div>
      </div>
      <div>
        <button class="btn ghost" data-role="theme"><span>🌓</span> <span data-role="theme-label">Dark</span></button>
      </div>
    </div>

    <div class="grid">
      <div class="card kpi">
        <div class="label">Best Accuracy</div>
        <div class="value">{_k(best)}</div>
        <div class="sub">Across all runs</div>
      </div>
      <div class="card kpi">
        <div class="label">Average Accuracy</div>
        <div class="value">{_k(avg)}</div>
        <div class="sub">Mean of scored cells</div>
      </div>
      <div class="card kpi">
        <div class="label">Runs × Benchmarks</div>
        <div class="value">{len(runs)} × {n_bench}</div>
        <div class="sub">{n_cells} result cells</div>
      </div>
      <div class="card kpi">
        <div class="label">Artifacts</div>
        <div class="value">{'1 plot' if plot_rel else 'No plot'}</div>
        <div class="sub"><a href="summary.json" download>Download summary.json</a></div>
      </div>
    </div>

    <div class="controls">
      <input class="input" type="search" placeholder="Search…" data-role="search" />
      <select class="select" data-role="bench-filter">
        <option value="">All benchmarks</option>
        {bench_options}
      </select>
      <a class="btn" href="summary.json" download>⬇ Download JSON</a>
      <a class="btn" href="../index.html">← Back to all sweeps</a>
    </div>

    <div class="table-wrap card">
      <table data-sortable>
        <thead>
          <tr>
            <th>Benchmark</th>
            <th>Run label</th>
            <th>Model</th>
            <th data-type="num">Accuracy</th>
            <th>Log</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>

    <div class="grid">
      <div class="card">
        <h3 style="margin:4px 0 12px;">Plot</h3>
        {plot_tag}
      </div>
      <div class="card">
        <h3 style="margin:4px 0 12px;">Inputs</h3>
        {config_block}
        {models_block}
      </div>
    </div>

    <footer>OpenBench · static sweep report</footer>
  </div>
</body>
</html>
"""
    (sweep_dir / "index.html").write_text(html_text, encoding="utf-8")


def _write_sweeps_root_index(base_dir: Path) -> None:
    """
    Generate ./logs/sweeps/index.html that lists all sweeps with links.
    """
    assets_rel = "./assets"
    rows = []
    # Read all summary.json files
    for d in sorted([p for p in base_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        summ = d / "summary.json"
        if not summ.exists():
            continue
        try:
            js = json.loads(summ.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = js.get("name") or d.name
        created = js.get("created_at", "")
        rows.append((name, d.name, created))

    body_rows = []
    for (name, folder, created) in rows:
        body_rows.append(
            f"<tr data-row='sweep'>"
            f"<td><a href='{html.escape(folder)}/index.html'>{html.escape(name)}</a></td>"
            f"<td class='code'>{html.escape(folder)}</td>"
            f"<td>{html.escape(created or '')}</td>"
            f"</tr>"
        )

    html_text = f"""<!doctype html>
<html data-theme="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenBench Sweeps</title>
  <link rel="stylesheet" href="{assets_rel}/style.css" />
  <script defer src="{assets_rel}/app.js"></script>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <div class="breadcrumb">Sweeps</div>
        <div class="h1">All Sweeps</div>
        <div class="meta">Directory: <span class="path">{html.escape(str(base_dir.resolve()))}</span></div>
      </div>
      <div>
        <button class="btn ghost" data-role="theme"><span>🌓</span> <span data-role="theme-label">Dark</span></button>
      </div>
    </div>

    <div class="controls">
      <input class="input" type="search" placeholder="Search by name or folder…" data-role="search" />
      <a class="btn" href="#" onclick="location.reload();return false;">↻ Refresh</a>
      <span class="help">Tip: Click a sweep to view its full report.</span>
    </div>

    <div class="table-wrap card">
      <table data-sortable>
        <thead>
          <tr>
            <th>Name</th>
            <th>Folder</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {''.join(body_rows) if body_rows else '<tr><td colspan="3"><em>No sweeps found yet.</em></td></tr>'}
        </tbody>
      </table>
    </div>

    <footer>OpenBench · static sweeps index</footer>
  </div>
</body>
</html>
"""
    (base_dir / "index.html").write_text(html_text, encoding="utf-8")


# -----------------------------
# Typer command
# -----------------------------
def run_sweep(
    config: Path = typer.Argument(..., help="Path to YAML/JSON sweep config"),
    plot: bool = typer.Option(True, "--plot/--no-plot", help="Plot a bar chart of accuracies"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse config and print the plan without running"),
) -> None:
    """
    Run a batch of benchmark configs and (optionally) plot the results.
    Also snapshots the exact inputs (config + pyfunc model files) at the START of the sweep,
    and writes a polished static site for easy sharing/viewing.
    """
    console.print()
    console.print(Panel("OpenBench Sweep", expand=False, box=ROUNDED))
    console.print()

    cfg = _load_config(config)
    name: str = cfg.get("name") or config.stem
    sweep_tag = _slug(name) + "-" + _now_tag()

    cfg_log_dir = cfg.get("log_dir")
    shared = _normalize_shared(cfg.get("shared", {}), cfg_log_dir)
    log_dir = Path(shared.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = _normalize_benchmarks(cfg.get("benchmarks", []))
    runs = _normalize_runs(cfg.get("runs", []))
    if not benchmarks:
        raise typer.BadParameter("Config must include at least one entry under 'benchmarks'.")
    if not runs:
        raise typer.BadParameter("Config must include at least one entry under 'runs'.")

    # Create sweep folder under logs/sweeps/<sweep_tag>
    sweeps_base = log_dir / "sweeps"
    sweep_dir = sweeps_base / sweep_tag
    (sweep_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Ensure shared CSS/JS assets exist
    _ensure_assets(sweeps_base)

    # Snapshot inputs at the START
    manifest = _snapshot_inputs(sweep_dir, config.resolve(), runs)
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[green]Snapshot saved[/green]: {sweep_dir}")

    # Preview plan
    table = Table(title="Sweep Plan", show_lines=False, box=None)
    table.add_column("Benchmark", style="cyan")
    table.add_column("Run Label", style="white")
    table.add_column("Model", style="green")
    table.add_column("Model Args", style="dim")
    for b in benchmarks:
        for r in runs:
            table.add_row(b.name, r.label, r.model, json.dumps(r.model_args))
    console.print(table)
    console.print()

    if dry_run:
        console.print("[yellow]Dry run complete. Nothing executed.[/yellow]")
        return

    # Patch display so interruptions suggest 'bench eval-retry'
    patch_display_results()

    # Results holder
    results: Dict[str, Dict[str, Optional[float]]] = {b.name: {} for b in benchmarks}
    log_index: Dict[tuple[str, str], Optional[Path]] = {}

    # Execute
    for b in benchmarks:
        console.rule(f"[bold green]{b.name}")
        for r in runs:
            sub_tag = f"{_slug(b.name)}-{_slug(r.label)}-{_now_tag()}"
            logfile_key = f"sweep-{sweep_tag}-{sub_tag}"
            patch_file_recorder_logfile(logfile_key)

            console.print(
                f"[bold]Running[/bold] bench=[cyan]{b.name}[/cyan] label=[white]{r.label}[/white] "
                f"model=[green]{r.model}[/green] → key=[dim]{logfile_key}[/dim]"
            )
            log_index[(b.name, r.label)] = None

            try:
                task_fn = load_task(b.name)
            except Exception as e:
                console.print(f"[red]Failed to load task '{b.name}': {e}[/red]")
                results[b.name][r.label] = None
                continue

            captured_out = ""
            captured_err = ""
            try:
                buf_out, buf_err = io.StringIO(), io.StringIO()
                tee_out = _Tee(sys.stdout, buf_out)
                tee_err = _Tee(sys.stderr, buf_err)
                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    inspect_eval(
                        tasks=[task_fn],
                        model=[r.model],
                        model_args=r.model_args,
                        epochs=shared.epochs,
                        limit=_parse_limit(shared.limit),
                        score=True,
                        temperature=shared.temperature,
                        top_p=shared.top_p,
                        max_tokens=shared.max_tokens,
                        timeout=shared.timeout,
                    )
                captured_out = buf_out.getvalue()
                captured_err = buf_err.getvalue()
            except Exception as e:
                try:
                    captured_out = captured_out or buf_out.getvalue()
                    captured_err = captured_err or buf_err.getvalue()
                except Exception:
                    pass
                console.print(f"[red]Eval failed:[/red] {e}")
                text_acc = _parse_accuracy_from_text((captured_out or "") + "\n" + (captured_err or ""))
                if isinstance(text_acc, (int, float)):
                    results[b.name][r.label] = float(text_acc)
                    p = _write_stdout_capture(log_dir, logfile_key, captured_out, captured_err)
                    log_index[(b.name, r.label)] = p
                    console.print(f"[yellow]Recovered accuracy from terminal output.[/yellow]")
                    console.print(f"Log (stdout): [dim]{p}[/dim]")
                else:
                    results[b.name][r.label] = None
                continue

            # Parse accuracy from any log file we can find
            acc, used_path = _extract_accuracy_any(log_dir, logfile_key)
            if used_path:
                log_index[(b.name, r.label)] = used_path
                console.print(f"Log: [dim]{used_path}[/dim]")

            # If not found, parse from terminal output and save it
            if acc is None:
                text_acc = _parse_accuracy_from_text(captured_out + "\n" + captured_err)
                if isinstance(text_acc, (int, float)):
                    acc = float(text_acc)
                    p = _write_stdout_capture(log_dir, logfile_key, captured_out, captured_err)
                    log_index[(b.name, r.label)] = p
                    console.print(f"Log (stdout): [dim]{p}[/dim]")
                    console.print("[green]Parsed accuracy from terminal output.[/green]")
                else:
                    console.print(
                        f"[yellow]No accuracy found for key '{logfile_key}'. "
                        f"Looked for .jsonl/.log.jsonl/.eval files and terminal output.[/yellow]"
                    )

            results[b.name][r.label] = acc
            if acc is not None:
                console.print(f"[bold]Accuracy[/bold]: {acc*100:.2f}%")

        console.print()

    # Collect logs into sweep_dir/logs (symlink or copy)
    for (bname, rlabel), p in log_index.items():
        if not p:
            continue
        dest = sweep_dir / "logs" / p.name
        _symlink_or_copy(p, dest)

    # Summarize table
    console.rule("[bold blue]Summary")
    for b in benchmarks:
        t = Table(title=b.name, show_lines=False, box=None)
        t.add_column("Run", style="white")
        t.add_column("Accuracy", style="magenta")
        t.add_column("Log", style="dim")
        for r in runs:
            acc = results[b.name].get(r.label)
            acc_str = f"{acc*100:.2f}%" if isinstance(acc, (int, float)) else "—"
            log_path = (sweep_dir / "logs" / (log_index.get((b.name, r.label)).name)) if log_index.get((b.name, r.label)) else None  # type: ignore
            t.add_row(r.label, acc_str, str(log_path) if log_path else "")
        console.print(t)
        console.print()

    # Plot inside the sweep folder
    plot_rel = None
    if plot:
        if not _ensure_matplotlib():
            console.print(
                "[yellow]matplotlib not installed; skipping plot. Install with 'pip install matplotlib' or 'uv pip install matplotlib'.[/yellow]"
            )
        else:
            import matplotlib.pyplot as plt  # type: ignore
            out_png = sweep_dir / "plot.png"
            fig, ax = plt.subplots(figsize=(max(6, len(runs) * 1.2), 4.5))

            if len(benchmarks) == 1:
                b = benchmarks[0]
                labels = [r.label for r in runs]
                vals = [results[b.name].get(r.label) or 0.0 for r in runs]
                ax.bar(labels, [v * 100 for v in vals], color="#4C78A8")
                for i, v in enumerate(vals):
                    ax.text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(f"{b.name} – {name}")
                plt.xticks(rotation=20, ha="right")
                plt.tight_layout()
                plt.savefig(out_png, dpi=200)
            else:
                import numpy as np  # type: ignore
                x = np.arange(len(runs))
                width = 0.8 / len(benchmarks)
                for idx, b in enumerate(benchmarks):
                    vals = [results[b.name].get(r.label) or 0.0 for r in runs]
                    ax.bar(x + idx * width, [v * 100 for v in vals], width, label=b.name)
                ax.set_xticks(x + width * (len(benchmarks) - 1) / 2)
                ax.set_xticklabels([r.label for r in runs], rotation=20, ha="right")
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(name)
                ax.legend()
                plt.tight_layout()
                plt.savefig(out_png, dpi=200)

            console.print(f"[bold green]Saved plot:[/bold green] {out_png}")
            plot_rel = "plot.png"

    # Write summary.json into sweep folder
    summary = {
        "name": name,
        "sweep_tag": sweep_tag,
        "created_at": created_at,
        "log_dir": str(log_dir.resolve()),
        "benchmarks": [dataclasses.asdict(b) for b in benchmarks],
        "runs": [dataclasses.asdict(r) for r in runs],
        "results": results,
        "logs": {f"{k[0]}::{k[1]}": (str((sweep_dir / "logs" / v.name).relative_to(sweep_dir)) if v else None) for k, v in log_index.items()},  # type: ignore
        "inputs_manifest": manifest,
        "plot": plot_rel,
    }
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Generate per-sweep HTML
    _write_sweep_index_html(
        sweeps_base=sweeps_base,
        sweep_dir=sweep_dir,
        name=name,
        sweep_tag=sweep_tag,
        created_at=created_at,
        benchmarks=benchmarks,
        runs=runs,
        results=results,
        log_index=log_index,
        manifest=manifest,
        plot_rel=plot_rel,
    )

    # Update root sweeps index (+ ensure assets)
    sweeps_base.mkdir(parents=True, exist_ok=True)
    _ensure_assets(sweeps_base)
    _write_sweeps_root_index(sweeps_base)

    console.print(
        f"\n[bold green]Sweep saved:[/bold green] {sweep_dir}\n"
        f"Open the results:\n"
        f"  1) [white]bench sweeps-view[/white]   (serves ./logs/sweeps)\n"
        f"  2) Visit: http://127.0.0.1:8765/  → click into [white]{sweep_tag}[/white] and open index.html\n"
    )