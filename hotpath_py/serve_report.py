"""Rich + plotext compact terminal renderer for hotpath serve-report.

Can be invoked directly by the C++ binary:
    python3 -m hotpath_py.serve_report <db_path>

Exit codes:
    0  — rendered successfully
    1  — error (bad db, missing data)
    2  — rich/plotext not available; caller should fall back to C++ renderer
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

try:
    import plotext as plt
    from rich.columns import Columns
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


# ── palette ──────────────────────────────────────────────────────────────────
G  = "bright_green"
Y  = "yellow"
R  = "bright_red"
M  = "magenta"
C  = "bright_cyan"
DM = "color(237)"   # very dim
BK = "bright_black" # muted / secondary
BD = "color(245)"   # body
TX = "color(251)"   # text
BR = "white"        # bright


# ── DB ───────────────────────────────────────────────────────────────────────

def _kv(db: str) -> dict[str, str]:
    with sqlite3.connect(db) as con:
        if con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='serve_analysis'").fetchone() is None:
            return {}
        return {k: v for k, v in con.execute("SELECT key,value FROM serve_analysis").fetchall()}

def _ts(db: str, like: str) -> list[tuple[float, float]]:
    try:
        with sqlite3.connect(db) as con:
            rows = con.execute(
                "SELECT sample_time,value FROM vllm_metrics WHERE metric LIKE ? ORDER BY sample_time",
                (f"%{like}%",),
            ).fetchall()
            return [(float(t), float(v)) for t, v in rows]
    except Exception:
        return []


# ── accessors ────────────────────────────────────────────────────────────────

def _f(kv: dict, k: str, d: float = -1.0) -> float:
    try: return float(kv.get(k) or d)
    except: return d

def _i(kv: dict, k: str, d: int = 0) -> int:
    try: return int(kv.get(k) or d)
    except: return d

def _s(kv: dict, k: str, d: str = "") -> str:
    return kv.get(k, d)

def _b(kv: dict, k: str) -> bool:
    return kv.get(k) == "true"


# ── formatting ────────────────────────────────────────────────────────────────

def _ms(v: float) -> str:
    if v < 0: return "—"
    if v < 1: return f"{v*1000:.0f}µs"
    return f"{v:.1f}"

def _lc(v: float, warn: float, crit: float) -> str:
    if v < 0: return BK
    if v >= crit: return R
    if v >= warn: return Y
    return TX

BAR = "█"; EMPTY = "░"

def _bar(pct: float, w: int, color: str) -> Text:
    n = round(pct / 100 * w)
    t = Text()
    t.append(BAR * n, style=color)
    t.append(EMPTY * (w - n), style=DM)
    return t


# ── chart ─────────────────────────────────────────────────────────────────────

def _chart(times: list[float], vals: list[float], color: str,
           step: bool = False, w: int = 38, h: int = 8) -> str:
    if not times: return "(no data)"
    t0 = times[0]
    times = [t - t0 for t in times]
    plt.clf()
    plt.theme("dark")
    plt.plotsize(w, h)
    plt.canvas_color("default")
    plt.axes_color("default")
    plt.ticks_color("default")
    if step:
        st, sv = [], []
        for i, (t, v) in enumerate(zip(times, vals)):
            st.append(t); sv.append(v)
            if i < len(times) - 1:
                st.append(times[i+1] - 0.001); sv.append(v)
        plt.plot(st, sv, color=color)
    else:
        plt.plot(times, vals, color=color)
    try: return plt.build()
    except: return "(chart error)"


# ── render ────────────────────────────────────────────────────────────────────

def render(db_path: str) -> int:
    if not DEPS_AVAILABLE:
        return 2

    p = Path(db_path)
    if not p.exists():
        print(f"error: file not found: {db_path}", file=sys.stderr)
        return 1

    kv = _kv(db_path)
    if not kv:
        print(f"error: no serve analysis in {db_path} — run serve-profile first", file=sys.stderr)
        return 1

    con = Console(highlight=False)

    # ── header (1 line) ──────────────────────────────────────────────────────
    model  = _s(kv, "meta.model", "?")
    engine = _s(kv, "meta.engine", "?")
    gpu    = f"{_i(kv,'meta.gpu_count',1)}x {_s(kv,'meta.gpu_name','GPU')}"
    h = Text()
    h.append("hotpath", style=f"bold {G}"); h.append("  ")
    h.append(model, style=f"bold {BR}"); h.append("  ·  ", style=DM)
    h.append(engine, style=BD); h.append("  ·  ", style=DM)
    h.append(gpu, style=C)
    con.print(h)

    # ── summary (1 line) ─────────────────────────────────────────────────────
    req = _i(kv, "meta.total_requests")
    dur = _f(kv, "meta.duration_seconds")
    rps = _f(kv, "meta.throughput_rps")
    s = Text()
    s.append(f"{req:,} req", style=BR); s.append("  ")
    s.append(f"{dur:.1f}s", style=TX);  s.append("  ")
    s.append(f"{rps:.1f} req/s", style=G)
    if _b(kv, "latency.server_timing_available"): s.append("  srv-log", style=BK)
    if _b(kv, "prefix.available"):               s.append("  pfx-cache", style=BK)
    con.print(s)
    con.rule(style=DM)

    # ── latency ──────────────────────────────────────────────────────────────
    lat = Table(show_header=True, header_style=BK, box=None,
                padding=(0, 1), show_edge=False)
    lat.add_column("latency",      style=BD, no_wrap=True, min_width=20)
    lat.add_column("p50",  justify="right", no_wrap=True, min_width=8)
    lat.add_column("p90",  justify="right", no_wrap=True, min_width=8)
    lat.add_column("p99",  justify="right", no_wrap=True, min_width=8)

    def _row(label, p50, p90, p99, em=False, warn=200., crit=600.):
        if p50 < 0 and p99 < 0: return
        lbl = Text()
        if em: lbl.append("→ ", style=G); lbl.append(label, style=f"bold {TX}")
        else:  lbl.append(label, style=BD)
        lat.add_row(lbl,
                    Text(_ms(p50), style=TX),
                    Text(_ms(p90), style=TX),
                    Text(_ms(p99), style=_lc(p99, warn, crit)))

    if _b(kv, "latency.queue_available"):
        _row("queue wait",      _f(kv,"latency.queue_p50"),         _f(kv,"latency.queue_p90"),         _f(kv,"latency.queue_p99"),         warn=20, crit=50)
    if _b(kv, "latency.server_timing_available"):
        _row("prefill (server)",_f(kv,"latency.server_prefill_p50"),_f(kv,"latency.server_prefill_p90"),_f(kv,"latency.server_prefill_p99"),warn=50, crit=150)
        _row("decode (server)", _f(kv,"latency.server_decode_p50"), _f(kv,"latency.server_decode_p90"), _f(kv,"latency.server_decode_p99"), warn=300,crit=700)
        _row("decode/tok",      _f(kv,"latency.decode_per_token_p50"),_f(kv,"latency.decode_per_token_p90"),_f(kv,"latency.decode_per_token_p99"),warn=10,crit=20)
    else:
        _row("prefill",         _f(kv,"latency.prefill_p50"),       _f(kv,"latency.prefill_p90"),       _f(kv,"latency.prefill_p99"),       warn=50, crit=150)
        _row("decode",          _f(kv,"latency.decode_total_p50"),  _f(kv,"latency.decode_total_p90"),  _f(kv,"latency.decode_total_p99"),  warn=300,crit=700)
    _row("end-to-end",          _f(kv,"latency.e2e_p50"),           _f(kv,"latency.e2e_p90"),           _f(kv,"latency.e2e_p99"),           em=True, warn=400, crit=800)
    con.print(lat)

    # ── GPU phase (inline, if available) ─────────────────────────────────────
    if _b(kv, "phase.available"):
        pre  = _f(kv, "phase.prefill_pct", 0)
        dec  = _f(kv, "phase.decode_pct",  0)
        oth  = _f(kv, "phase.other_pct",   0)
        sch  = max(0.0, 100 - pre - dec - oth)
        pt = Table.grid(padding=(0, 1))
        pt.add_column(min_width=9, style=BD)
        pt.add_column(min_width=5, justify="right")
        pt.add_column(min_width=20)
        pt.add_row(Text("gpu phase", style=BK), Text(""), Text(""))
        for name, pct, col in [("prefill",pre,M),("decode",dec,G),("schedule",sch,Y),("idle",oth,BK)]:
            pt.add_row(Text(name, style=BD), Text(f"{pct:.0f}%", style=f"bold {col}"), _bar(pct, 20, col))
        con.rule(style=DM)
        con.print(pt)

    # ── charts (side by side, compact) ───────────────────────────────────────
    tput_s  = _ts(db_path, "generation_throughput")
    batch_s = _ts(db_path, "num_requests_running")
    if tput_s or batch_s:
        con.rule(style=DM)
        CW = 38
        tc = _chart([r[0] for r in tput_s], [r[1] for r in tput_s], "green",  w=CW, h=8) if tput_s  else None
        bc = _chart([r[0] for r in batch_s],[r[1] for r in batch_s],"orange",step=True,w=CW,h=8) if batch_s else None
        from rich.panel import Panel
        panels = []
        if tc: panels.append(Panel(Text.from_ansi(tc.rstrip()), title="throughput tok/s", border_style=DM, title_align="left", padding=(0,0)))
        if bc: panels.append(Panel(Text.from_ansi(bc.rstrip()), title="batch size",       border_style=DM, title_align="left", padding=(0,0)))
        if panels:
            con.print(Columns(panels, equal=True))

    # ── kv cache + prefix sharing (2-col compact) ────────────────────────────
    con.rule(style=DM)
    left  = Table.grid(padding=(0, 1))
    right = Table.grid(padding=(0, 1))
    for t in (left, right):
        t.add_column(style=BK, min_width=18)
        t.add_column(justify="right")

    if _b(kv, "cache.hit_rate_available"):
        hit = _f(kv,"cache.hit_rate",0)*100
        left.add_row("kv hit rate", Text(f"{hit:.1f}%", style=f"bold {G if hit>60 else (Y if hit>30 else R)}"))
    if _b(kv, "cache.usage_available"):
        avg_u  = _f(kv,"cache.avg_usage", 0)*100
        peak_u = _f(kv,"cache.peak_usage",0)*100
        left.add_row("avg usage",  Text(f"{avg_u:.1f}%",  style=TX))
        left.add_row("peak usage", Text(f"{peak_u:.1f}%", style=Y if peak_u>90 else TX))
        left.add_row("evictions",  Text(str(_i(kv,"cache.evictions")), style=TX))

    if _b(kv, "prefix.available"):
        ctok = _f(kv,"prefix.cacheable_tokens_pct",0)
        right.add_row("unique prefixes",  Text(str(_i(kv,"prefix.unique_prefixes")), style=TX))
        right.add_row("cacheable tokens", Text(f"{ctok:.1f}%", style=G if ctok>50 else Y))
        if _b(kv,"cache.hit_rate_available"):
            ahr = _f(kv,"cache.hit_rate",0)*100
            right.add_row("actual hit rate", Text(f"{ahr:.1f}%", style=G if ahr>60 else Y))

    if left.row_count or right.row_count:
        con.print(Columns([left, right], equal=True))

    # ── cache hit distribution (inline compact) ──────────────────────────────
    if _b(kv, "cache.histogram_available"):
        counts = [_i(kv, f"cache.histogram_{i}") for i in range(5)]
        total  = sum(counts) or 1
        mx     = max(counts) or 1
        ht = Table.grid(padding=(0, 1))
        ht.add_column(min_width=6,  justify="right", style=BK)
        ht.add_column(min_width=22)
        ht.add_column(min_width=12, justify="right", style=BK)
        for bucket, cnt, col in zip(["0%","1-25%","25-50%","50-75%","75%+"], counts,
                                    [BK, BK, BD, G, G]):
            ht.add_row(bucket, _bar(cnt/mx*100, 20, col), f"{cnt} ({cnt/total*100:.0f}%)")
        con.rule(style=DM)
        con.print(ht)

    # ── disagg advisor ────────────────────────────────────────────────────────
    con.rule(style=DM)
    should = _b(kv, "disagg.should")
    wc     = _s(kv, "disagg.workload_class")
    mono   = _f(kv, "disagg.mono_p99_ttft",  -1)
    disagg = _f(kv, "disagg.disagg_p99_ttft",-1)
    proj   = _f(kv, "disagg.disagg_throughput",-1)
    impr   = _f(kv, "disagg.throughput_improvement",-1)
    opt_p  = _i(kv, "disagg.optimal_p")
    opt_d  = _i(kv, "disagg.optimal_d")
    bw     = _f(kv, "disagg.min_bandwidth",-1)
    caveat = _s(kv, "disagg.caveat")

    da = Table.grid(padding=(0, 1))
    da.add_column(style=BK, min_width=20)
    da.add_column()

    rec = Text()
    rec.append(" ✦ DISAGGREGATE " if should else " ✓ MONOLITHIC ", style=f"bold reverse {G if should else Y}")
    da.add_row("advisor", rec)
    if wc: da.add_row("workload", Text(wc, style=TX))
    if mono > 0 and disagg > 0:
        pct = (mono - disagg) / mono * 100
        da.add_row("prefill contention", Text(f"+{pct:.0f}% p99 latency", style=Y))
    if rps > 0 and proj > 0 and impr > 0:
        da.add_row("throughput",  Text(f"{rps:.1f} → {proj:.1f} req/s  (+{(impr-1)*100:.0f}%)", style=G))
    if mono > 0 and disagg > 0:
        ttft_d = (mono - disagg) / mono * 100
        da.add_row("p99 TTFT",    Text(f"{mono:.1f} → {disagg:.1f} ms  (-{ttft_d:.0f}%)", style=G))
    if opt_p > 0 and opt_d > 0:
        da.add_row("P/D ratio",   Text(f"1:{opt_d//opt_p}  ({opt_p}P + {opt_d}D)", style=TX))
    if bw > 0:
        da.add_row("min bandwidth", Text(f"{bw:.0f} Gbps" + ("  ⚠ below this disagg hurts" if bw < 50 else ""), style=Y if bw < 50 else TX))
    if caveat:
        short = caveat.split(".")[0].strip()  # first sentence only
        da.add_row("note", Text(short, style=BK))

    con.print(da)
    con.rule(style=DM)

    # ── footer (1 line) ──────────────────────────────────────────────────────
    f = Text()
    f.append("hotpath disagg-config ", style=BK)
    f.append(Path(db_path).name, style=BD)
    f.append("  →  vLLM / llm-d / Dynamo config", style=BK)
    con.print(f)

    return 0


if __name__ == "__main__":
    if not DEPS_AVAILABLE:
        sys.exit(2)
    if len(sys.argv) < 2:
        print("usage: python3 -m hotpath_py.serve_report <serve_profile.db>", file=sys.stderr)
        sys.exit(1)
    sys.exit(render(sys.argv[1]))
