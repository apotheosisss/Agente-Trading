"""
Dashboard de monitoreo — Trading Agent.

Uso:
    uv run streamlit run dashboard.py
"""
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Configuracion ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

BRANCHES = {
    "📊 Polymarket — Largo plazo": BASE / ".claude/worktrees/polymarket-work",
    "₿ Crypto — Corto plazo":     BASE / ".claude/worktrees/wonderful-villani-fa15cf",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def latest_versioned(directory: Path, filename: str) -> Path | None:
    d = directory / filename
    if not d.exists():
        return None
    files = sorted(d.glob(f"*/{filename}"), key=os.path.getmtime)
    return files[-1] if files else None


def load_metrics(data_path: Path) -> pd.DataFrame | None:
    p = data_path / "data/08_reporting/metrics.csv"
    return pd.read_csv(p) if p.exists() else None


def load_walk_forward(data_path: Path) -> pd.DataFrame | None:
    p = data_path / "data/08_reporting/walk_forward.csv"
    return pd.read_csv(p) if p.exists() else None


def load_equity_curve(data_path: Path) -> dict | None:
    p = data_path / "data/08_reporting/equity_curve.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_benchmark_curve(data_path: Path) -> dict | None:
    p = data_path / "data/08_reporting/benchmark_curve.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_signal(data_path: Path) -> pd.DataFrame | None:
    p = latest_versioned(data_path / "data/07_model_output", "signal.json")
    if p is None:
        return None
    try:
        return pd.read_json(p)
    except Exception:
        return None


def load_execution(data_path: Path) -> pd.DataFrame | None:
    p = latest_versioned(data_path / "data/07_model_output", "execution_record.csv")
    if p is None:
        return None
    return pd.read_csv(p)


def load_daily_logs(data_path: Path) -> list[tuple[str, str]]:
    log_dir = data_path / "data/08_reporting/daily_log"
    if not log_dir.exists():
        return []
    logs = sorted(log_dir.glob("*.txt"), reverse=True)
    result = []
    for log in logs[:30]:
        date = log.stem
        content = log.read_text(encoding="utf-8")
        result.append((date, content))
    return result


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Trading Agent")
    st.caption("Sistema de inversión algorítmica")
    st.divider()

    branch_name = st.radio("Estrategia", list(BRANCHES.keys()))
    data_path = BRANCHES[branch_name]

    st.divider()
    st.caption(f"Datos: `{data_path.name}`")
    st.caption(f"Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_perf, tab_signals, tab_logs = st.tabs([
    "📈 Rendimiento histórico",
    "🎯 Señales de hoy",
    "📋 Historial de logs",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RENDIMIENTO HISTÓRICO
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    metrics = load_metrics(data_path)
    wf      = load_walk_forward(data_path)
    ec      = load_equity_curve(data_path)
    bench   = load_benchmark_curve(data_path)

    if metrics is None:
        st.warning("Sin datos de backtest. Ejecuta `kedro run` primero.")
    else:
        m = metrics.iloc[0]

        # ── KPIs ──────────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CAGR", f"{m['cagr_pct']:.1f}%")
        c2.metric("Sharpe", f"{m['sharpe_ratio']:.2f}")
        c3.metric("Max Drawdown", f"{m['max_drawdown_pct']:.1f}%")
        c4.metric("Win Rate", f"{m['win_rate_pct']:.1f}%")
        c5.metric("Capital final", f"${m['final_equity_usd']:,.0f}",
                  delta=f"+{m['total_return_pct']:.0f}% total")

        st.divider()

        # ── Equity curve ──────────────────────────────────────────────────────
        if ec:
            fig = go.Figure()

            # Equity del sistema
            eq_trace = ec["data"][0]
            fig.add_trace(go.Scatter(
                x=eq_trace["x"],
                y=eq_trace["y"],
                name="Estrategia",
                line=dict(color="#00b4d8", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,180,216,0.08)",
            ))

            # Benchmark SPY
            if bench:
                bm_trace = bench["data"][0]
                fig.add_trace(go.Scatter(
                    x=bm_trace["x"],
                    y=bm_trace["y"],
                    name="SPY (buy & hold)",
                    line=dict(color="#f4a261", width=1.5, dash="dash"),
                ))

            fig.update_layout(
                title="Evolución del capital (backtest)",
                xaxis_title="Fecha",
                yaxis_title="Capital (USD)",
                hovermode="x unified",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        # ── Walk-forward ──────────────────────────────────────────────────────
        if wf is not None:
            st.subheader("Walk-forward validation")
            wf_display = wf.copy()
            wf_display.columns = ["Período", "Sharpe", "Max DD %", "CAGR %", "Días"]
            st.dataframe(
                wf_display.style.format({
                    "Sharpe": "{:.2f}",
                    "Max DD %": "{:.1f}%",
                    "CAGR %": "{:.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # ── Stats extra ───────────────────────────────────────────────────────
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Trades totales", int(m["n_trades"]))
        c2.metric("Trades/año", f"{m['trades_per_year']:.1f}")
        c3.metric("Circuit breaks", int(m["circuit_break_events"]))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEÑALES DE HOY
# ══════════════════════════════════════════════════════════════════════════════
with tab_signals:
    signal_df = load_signal(data_path)
    exec_df   = load_execution(data_path)

    if signal_df is None:
        st.warning("Sin señales. Ejecuta `uv run python scheduler.py --force`.")
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        st.subheader(f"Señales generadas — {today}")

        # Colorear por señal
        def color_signal(val):
            if val == "BUY":
                return "background-color: #1a472a; color: #90ee90"
            elif val == "SELL":
                return "background-color: #5a0000; color: #ffb3b3"
            return ""

        cols_show = ["ticker", "signal", "score", "poly_boost"]
        cols_available = [c for c in cols_show if c in signal_df.columns]
        display = signal_df[cols_available].copy()
        display.columns = ["Ticker", "Señal", "Score", "Boost Poly"][:len(cols_available)]

        st.dataframe(
            display.style
                .map(color_signal, subset=["Señal"])
                .format({"Score": "{:.1f}", "Boost Poly": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        # ── Resumen BUY/HOLD/SELL ─────────────────────────────────────────────
        n_buy  = (signal_df["signal"] == "BUY").sum()
        n_hold = (signal_df["signal"] == "HOLD").sum()
        n_sell = (signal_df["signal"] == "SELL").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("BUY", n_buy, delta="entrada recomendada" if n_buy else None)
        c2.metric("HOLD", n_hold)
        c3.metric("SELL", n_sell)

        # ── Órdenes papel ─────────────────────────────────────────────────────
        if exec_df is not None and not exec_df.empty:
            st.divider()
            st.subheader("Órdenes simuladas (paper trading)")
            filled = exec_df[exec_df["status"] == "FILLED"]
            if not filled.empty:
                st.dataframe(
                    filled[["ticker", "signal", "order_size_usd", "confidence", "timestamp"]]
                    .rename(columns={
                        "ticker": "Ticker",
                        "signal": "Señal",
                        "order_size_usd": "Tamaño (USD)",
                        "confidence": "Confianza",
                        "timestamp": "Hora",
                    })
                    .style.format({"Tamaño (USD)": "${:,.0f}", "Confianza": "{:.2f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Sin órdenes ejecutadas hoy.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORIAL DE LOGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_logs:
    logs = load_daily_logs(data_path)

    if not logs:
        st.warning("Sin logs. El scheduler aún no ha corrido.")
    else:
        st.subheader(f"Últimos {len(logs)} días de señales")

        dates = [date for date, _ in logs]
        selected = st.selectbox("Selecciona fecha", dates)

        content = next(c for d, c in logs if d == selected)
        st.code(content, language=None)

        st.divider()
        st.caption(f"Logs disponibles: {len(logs)} días | Directorio: `data/08_reporting/daily_log/`")
