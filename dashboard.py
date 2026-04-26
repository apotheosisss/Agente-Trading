"""Dashboard de monitoreo — Trading Agent (M5)."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent
DATA = ROOT / "data"


# ─────────────────────────────────────────────────────────────────────────────
# 1 — PATH HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _latest_versioned(base_dir: Path, filename: str) -> Path:
    """Devuelve la ruta al archivo dentro del subdirectorio de versión más reciente."""
    try:
        subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    except OSError:
        raise FileNotFoundError(f"Directorio no encontrado: {base_dir}")
    if not subdirs:
        raise FileNotFoundError(f"Sin versiones en {base_dir}")
    latest = max(subdirs, key=lambda p: p.name)
    return latest / filename


# ─────────────────────────────────────────────────────────────────────────────
# 2 — DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_signal() -> dict | None:
    try:
        path = _latest_versioned(DATA / "07_model_output" / "signal.json", "signal.json")
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(ttl=60)
def load_portfolio() -> pd.DataFrame | None:
    path = DATA / "07_model_output" / "portfolio_state.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


@st.cache_data(ttl=60)
def load_execution() -> pd.DataFrame | None:
    try:
        path = _latest_versioned(
            DATA / "07_model_output" / "execution_record.csv",
            "execution_record.csv",
        )
        return pd.read_csv(path)
    except (FileNotFoundError, Exception):
        return None


@st.cache_data(ttl=60)
def load_metrics() -> pd.Series | None:
    path = DATA / "08_reporting" / "metrics.csv"
    try:
        return pd.read_csv(path).iloc[0]
    except FileNotFoundError:
        return None


@st.cache_data(ttl=60)
def load_equity_curve() -> dict | None:
    path = DATA / "08_reporting" / "equity_curve.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(ttl=60)
def load_features() -> pd.DataFrame | None:
    path = DATA / "04_feature" / "feature_vector.parquet"
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3 — CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_equity_chart(equity_json: dict) -> go.Figure:
    trace = equity_json.get("data", [{}])[0]
    x = trace.get("x", [])
    y = trace.get("y", [])

    fig = go.Figure()

    if len(x) and len(y):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="#00b4d8", width=1.5),
                name="Equity",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_hline(
            y=10_000,
            line_dash="dot",
            line_color="rgba(255,255,255,0.35)",
            annotation_text="Capital inicial",
            annotation_position="bottom right",
        )
    else:
        fig.add_annotation(
            text="Sin datos de equity", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14),
        )

    fig.update_layout(
        template="plotly_dark",
        title=None,
        xaxis_title="Fecha",
        yaxis_title="USD",
        hovermode="x unified",
        xaxis=dict(type="date", rangeslider=dict(visible=True)),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        margin=dict(l=60, r=20, t=20, b=60),
        height=420,
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
    )
    return fig


def build_technical_chart(df: pd.DataFrame, window: str) -> go.Figure:
    _window_map = {"90d": "90D", "180d": "180D", "1a": "365D", "Todo": None}
    w = _window_map.get(window)
    subset = df.last(w) if w else df

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.18, 0.18, 0.14],
        vertical_spacing=0.03,
        subplot_titles=(
            "Precio & Bandas de Bollinger",
            "RSI (14)",
            "MACD (12/26/9)",
            "Sentimiento",
        ),
    )

    if len(subset) == 0:
        fig.add_annotation(
            text="Sin datos para el período seleccionado",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    x = subset.index

    # ── Fila 1: Precio + Bollinger + EMAs ───────────────────────────────────
    # Orden crítico: upper → lower (fill) → mid → close → ema20 → ema50
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["bb_upper"], mode="lines",
            line=dict(color="rgba(100,180,255,0.30)", width=1),
            name="BB Superior", showlegend=True,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["bb_lower"], mode="lines",
            line=dict(color="rgba(100,180,255,0.30)", width=1),
            name="BB Inferior", fill="tonexty",
            fillcolor="rgba(100,180,255,0.07)", showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["bb_mid"], mode="lines",
            line=dict(color="rgba(100,180,255,0.50)", width=1, dash="dot"),
            name="BB Media",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["close"], mode="lines",
            line=dict(color="#ffffff", width=1.5),
            name="Precio",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["ema_20"], mode="lines",
            line=dict(color="#f4a261", width=1.2),
            name="EMA 20",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["ema_50"], mode="lines",
            line=dict(color="#e76f51", width=1.2),
            name="EMA 50",
        ),
        row=1, col=1,
    )

    # ── Fila 2: RSI ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["rsi"], mode="lines",
            line=dict(color="#80cbc4", width=1.5),
            name="RSI",
            hovertemplate="%{y:.1f}<extra>RSI</extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(
        y=70, row=2, col=1,
        line_dash="dot", line_color="rgba(255,100,100,0.50)",
        annotation_text="Sobrecompra", annotation_position="top right",
        annotation_font_size=10,
    )
    fig.add_hline(
        y=30, row=2, col=1,
        line_dash="dot", line_color="rgba(100,255,130,0.50)",
        annotation_text="Sobreventa", annotation_position="bottom right",
        annotation_font_size=10,
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    # ── Fila 3: MACD ─────────────────────────────────────────────────────────
    bar_colors = [
        "#26a69a" if v >= 0 else "#ef5350"
        for v in subset["macd_hist"]
    ]
    fig.add_trace(
        go.Bar(
            x=x, y=subset["macd_hist"],
            marker_color=bar_colors,
            name="Histograma MACD",
            hovertemplate="%{y:.2f}<extra>Hist</extra>",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["macd"], mode="lines",
            line=dict(color="#2196f3", width=1.2),
            name="MACD",
            hovertemplate="%{y:.2f}<extra>MACD</extra>",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["macd_signal"], mode="lines",
            line=dict(color="#ff9800", width=1.2),
            name="Señal MACD",
            hovertemplate="%{y:.2f}<extra>Señal</extra>",
        ),
        row=3, col=1,
    )

    # ── Fila 4: Sentimiento ───────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["sentiment_score"], mode="lines",
            line=dict(color="#ce93d8", width=1.2),
            fill="tozeroy", fillcolor="rgba(206,147,216,0.10)",
            name="Sentimiento",
            hovertemplate="%{y:.3f}<extra>Sentimiento</extra>",
        ),
        row=4, col=1,
    )
    fig.add_hline(
        y=0, row=4, col=1,
        line_dash="solid", line_color="rgba(255,255,255,0.20)",
    )

    # ── Layout global ─────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        height=800,
        hovermode="x unified",
        barmode="relative",
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1, font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=90, label="90d", step="day", stepmode="backward"),
                dict(count=180, label="180d", step="day", stepmode="backward"),
                dict(count=1, label="1a", step="year", stepmode="backward"),
                dict(step="all", label="Todo"),
            ]
        ),
        row=1, col=1,
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4 — TAB RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def _signal_badge(signal_val: str) -> str:
    colors = {
        "BUY":  ("🟢", "#1b5e20", "#a5d6a7"),
        "SELL": ("🔴", "#b71c1c", "#ef9a9a"),
        "HOLD": ("⚪", "#37474f", "#b0bec5"),
    }
    emoji, bg, fg = colors.get(signal_val, colors["HOLD"])
    return (
        f'<div style="display:inline-block;background:{bg};color:{fg};'
        f'font-size:2.6rem;font-weight:900;padding:0.35em 0.8em;'
        f'border-radius:12px;letter-spacing:0.08em;">'
        f'{emoji} {signal_val}</div>'
    )


def render_tab_signal(signal: dict | None) -> None:
    if signal is None:
        st.warning("No hay datos de señal disponibles. Ejecuta el pipeline primero.")
        return

    val = signal["signal"]["0"]
    conf = float(signal["confidence"]["0"])
    score = float(signal["score"]["0"])
    reasoning = str(signal["reasoning"]["0"])
    ticker = str(signal["ticker"]["0"])
    ts = str(signal["timestamp"]["0"])[:19].replace("T", " ")

    st.markdown(_signal_badge(val), unsafe_allow_html=True)
    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric("Confianza", f"{conf:.0%}")
    col2.metric("Score", f"{score:+.1f}")
    col3.metric("Ticker", ticker)

    st.progress(conf, text=f"Confianza: {conf:.0%}")
    st.caption(f"Señal generada: {ts} UTC")
    st.divider()

    with st.expander("Ver razonamiento completo"):
        st.text(reasoning)


def render_tab_portfolio(portfolio: pd.DataFrame | None, execution: pd.DataFrame | None) -> None:
    if portfolio is None:
        st.warning("No hay datos de portafolio disponibles.")
        return

    row = portfolio.iloc[0]
    st.info("Modo: Paper Trading — sin dinero real comprometido")

    col1, col2, col3 = st.columns(3)
    col1.metric("Efectivo disponible", f"${float(row['cash']):,.2f}")
    col2.metric("Valor posición", f"${float(row['position_value']):,.2f}")
    col3.metric("Valor total", f"${float(row['total_value']):,.2f}")

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Stop-Loss", f"{float(row['stop_loss_pct']):.1%}")
    c2.metric("Take-Profit", f"{float(row['take_profit_pct']):.1%}")

    st.divider()
    st.subheader("Última orden")

    if execution is not None:
        exc = execution.iloc[0]
        status = str(exc["status"])
        badge_color = "#1b5e20" if status == "FILLED" else "#37474f"
        badge_fg = "#a5d6a7" if status == "FILLED" else "#b0bec5"
        st.markdown(
            f'<span style="background:{badge_color};color:{badge_fg};'
            f'padding:3px 12px;border-radius:8px;font-weight:700;">{status}</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        e1, e2 = st.columns(2)
        e1.metric("Tamaño orden", f"${float(exc['order_size_usd']):,.2f}")
        e2.metric("Confianza", f"{float(exc['confidence']):.0%}")
        st.caption(f"Razón: {exc['reason']}")
        ts = str(exc["timestamp"])[:19].replace("T", " ")
        st.caption(f"Timestamp: {ts} UTC  |  Modo: {exc['mode']}")
    else:
        st.caption("Sin registros de ejecución.")


def render_tab_backtesting(metrics: pd.Series | None, equity_json: dict | None) -> None:
    if metrics is None:
        st.warning("No hay métricas de backtesting disponibles.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio", f"{float(metrics['sharpe_ratio']):.4f}")
    col2.metric("Max Drawdown", f"{float(metrics['max_drawdown_pct']):.2f}%")
    col3.metric("Win Rate", f"{float(metrics['win_rate_pct']):.1f}%")
    col4.metric("CAGR", f"{float(metrics['cagr_pct']):.2f}%")

    col5, col6, _ = st.columns([1, 1, 2])
    col5.metric("Equity Final", f"${float(metrics['final_equity_usd']):,.2f}")
    col6.metric("N° Operaciones", int(metrics["n_trades"]))

    st.divider()
    st.subheader("Curva de Equity")

    if equity_json:
        fig = build_equity_chart(equity_json)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Gráfico de equity no disponible.")


def render_tab_tecnico(df: pd.DataFrame | None) -> None:
    if df is None:
        st.warning("No hay datos de features disponibles.")
        return

    window = st.radio(
        "Período",
        ["90d", "180d", "1a", "Todo"],
        index=3,
        horizontal=True,
    )

    fig = build_technical_chart(df, window)
    st.plotly_chart(fig, use_container_width=True, key=f"tecnico_{window}")


# ─────────────────────────────────────────────────────────────────────────────
# 5 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Trading Agent Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("📈 Trading Agent")
        st.caption("Dashboard de monitoreo — M5")
        st.divider()
        if st.button("🔄 Actualizar datos", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption("Datos desde `data/`")
        st.caption("Paper trading únicamente")

    st.title("Trading Agent — Dashboard")

    signal    = load_signal()
    portfolio = load_portfolio()
    execution = load_execution()
    metrics   = load_metrics()
    equity    = load_equity_curve()
    features  = load_features()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Señal", "💼 Portafolio", "📈 Backtesting", "🔬 Análisis Técnico"]
    )

    with tab1:
        render_tab_signal(signal)

    with tab2:
        render_tab_portfolio(portfolio, execution)

    with tab3:
        render_tab_backtesting(metrics, equity)

    with tab4:
        render_tab_tecnico(features)


if __name__ == "__main__":
    main()
