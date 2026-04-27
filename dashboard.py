"""Dashboard de monitoreo — Trading Agent (M6 — Multi-Asset)."""

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
    """Ruta al archivo dentro del subdirectorio de versión más reciente."""
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
def load_signal() -> pd.DataFrame | None:
    """Ranking multi-ticker (trading_signal) como DataFrame."""
    try:
        path = _latest_versioned(DATA / "07_model_output" / "signal.json", "signal.json")
        raw = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(raw)
        # Handle both orient='columns' and orient='records' formats
        if isinstance(raw, dict):
            df = pd.DataFrame.from_dict(raw)
        elif isinstance(raw, list):
            df = pd.DataFrame(raw)
        return df.sort_values("score", ascending=False).reset_index(drop=True)
    except (FileNotFoundError, json.JSONDecodeError, Exception):
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
def load_benchmark_curve() -> dict | None:
    path = DATA / "08_reporting" / "benchmark_curve.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@st.cache_data(ttl=60)
def load_features() -> pd.DataFrame | None:
    path = DATA / "04_feature" / "feature_vector.parquet"
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except FileNotFoundError:
        return None


@st.cache_data(ttl=60)
def load_walk_forward() -> pd.DataFrame | None:
    path = DATA / "08_reporting" / "walk_forward.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3 — CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_equity_chart(equity_json: dict, benchmark_json: dict | None = None) -> go.Figure:
    """Curva de equity de la estrategia + benchmark SPY buy-and-hold."""
    fig = go.Figure()

    trace = equity_json.get("data", [{}])[0]
    x = trace.get("x", [])
    y = trace.get("y", [])

    if x and y:
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color="#00b4d8", width=1.8),
                name="Estrategia",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra>Estrategia</extra>",
            )
        )

    if benchmark_json:
        bt = benchmark_json.get("data", [{}])[0]
        bx = bt.get("x", [])
        by = bt.get("y", [])
        if bx and by:
            fig.add_trace(
                go.Scatter(
                    x=bx, y=by, mode="lines",
                    line=dict(color="#ffa726", width=1.2, dash="dot"),
                    name="SPY B&H",
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra>SPY B&H</extra>",
                )
            )

    fig.add_hline(
        y=10_000,
        line_dash="dot",
        line_color="rgba(255,255,255,0.30)",
        annotation_text="Capital inicial",
        annotation_position="bottom right",
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Fecha",
        yaxis_title="USD",
        hovermode="x unified",
        xaxis=dict(type="date", rangeslider=dict(visible=True)),
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        margin=dict(l=60, r=20, t=20, b=60),
        height=440,
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
    )
    return fig


def build_technical_chart(df: pd.DataFrame, ticker: str, window: str) -> go.Figure:
    """Gráfico técnico de 4 paneles para un ticker específico."""
    _window_map = {"90d": "90D", "180d": "180D", "1a": "365D", "Todo": None}
    w = _window_map.get(window)

    subset = df[df["ticker"] == ticker].copy()
    if w:
        cutoff = subset.index.max() - pd.Timedelta(w)
        subset = subset[subset.index >= cutoff]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.18, 0.18, 0.14],
        vertical_spacing=0.03,
        subplot_titles=(
            f"Precio & Bollinger — {ticker}",
            "RSI (14)",
            "MACD (12/26/9)",
            "Sentimiento",
        ),
    )

    if len(subset) == 0:
        fig.add_annotation(
            text=f"Sin datos para {ticker}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    x = subset.index

    # ── Fila 1: Precio + Bollinger + EMAs ────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["bb_upper"], mode="lines",
            line=dict(color="rgba(100,180,255,0.30)", width=1),
            name="BB Superior", showlegend=True,
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["bb_lower"], mode="lines",
            line=dict(color="rgba(100,180,255,0.30)", width=1),
            name="BB Inferior", fill="tonexty",
            fillcolor="rgba(100,180,255,0.07)", showlegend=False,
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["bb_mid"], mode="lines",
            line=dict(color="rgba(100,180,255,0.50)", width=1, dash="dot"),
            name="BB Media",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["close"], mode="lines",
            line=dict(color="#ffffff", width=1.8),
            name="Precio",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.2f}<extra></extra>",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["ema_20"], mode="lines",
            line=dict(color="#f4a261", width=1.2), name="EMA 20",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["ema_50"], mode="lines",
            line=dict(color="#e76f51", width=1.2), name="EMA 50",
        ), row=1, col=1,
    )
    if "ema_200" in subset.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=subset["ema_200"], mode="lines",
                line=dict(color="#9c27b0", width=1.5, dash="dash"),
                name="EMA 200",
            ), row=1, col=1,
        )

    # ── Fila 2: RSI ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["rsi"], mode="lines",
            line=dict(color="#80cbc4", width=1.5), name="RSI",
            hovertemplate="%{y:.1f}<extra>RSI</extra>",
        ), row=2, col=1,
    )
    for level, label, color in [(70, "Sobrecompra", "rgba(255,100,100,0.50)"),
                                  (30, "Sobreventa",  "rgba(100,255,130,0.50)")]:
        fig.add_hline(
            y=level, row=2, col=1,
            line_dash="dot", line_color=color,
            annotation_text=label,
            annotation_position="top right" if level == 70 else "bottom right",
            annotation_font_size=10,
        )
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    # ── Fila 3: MACD ─────────────────────────────────────────────────────────
    bar_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in subset["macd_hist"]]
    fig.add_trace(
        go.Bar(
            x=x, y=subset["macd_hist"],
            marker_color=bar_colors, name="Histograma MACD",
            hovertemplate="%{y:.4f}<extra>Hist</extra>",
        ), row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["macd"], mode="lines",
            line=dict(color="#2196f3", width=1.2), name="MACD",
        ), row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["macd_signal"], mode="lines",
            line=dict(color="#ff9800", width=1.2), name="Senal MACD",
        ), row=3, col=1,
    )

    # ── Fila 4: Sentimiento ───────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x, y=subset["sentiment_score"], mode="lines",
            line=dict(color="#ce93d8", width=1.2),
            fill="tozeroy", fillcolor="rgba(206,147,216,0.10)",
            name="Sentimiento",
        ), row=4, col=1,
    )
    fig.add_hline(y=0, row=4, col=1, line_dash="solid",
                  line_color="rgba(255,255,255,0.20)")

    fig.update_layout(
        template="plotly_dark",
        height=820,
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
                dict(count=90,  label="90d",  step="day",  stepmode="backward"),
                dict(count=180, label="180d", step="day",  stepmode="backward"),
                dict(count=1,   label="1a",   step="year", stepmode="backward"),
                dict(step="all", label="Todo"),
            ]
        ),
        row=1, col=1,
    )
    fig.update_yaxes(tickformat=",.2f", row=1, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4 — TAB RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_COLORS = {
    "BUY":  ("#1b5e20", "#a5d6a7", "🟢"),
    "SELL": ("#b71c1c", "#ef9a9a", "🔴"),
    "HOLD": ("#37474f", "#b0bec5", "⚪"),
}


def _badge_html(signal_val: str, small: bool = False) -> str:
    bg, fg, emoji = _SIGNAL_COLORS.get(signal_val, _SIGNAL_COLORS["HOLD"])
    size = "1.0rem" if small else "2.6rem"
    return (
        f'<div style="display:inline-block;background:{bg};color:{fg};'
        f'font-size:{size};font-weight:900;padding:0.2em 0.6em;'
        f'border-radius:8px;">{emoji} {signal_val}</div>'
    )


def render_tab_signal(signal_df: pd.DataFrame | None) -> None:
    """Ranking multi-ticker con señal BUY/HOLD/SELL, score y confianza."""
    if signal_df is None or len(signal_df) == 0:
        st.warning("No hay datos de señal. Ejecuta `kedro run` primero.")
        return

    ts = str(signal_df.get("timestamp", pd.Series(["—"])).iloc[0])[:19].replace("T", " ")
    st.caption(f"Señales generadas: {ts} UTC")

    buys = signal_df[signal_df["signal"] == "BUY"]
    if len(buys) > 0:
        st.success(f"**{len(buys)} señal(es) BUY** detectadas en el último período")
    else:
        st.info("No hay señales BUY activas (mercado en consolidación o por debajo de EMA200)")

    st.divider()

    # Tabla de ranking
    for _, row in signal_df.iterrows():
        ticker = str(row.get("ticker", "—"))
        signal_val = str(row.get("signal", "HOLD"))
        score = float(row.get("score", 0.0))
        conf = float(row.get("confidence", 0.0))
        reasoning = str(row.get("reasoning", "—"))

        col_badge, col_ticker, col_score, col_conf, col_reason = st.columns(
            [1.2, 1.0, 1.0, 1.0, 4.0]
        )
        col_badge.markdown(_badge_html(signal_val, small=True), unsafe_allow_html=True)
        col_ticker.metric(label="Ticker", value=ticker)
        col_score.metric(label="Score", value=f"{score:+.1f}" if score != -999.0 else "excluido")
        col_conf.metric(label="Confianza", value=f"{conf:.0%}" if conf > 0 else "—")
        with col_reason:
            with st.expander("Razonamiento"):
                st.caption(reasoning[:300] + ("..." if len(reasoning) > 300 else ""))

    st.divider()
    st.caption("Score = -999 → activo excluido por filtro de tendencia (close < EMA 200)")


def render_tab_portfolio(portfolio: pd.DataFrame | None, execution: pd.DataFrame | None) -> None:
    """Estado multi-posición del portfolio paper trading."""
    if portfolio is None:
        st.warning("No hay datos de portafolio disponibles.")
        return

    st.info("Modo: Paper Trading — sin dinero real comprometido")

    filled = portfolio[portfolio.get("last_order_status", pd.Series()) == "FILLED"] if "last_order_status" in portfolio.columns else pd.DataFrame()
    has_positions = len(filled) > 0

    if has_positions:
        first = filled.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Efectivo", f"${float(first.get('cash', 0)):,.2f}")
        col2.metric("Posiciones abiertas", len(filled))
        col3.metric("Valor total", f"${float(first.get('total_value', 0)):,.2f}")

        st.divider()
        st.subheader("Posiciones activas")
        for _, pos in filled.iterrows():
            ticker = str(pos.get("ticker", "—"))
            pos_val = float(pos.get("position_value", 0))
            stop_mult = float(pos.get("stop_loss_atr_mult", 2.0))
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Ticker", ticker)
            c2.metric("Valor posición", f"${pos_val:,.2f}")
            c3.metric("Stop-Loss mult.", f"{stop_mult}× ATR")
    else:
        row = portfolio.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Efectivo", f"${float(row.get('cash', 0)):,.2f}")
        col2.metric("Posiciones abiertas", 0)
        col3.metric("Valor total", f"${float(row.get('total_value', 0)):,.2f}")
        st.info("Sin posiciones activas — capital en efectivo.")

    if execution is not None and len(execution) > 0:
        st.divider()
        st.subheader("Última sesión de órdenes")
        filled_orders = execution[execution["status"] == "FILLED"]
        if len(filled_orders) > 0:
            st.dataframe(
                filled_orders[["ticker", "signal", "order_size_usd", "confidence", "mode"]],
                use_container_width=True,
            )
        else:
            exc = execution.iloc[0]
            st.caption(f"Sin órdenes ejecutadas — razón: {exc.get('reason', '—')}")


def render_tab_backtesting(
    metrics: pd.Series | None,
    equity_json: dict | None,
    benchmark_json: dict | None,
    walk_forward: pd.DataFrame | None = None,
) -> None:
    """Métricas de rendimiento + curva de equity vs benchmark + walk-forward."""
    if metrics is None:
        st.warning("No hay métricas de backtesting disponibles.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Sharpe Ratio", f"{float(metrics['sharpe_ratio']):.2f}")
    col2.metric("Max Drawdown", f"{float(metrics['max_drawdown_pct']):.1f}%")
    col3.metric("CAGR", f"{float(metrics['cagr_pct']):.1f}%")
    col4.metric("Win Rate", f"{float(metrics['win_rate_pct']):.1f}%")
    col5.metric("Trades/año", f"{float(metrics.get('trades_per_year', 0)):.0f}")

    col6, col7, col8, col9 = st.columns([1, 1, 1, 1])
    col6.metric("Equity final", f"${float(metrics['final_equity_usd']):,.0f}")
    col7.metric("Retorno total", f"{float(metrics['total_return_pct']):.1f}%")
    col8.metric("N° Operaciones", int(metrics["n_trades"]))
    cb = int(metrics.get("circuit_break_events", 0))
    col9.metric("Circuit Breaks", cb, help="Veces que el circuit breaker protegió el capital")

    st.divider()
    st.subheader("Curva de Equity vs SPY Buy & Hold")
    if equity_json:
        fig = build_equity_chart(equity_json, benchmark_json)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Gráfico de equity no disponible.")

    st.caption(
        "La línea naranja (--) representa el benchmark SPY buy-and-hold "
        "con el mismo capital inicial."
    )

    # ── Walk-Forward Validation ────────────────────────────────────────────────
    if walk_forward is not None and len(walk_forward) > 0:
        st.divider()
        st.subheader("Validación Walk-Forward")
        st.caption(
            "Compara el rendimiento en el período de entrenamiento (in-sample) "
            "vs el período no visto (out-of-sample).  Un sistema robusto muestra "
            "métricas similares en ambas ventanas."
        )

        cols = st.columns(len(walk_forward))
        for col, (_, row) in zip(cols, walk_forward.iterrows()):
            periodo = str(row["periodo"])
            sharpe = float(row["sharpe"])
            max_dd = float(row["max_drawdown_pct"])
            cagr = float(row["cagr_pct"])
            n_dias = int(row["n_dias"])

            label = "In-Sample" if "In-sample" in periodo else (
                "Out-of-Sample" if "Out-of-sample" in periodo else "Completo"
            )
            with col:
                st.markdown(f"**{label}**")
                st.caption(f"({n_dias} dias)")
                color = "normal" if sharpe > 0 else "inverse"
                st.metric("Sharpe", f"{sharpe:.2f}")
                st.metric("MaxDD", f"{max_dd:.1f}%")
                st.metric("CAGR", f"{cagr:.1f}%")

        if len(walk_forward) >= 2:
            is_row = walk_forward[walk_forward["periodo"].str.contains("In-sample")].iloc[0]
            oos_row = walk_forward[walk_forward["periodo"].str.contains("Out-of-sample")].iloc[0]
            sharpe_ratio = float(oos_row["sharpe"]) / max(float(is_row["sharpe"]), 0.01)
            if sharpe_ratio > 0.5:
                st.success(
                    f"Robustez aceptable: Sharpe out-of-sample = "
                    f"{float(oos_row['sharpe']):.2f} "
                    f"({sharpe_ratio:.0%} del in-sample)."
                )
            else:
                st.warning(
                    f"Divergencia in/out-of-sample detectada (ratio={sharpe_ratio:.0%}). "
                    f"Considera incrementar la muestra o reducir la complejidad del modelo."
                )


def render_tab_tecnico(features: pd.DataFrame | None) -> None:
    """Análisis técnico multi-ticker con selector de activo y EMA 200."""
    if features is None:
        st.warning("No hay datos de features disponibles.")
        return

    tickers = sorted(features["ticker"].unique().tolist())

    col_sel, col_win = st.columns([2, 3])
    with col_sel:
        ticker = st.selectbox("Activo", tickers, index=0)
    with col_win:
        window = st.radio("Período", ["90d", "180d", "1a", "Todo"], index=2, horizontal=True)

    fig = build_technical_chart(features, ticker, window)
    st.plotly_chart(fig, use_container_width=True, key=f"tech_{ticker}_{window}")

    # Resumen del último día para el ticker seleccionado
    last_row = features[features["ticker"] == ticker].iloc[-1]
    date_str = features[features["ticker"] == ticker].index[-1].strftime("%Y-%m-%d")
    st.caption(
        f"**{ticker}** — {date_str} | "
        f"Close: ${float(last_row['close']):,.2f} | "
        f"EMA200: ${float(last_row.get('ema_200', 0)):,.2f} | "
        f"RSI: {float(last_row['rsi']):.1f} | "
        f"ATR: ${float(last_row['atr']):,.2f}"
    )


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
        st.caption("Dashboard multi-asset — M6")
        st.divider()
        if st.button("🔄 Actualizar datos", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption("📁 Datos desde `data/`")
        st.caption("🔒 Paper trading únicamente")
        st.caption("📅 Universo: SPY · QQQ · GLD · TLT · AAPL · MSFT · BTC-USD")

    st.title("Trading Agent — Dashboard Multi-Asset")

    signal       = load_signal()
    portfolio    = load_portfolio()
    execution    = load_execution()
    metrics      = load_metrics()
    equity       = load_equity_curve()
    benchmark    = load_benchmark_curve()
    features     = load_features()
    walk_forward = load_walk_forward()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Ranking Señales", "💼 Portafolio", "📈 Backtesting", "🔬 Análisis Técnico"]
    )

    with tab1:
        render_tab_signal(signal)

    with tab2:
        render_tab_portfolio(portfolio, execution)

    with tab3:
        render_tab_backtesting(metrics, equity, benchmark, walk_forward)

    with tab4:
        render_tab_tecnico(features)


if __name__ == "__main__":
    main()
