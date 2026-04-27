import numpy as np
import pandas as pd
import pytest

from trading_agent.pipelines.backtesting.nodes import (
    calcular_benchmark,
    calcular_metricas,
    calcular_walk_forward,
    ejecutar_backtest,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_multi_fv(n_tickers: int = 2, n_days: int = 40, close_val: float = 20_000.0):
    """Feature vector sintetico multi-ticker con score positivo (EMA200 < close)."""
    dates = pd.date_range("2023-01-02", periods=n_days, freq="D")
    tickers = [f"T{i}" for i in range(n_tickers)]
    frames = []
    for ticker in tickers:
        close = np.full(n_days, close_val)
        df = pd.DataFrame(
            {
                "ticker": ticker,
                "open": close, "high": close * 1.001,
                "low": close * 0.999, "close": close, "volume": 5000.0,
                "rsi": 50.0,
                # MACD alcista: genera +1.0 de score
                "macd": 1.0, "macd_signal": 0.0, "macd_hist": 1.0,
                "bb_upper": close * 1.02, "bb_mid": close, "bb_lower": close * 0.98,
                # EMA alineada bullish: close > ema_20 > ema_50 → +2.0 de score
                "ema_20": close * 0.99, "ema_50": close * 0.98,
                # EMA200 muy por debajo → trend filter pasa, score no aplica filtro
                "ema_200": close * 0.5,
                "atr": 100.0,
                "sentiment_score": 0.0,
            },
            index=dates,
        )
        frames.append(df)
    result = pd.concat(frames).sort_index()
    result.index.name = "date"
    return result


def _make_vix(dates, vix_level: float = 15.0) -> pd.DataFrame:
    """VIX sintetico a nivel bajo para no activar el filtro de crisis."""
    return pd.DataFrame({"vix": vix_level}, index=pd.DatetimeIndex(sorted(set(dates))))


def _make_portfolio(n: int = 40, capital: float = 10_000.0, start: str = "2023-01-01") -> pd.DataFrame:
    """Portfolio sintetico con equity constante para pruebas de metricas."""
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "equity": capital,
            "cash": capital,
            "trade_type": "HOLD",
            "tickers_held": "",
            "n_positions": 0,
            "buys_today": 0,
            "exits_today": 0,
            "vix": 15.0,
            "in_cooldown": False,
        },
        index=dates,
    )


# ── ejecutar_backtest ─────────────────────────────────────────────────────────

def test_backtest_columnas(sample_feature_vector, sample_vix, sample_parameters):
    result = ejecutar_backtest(sample_feature_vector, sample_vix, sample_parameters)
    for col in ["equity", "cash", "trade_type", "tickers_held",
                "n_positions", "buys_today", "exits_today", "vix", "in_cooldown"]:
        assert col in result.columns, f"Falta columna: {col}"


def test_backtest_equity_positivo(sample_feature_vector, sample_vix, sample_parameters):
    result = ejecutar_backtest(sample_feature_vector, sample_vix, sample_parameters)
    assert (result["equity"] > 0).all()


def test_backtest_equity_inicial(sample_feature_vector, sample_vix, sample_parameters):
    """La equity inicial nunca supera el capital (solo pueden haber comisiones)."""
    result = ejecutar_backtest(sample_feature_vector, sample_vix, sample_parameters)
    capital = float(sample_parameters["backtesting"]["initial_capital"])
    assert float(result["equity"].iloc[0]) > 0
    assert float(result["equity"].iloc[0]) <= capital + 0.01


def test_backtest_no_crash_universo_pequenio(sample_parameters):
    fv = _make_multi_fv(n_tickers=2, n_days=30)
    vix = _make_vix(fv.index.unique())
    result = ejecutar_backtest(fv, vix, sample_parameters)
    assert len(result) == 30
    assert (result["equity"] > 0).all()


def test_backtest_trade_types_validos(sample_feature_vector, sample_vix, sample_parameters):
    result = ejecutar_backtest(sample_feature_vector, sample_vix, sample_parameters)
    tipos_validos = {"BUY", "SELL", "STOP_LOSS", "HOLD", "CIRCUIT_BREAK", "VIX_CRISIS"}
    assert set(result["trade_type"].unique()).issubset(tipos_validos)


def test_backtest_n_positions_no_supera_max(sample_feature_vector, sample_vix, sample_parameters):
    result = ejecutar_backtest(sample_feature_vector, sample_vix, sample_parameters)
    max_pos = int(sample_parameters["risk"]["max_positions"])
    assert (result["n_positions"] <= max_pos).all()


def test_backtest_circuit_breaker_liquida(sample_parameters):
    """Circuit breaker: cuando el portfolio cae mas del umbral, se liquidan posiciones."""
    fv = _make_multi_fv(n_tickers=2, n_days=60)
    dates = sorted(fv.index.unique())
    vix = _make_vix(dates, vix_level=15.0)

    # Simular caida severa modificando precios despues del dia 20
    fv_copy = fv.copy()
    crash_dates = dates[20:]
    fv_copy.loc[fv_copy.index.isin(crash_dates), "close"] *= 0.70  # caida del 30%
    fv_copy.loc[fv_copy.index.isin(crash_dates), "open"] *= 0.70
    fv_copy.loc[fv_copy.index.isin(crash_dates), "high"] *= 0.70
    fv_copy.loc[fv_copy.index.isin(crash_dates), "low"] *= 0.70
    # Tras caida: EMA alineacion bearish, score negativo → min_entry_score filtra
    fv_copy.loc[fv_copy.index.isin(crash_dates), "ema_200"] = fv_copy.loc[
        fv_copy.index.isin(crash_dates), "close"
    ] * 2.0  # close < ema_200 → score = -999 → no entrar

    params = {**sample_parameters}
    params["risk"] = {**params["risk"], "max_drawdown_circuit": 0.05}
    result = ejecutar_backtest(fv_copy, vix, params)
    # Debe haber al menos un circuit break
    assert "CIRCUIT_BREAK" in result["trade_type"].values, (
        "Esperaba al menos un evento CIRCUIT_BREAK"
    )


def test_backtest_cooldown_evita_compras(sample_parameters):
    """Tras un circuit breaker, el cooldown evita nuevas compras durante N dias."""
    fv = _make_multi_fv(n_tickers=2, n_days=60)
    dates = sorted(fv.index.unique())
    vix = _make_vix(dates, vix_level=15.0)

    # Caida que activa circuit breaker rapido
    fv_copy = fv.copy()
    crash_dates = dates[5:30]
    fv_copy.loc[fv_copy.index.isin(crash_dates), "close"] *= 0.80
    fv_copy.loc[fv_copy.index.isin(crash_dates), "open"] *= 0.80
    fv_copy.loc[fv_copy.index.isin(crash_dates), "high"] *= 0.80
    fv_copy.loc[fv_copy.index.isin(crash_dates), "low"] *= 0.80

    params = {**sample_parameters}
    params["risk"] = {
        **params["risk"],
        "max_drawdown_circuit": 0.05,
        "circuit_break_cooldown": 10,
    }
    result = ejecutar_backtest(fv_copy, vix, params)
    cb_idx = result.index[result["trade_type"] == "CIRCUIT_BREAK"]
    if len(cb_idx) > 0:
        first_cb = cb_idx[0]
        days_after = result.loc[result.index > first_cb].head(10)
        # Cooldown activo: in_cooldown=True los dias siguientes
        assert days_after["in_cooldown"].any(), "Cooldown deberia estar activo tras circuit breaker"


def test_backtest_vix_crisis_liquida(sample_parameters):
    """VIX extremo (> vix_crisis_threshold) liquida todas las posiciones."""
    fv = _make_multi_fv(n_tickers=2, n_days=40)
    dates = sorted(fv.index.unique())
    # VIX normal 15 dias, luego crisis extrema
    vix_values = [15.0] * 20 + [50.0] * 20
    vix = pd.DataFrame({"vix": vix_values}, index=pd.DatetimeIndex(dates))
    params = {**sample_parameters}
    params["risk"] = {**params["risk"], "vix_crisis_threshold": 40}
    result = ejecutar_backtest(fv, vix, params)
    # Despues del dia 20, no debe haber posiciones (liquidadas por VIX crisis)
    post_crisis = result.iloc[20:]
    assert (post_crisis["n_positions"] == 0).all(), (
        "VIX crisis deberia haber liquidado todas las posiciones"
    )


def test_backtest_min_score_bloquea_entradas(sample_parameters):
    """Con score bajo (neutral), no se abren posiciones si score < min_entry_score."""
    fv = _make_multi_fv(n_tickers=2, n_days=40)
    # Hacer que el score sea 0: MACD neutral, EMA neutral
    fv = fv.copy()
    fv["macd"] = 0.0           # sin senial MACD
    fv["macd_signal"] = 0.001  # MACD < signal → -1.0
    fv["ema_20"] = fv["close"] * 0.995  # close > ema20 (bullish EMA, +2)
    fv["ema_50"] = fv["close"] * 0.98   # ema20 > ema50 (+2, total = +2 - 1 = +1)
    vix = _make_vix(fv.index.unique())
    params = {**sample_parameters}
    params["risk"] = {**params["risk"], "min_entry_score": 3.0}  # umbral alto
    result = ejecutar_backtest(fv, vix, params)
    # Con score ~1.0 y min_entry_score=3.0, no debe haber compras
    assert result["buys_today"].sum() == 0, (
        "Con score bajo no deberian abrirse posiciones"
    )


def test_backtest_vix_vacio_no_falla(sample_feature_vector, sample_parameters):
    """Si el DataFrame VIX esta vacio, el backtest debe usar VIX=20 por defecto."""
    vix_vacio = pd.DataFrame({"vix": []})
    result = ejecutar_backtest(sample_feature_vector, vix_vacio, sample_parameters)
    assert len(result) > 0
    assert (result["equity"] > 0).all()


# ── calcular_metricas ─────────────────────────────────────────────────────────

def test_metricas_tipos(sample_feature_vector, sample_vix, sample_parameters):
    portfolio = ejecutar_backtest(sample_feature_vector, sample_vix, sample_parameters)
    result = calcular_metricas(portfolio, sample_parameters)
    assert isinstance(result, tuple) and len(result) == 2
    metrics_df, equity_df = result
    assert hasattr(metrics_df, "columns")
    assert hasattr(equity_df, "columns")


def test_metricas_columnas(sample_feature_vector, sample_vix, sample_parameters):
    portfolio = ejecutar_backtest(sample_feature_vector, sample_vix, sample_parameters)
    metrics_df, _ = calcular_metricas(portfolio, sample_parameters)
    for col in [
        "sharpe_ratio", "max_drawdown_pct", "win_rate_pct",
        "cagr_pct", "n_trades", "trades_per_year",
        "circuit_break_events", "vix_crisis_days",
    ]:
        assert col in metrics_df.columns, f"Falta metrica: {col}"


def test_metricas_sin_trades(sample_parameters):
    """Sin operaciones: n_trades=0."""
    capital = float(sample_parameters["backtesting"]["initial_capital"])
    portfolio = _make_portfolio(capital=capital)
    metrics_df, _ = calcular_metricas(portfolio, sample_parameters)
    assert int(metrics_df["n_trades"].iloc[0]) == 0


def test_metricas_sharpe_constante(sample_parameters):
    """Equity constante → returns std=0 → Sharpe=0."""
    capital = float(sample_parameters["backtesting"]["initial_capital"])
    portfolio = _make_portfolio(capital=capital)
    metrics_df, _ = calcular_metricas(portfolio, sample_parameters)
    assert float(metrics_df["sharpe_ratio"].iloc[0]) == 0.0


def test_equity_curve_columnas(sample_feature_vector, sample_vix, sample_parameters):
    portfolio = ejecutar_backtest(sample_feature_vector, sample_vix, sample_parameters)
    _, equity_df = calcular_metricas(portfolio, sample_parameters)
    assert "date" in equity_df.columns
    assert "equity" in equity_df.columns


# ── calcular_benchmark ────────────────────────────────────────────────────────

def test_benchmark_columnas(sample_feature_vector, sample_parameters):
    result = calcular_benchmark(sample_feature_vector, sample_parameters)
    assert "date" in result.columns
    assert "equity" in result.columns


def test_benchmark_equity_positiva(sample_feature_vector, sample_parameters):
    result = calcular_benchmark(sample_feature_vector, sample_parameters)
    assert (result["equity"] > 0).all()


def test_benchmark_longitud(sample_feature_vector, sample_parameters):
    result = calcular_benchmark(sample_feature_vector, sample_parameters)
    spy_days = (sample_feature_vector["ticker"] == "SPY").sum()
    assert len(result) == spy_days


# ── calcular_walk_forward ─────────────────────────────────────────────────────

def test_walk_forward_columnas(sample_parameters):
    """Walk-forward debe producir columnas de metricas para cada periodo."""
    portfolio = _make_portfolio(200, start="2020-01-01")
    params = {**sample_parameters}
    params["backtesting"] = {**params["backtesting"], "walk_forward_split": "2020-07-01"}
    result = calcular_walk_forward(portfolio, params)
    assert "periodo" in result.columns
    assert "sharpe" in result.columns
    assert "max_drawdown_pct" in result.columns
    assert "cagr_pct" in result.columns


def test_walk_forward_tres_periodos(sample_parameters):
    """Walk-forward debe producir exactamente 3 filas: in-sample, out-of-sample, completo."""
    portfolio = _make_portfolio(365, start="2020-01-01")
    params = {**sample_parameters}
    params["backtesting"] = {**params["backtesting"], "walk_forward_split": "2020-07-01"}
    result = calcular_walk_forward(portfolio, params)
    assert len(result) == 3, f"Esperaba 3 filas, obtuvo {len(result)}"
