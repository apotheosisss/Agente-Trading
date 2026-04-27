# src/trading_agent/pipelines/feature_engineering/nodes.py

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _indicadores_single(grupo: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Calcula todos los indicadores para un único ticker."""
    df = grupo.copy()

    rsi_period = int(p["rsi_period"])
    macd_fast = int(p["macd_fast"])
    macd_slow = int(p["macd_slow"])
    macd_signal_p = int(p["macd_signal"])
    bb_period = int(p["bb_period"])
    bb_std = float(p["bb_std"])
    ema_200_period = int(p.get("ema_200", 200))

    df["rsi"] = _rsi(df["close"], rsi_period)

    ema_fast = _ema(df["close"], macd_fast)
    ema_slow = _ema(df["close"], macd_slow)
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = _ema(df["macd"], macd_signal_p)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    sma = df["close"].rolling(window=bb_period).mean()
    std = df["close"].rolling(window=bb_period).std()
    df["bb_mid"] = sma
    df["bb_upper"] = sma + bb_std * std
    df["bb_lower"] = sma - bb_std * std

    df["atr"] = _atr(df["high"], df["low"], df["close"])
    df["ema_20"] = _ema(df["close"], 20)
    df["ema_50"] = _ema(df["close"], 50)
    df["ema_200"] = _ema(df["close"], ema_200_period)

    # ── Momentum de precio ─────────────────────────────────────────────────
    # Retorno acumulado 90d (trimestral) y 252d (anual) — señal primaria de
    # ranking en la estrategia Momentum Concentrado.
    # Usar fillna(0.0) para no propagar NaNs extra al inicio de la serie.
    df["momentum_90d"] = df["close"].pct_change(90).fillna(0.0)
    df["momentum_252d"] = df["close"].pct_change(252).fillna(0.0)

    return df.dropna()


def calcular_indicadores_tecnicos(ohlcv: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Calcula RSI, MACD, Bandas de Bollinger, ATR y EMAs por ticker.

    Nota: usa loop explícito en lugar de groupby.apply para garantizar
    compatibilidad con pandas 3.0+ (donde include_groups=False es el default
    y la columna groupby es excluida de los grupos).
    """
    frames = []
    for ticker, grupo in ohlcv.sort_index().groupby("ticker"):
        grupo = grupo.copy()
        grupo["ticker"] = ticker  # garantizar columna ticker en el grupo
        frames.append(_indicadores_single(grupo, parameters))

    if not frames:
        raise ValueError("No se obtuvieron indicadores para ningún ticker")

    result = pd.concat(frames).sort_index()
    n_tickers = result["ticker"].nunique()
    logger.info(
        f"Indicadores calculados: {len(result)} filas, {n_tickers} tickers, "
        f"{len(result.columns)} columnas"
    )
    return result


def calcular_sentimiento(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Proxy de sentimiento: tanh(retorno_diario × 10) por ticker."""
    frames = []
    for ticker, grupo in ohlcv.sort_index().groupby("ticker"):
        daily_return = grupo["close"].pct_change()
        score = daily_return.apply(
            lambda r: float(np.tanh(r * 10)) if pd.notna(r) else 0.0
        )
        frame = pd.DataFrame(
            {"ticker": ticker, "sentiment_score": score},
            index=grupo.index,
        ).dropna()
        frames.append(frame)

    if not frames:
        raise ValueError("No se obtuvo sentimiento para ningún ticker")

    result = pd.concat(frames).sort_index()
    logger.info(
        f"Sentimiento calculado: {len(result)} filas, "
        f"score medio={result['sentiment_score'].mean():.3f}"
    )
    return result


def ensamblar_vector_features(technical: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Combina features técnicas y sentimiento alineando por (date, ticker)."""
    tech = technical.set_index("ticker", append=True)
    sent = sentiment.set_index("ticker", append=True)[["sentiment_score"]]

    vector = tech.join(sent, how="left")
    vector["sentiment_score"] = vector["sentiment_score"].fillna(0.0)
    vector = vector.reset_index(level="ticker")

    columnas_salida = [
        "ticker",
        "open", "high", "low", "close", "volume",
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower",
        "ema_20", "ema_50", "ema_200", "atr",
        "momentum_90d", "momentum_252d",
        "sentiment_score",
    ]
    columnas_salida = [c for c in columnas_salida if c in vector.columns]
    vector = vector[columnas_salida].sort_index()

    logger.info(
        f"Feature vector ensamblado: {len(vector)} filas, "
        f"{vector['ticker'].nunique()} tickers, {len(vector.columns)} columnas"
    )
    return vector
