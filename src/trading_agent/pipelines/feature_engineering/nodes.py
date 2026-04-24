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


def calcular_indicadores_tecnicos(ohlcv: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Calcula RSI, MACD, Bandas de Bollinger, ATR y EMAs sobre datos OHLCV limpios."""
    df = ohlcv.copy()
    p = parameters

    rsi_period = int(p["rsi_period"])
    macd_fast = int(p["macd_fast"])
    macd_slow = int(p["macd_slow"])
    macd_signal_p = int(p["macd_signal"])
    bb_period = int(p["bb_period"])
    bb_std = float(p["bb_std"])

    # RSI
    df["rsi"] = _rsi(df["close"], rsi_period)

    # MACD
    ema_fast = _ema(df["close"], macd_fast)
    ema_slow = _ema(df["close"], macd_slow)
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = _ema(df["macd"], macd_signal_p)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bandas de Bollinger
    sma = df["close"].rolling(window=bb_period).mean()
    std = df["close"].rolling(window=bb_period).std()
    df["bb_mid"] = sma
    df["bb_upper"] = sma + bb_std * std
    df["bb_lower"] = sma - bb_std * std

    # ATR
    df["atr"] = _atr(df["high"], df["low"], df["close"])

    # EMAs
    df["ema_20"] = _ema(df["close"], 20)
    df["ema_50"] = _ema(df["close"], 50)

    df = df.dropna()
    logger.info(f"Indicadores tecnicos calculados: {len(df)} filas, {len(df.columns)} columnas")
    return df


def calcular_sentimiento(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Proxy de sentimiento basado en momentum de precio (tanh del retorno diario).

    Sustituye FinBERT hasta integrar el pipeline de ingesta de noticias (NewsAPI).
    Rango de salida: [-1.0, +1.0], donde positivo = momentum alcista.
    """
    daily_return = ohlcv["close"].pct_change()
    sentiment_score = daily_return.apply(
        lambda r: float(np.tanh(r * 10)) if pd.notna(r) else 0.0
    )

    result = pd.DataFrame({"sentiment_score": sentiment_score}, index=ohlcv.index)
    result = result.dropna()

    logger.info(
        f"Sentimiento calculado: {len(result)} filas, score medio={result['sentiment_score'].mean():.3f}"
    )
    return result


def ensamblar_vector_features(technical: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Combina features tecnicas y de sentimiento en un unico DataFrame alineado por fecha."""
    vector = technical.join(sentiment[["sentiment_score"]], how="left")
    vector["sentiment_score"] = vector["sentiment_score"].fillna(0.0)

    columnas_salida = [
        "open", "high", "low", "close", "volume",
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower",
        "ema_20", "ema_50", "atr",
        "sentiment_score",
    ]
    columnas_salida = [c for c in columnas_salida if c in vector.columns]
    vector = vector[columnas_salida]

    logger.info(f"Feature vector ensamblado: {len(vector)} filas, {len(vector.columns)} columnas")
    return vector
