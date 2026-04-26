# src/trading_agent/pipelines/ingestion/nodes.py

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def obtener_datos_mercado(universe: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Descarga datos OHLCV para todos los tickers del universo desde Yahoo Finance.

    Retorna un DataFrame plano con columna 'ticker' e índice DatetimeIndex ('date').
    """
    frames = []
    for ticker in universe:
        logger.info(f"Descargando {ticker} desde {start_date} hasta {end_date}")
        df = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            logger.warning(f"Sin datos para {ticker} — omitido")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]

        df.index.name = "date"
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.insert(0, "ticker", ticker)
        frames.append(df)

    if not frames:
        raise ValueError("No se obtuvieron datos para ningún ticker del universo")

    result = pd.concat(frames).sort_index()
    logger.info(f"Datos descargados: {len(result)} filas, {result['ticker'].nunique()} tickers")
    return result


def validar_datos_mercado(df: pd.DataFrame) -> pd.DataFrame:
    """Valida integridad del DataFrame OHLCV multi-ticker."""
    columnas_requeridas = ["ticker", "open", "high", "low", "close", "volume"]
    faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes: {faltantes}")

    resultados = []
    for ticker, grupo in df.groupby("ticker"):
        nans = grupo.isnull().sum().sum()
        if nans > 0:
            logger.warning(f"{ticker}: {nans} valores NaN — eliminados")
            grupo = grupo.dropna()

        precios = ["open", "high", "low", "close"]
        if (grupo[precios] <= 0).any().any():
            raise ValueError(f"{ticker}: precios negativos o cero detectados")

        if not grupo.index.is_monotonic_increasing:
            logger.warning(f"{ticker}: índice desordenado — reordenando")
            grupo = grupo.sort_index()

        resultados.append(grupo)

    result = pd.concat(resultados).sort_index()
    logger.info(f"Validación completa: {len(result)} filas limpias, {result['ticker'].nunique()} tickers")
    return result
