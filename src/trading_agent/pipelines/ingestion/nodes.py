# src/trading_agent/pipelines/ingestion/nodes.py

import pandas as pd
import yfinance as yf
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def obtener_datos_mercado(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Descarga datos OHLCV desde Yahoo Finance."""
    logger.info(f"Descargando datos de {ticker} desde {start_date} hasta {end_date}")
    
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}")

    # Aplanar columnas multi-index si existen
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]]

    logger.info(f"Datos descargados: {len(df)} filas")
    return df


def validar_datos_mercado(df: pd.DataFrame) -> pd.DataFrame:
    """Valida integridad del DataFrame OHLCV."""
    columnas_requeridas = ["open", "high", "low", "close", "volume"]
    
    # Verificar columnas
    faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes: {faltantes}")

    # Reporte de NaN antes de limpiar
    nans = df.isnull().sum().sum()
    if nans > 0:
        logger.warning(f"Se encontraron {nans} valores NaN — serán eliminados")
        df = df.dropna()

    # Validar precios positivos
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Se detectaron precios negativos o cero")

    # Validar orden cronológico
    if not df.index.is_monotonic_increasing:
        logger.warning("Índice no ordenado — reordenando")
        df = df.sort_index()

    logger.info(f"Validación completa: {len(df)} filas limpias")
    return df