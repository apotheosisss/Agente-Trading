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


def obtener_vix(start_date: str, end_date: str) -> pd.DataFrame:
    """Descarga el índice de volatilidad VIX (^VIX) desde Yahoo Finance.

    El VIX es el "índice del miedo": mide la volatilidad implícita del S&P 500.
    - VIX < 20: mercado calmado, condiciones favorables para invertir
    - VIX 20-25: volatilidad moderada, precaución
    - VIX > 25: miedo elevado — suspender nuevas compras (``vix_fear_threshold``)
    - VIX > 35: pánico — liquidar posiciones (``vix_crisis_threshold``)

    Retorna DataFrame con DatetimeIndex (date) y columna ``vix``.
    """
    logger.info("Descargando VIX (^VIX) desde %s hasta %s", start_date, end_date)
    df = yf.download(
        tickers="^VIX",
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        logger.warning("VIX no disponible — usando valor por defecto 20.0")
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        return pd.DataFrame({"vix": 20.0}, index=dates)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    df.index.name = "date"
    result = df[["close"]].rename(columns={"close": "vix"})

    logger.info(
        "VIX descargado: %d filas | media=%.1f | max=%.1f | min=%.1f",
        len(result),
        float(result["vix"].mean()),
        float(result["vix"].max()),
        float(result["vix"].min()),
    )
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
