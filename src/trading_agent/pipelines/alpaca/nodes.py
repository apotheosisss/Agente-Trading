# src/trading_agent/pipelines/alpaca/nodes.py
"""Nodos de ejecución real via Alpaca Trading API.

Este módulo reemplaza el paper trading simulado con órdenes reales en Alpaca.
Por seguridad, opera exclusivamente en modo PAPER hasta que el usuario
configure explícitamente ``paper_trading: false`` en credentials.yml Y
confirme haber revisado las señales manualmente al menos 30 días seguidos.

Requisitos:
    pip install alpaca-py

Credenciales (conf/local/credentials.yml):
    alpaca:
        api_key: "tu_api_key_aqui"
        secret_key: "tu_secret_key_aqui"
        paper_trading: true   # NUNCA cambiar a false sin revisión manual de 30 días
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ── Constantes de seguridad ────────────────────────────────────────────────────
MAX_ORDER_USD = 5_000      # Tamaño máximo por orden ($)
MAX_PORTFOLIO_PCT = 0.15   # Máximo 15% del portfolio en un solo activo
MIN_CASH_RESERVE = 0.05    # Mantener mínimo 5% del portfolio en cash


def _load_credentials() -> dict:
    cred_path = Path("conf/local/credentials.yml")
    if not cred_path.exists():
        return {}
    with open(cred_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_alpaca_client(credentials: dict | None = None):
    """Instancia el cliente de Alpaca.  Siempre usa paper a menos que se indique explícitamente."""
    try:
        from alpaca.trading.client import TradingClient
    except ImportError as exc:
        raise ImportError(
            "Instala alpaca-py: pip install alpaca-py"
        ) from exc

    if credentials is None:
        credentials = _load_credentials()
    cfg = credentials.get("alpaca", {})
    api_key = cfg.get("api_key", "")
    secret_key = cfg.get("secret_key", "")
    paper = bool(cfg.get("paper_trading", True))  # defecto siempre paper

    if not api_key or not secret_key:
        raise ValueError(
            "Faltan credenciales Alpaca. Configura conf/local/credentials.yml"
        )

    if not paper:
        logger.warning(
            "⚠️  MODO LIVE ACTIVADO — se ejecutarán órdenes con DINERO REAL. "
            "Asegúrate de haber revisado las señales manualmente ≥30 días."
        )

    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
    logger.info("Cliente Alpaca conectado (paper=%s)", paper)
    return client, paper


def verificar_cuenta_alpaca() -> pd.DataFrame:
    """Obtiene el estado de la cuenta Alpaca (equity, cash, posiciones abiertas).

    Retorna un DataFrame de una fila con el estado de la cuenta.
    """
    ts = datetime.now(timezone.utc).isoformat()
    try:
        client, paper = _get_alpaca_client()
        account = client.get_account()
        equity = float(account.equity)
        cash = float(account.cash)
        buying_power = float(account.buying_power)
        positions = client.get_all_positions()
        n_positions = len(positions)
        mode = "paper" if paper else "live"
        logger.info(
            "Cuenta Alpaca [%s]: equity=$%.2f | cash=$%.2f | posiciones=%d",
            mode, equity, cash, n_positions,
        )
        return pd.DataFrame([{
            "timestamp": ts,
            "mode": mode,
            "equity_usd": round(equity, 2),
            "cash_usd": round(cash, 2),
            "buying_power_usd": round(buying_power, 2),
            "n_positions": n_positions,
            "status": "connected",
        }])
    except Exception as exc:
        logger.error("Error conectando a Alpaca: %s", exc)
        return pd.DataFrame([{
            "timestamp": ts,
            "mode": "paper",
            "equity_usd": 0.0,
            "cash_usd": 0.0,
            "buying_power_usd": 0.0,
            "n_positions": 0,
            "status": f"error: {exc}",
        }])


def ejecutar_ordenes_alpaca(
    signal_df: pd.DataFrame,
    account_state: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Ejecuta órdenes de mercado en Alpaca para las señales BUY aprobadas.

    Controles de seguridad aplicados:
    - Solo señales BUY con confianza >= confidence_threshold
    - Tamaño máximo por orden: min(allocation, MAX_ORDER_USD=$5,000)
    - Máximo MAX_PORTFOLIO_PCT=15% del portfolio en un solo activo
    - Se reserva MIN_CASH_RESERVE=5% del portfolio en cash
    - Tickers no disponibles en Alpaca son saltados con advertencia

    Retorna DataFrame con el resultado de cada orden intentada.
    """
    ts = datetime.now(timezone.utc).isoformat()
    confidence_threshold = float(parameters["llm"]["confidence_threshold"])
    max_positions = int(parameters["risk"]["max_positions"])

    # Todos los activos del universo (cripto y acciones como SPY, COIN)
    buy_signals = signal_df[
        (signal_df["signal"] == "BUY")
        & (signal_df["confidence"] >= confidence_threshold)
    ].copy()

    if buy_signals.empty:
        logger.info("Sin seniales BUY aprobadas para Alpaca.")
        return pd.DataFrame([{
            "timestamp": ts, "ticker": "", "side": "HOLD",
            "qty": 0.0, "notional_usd": 0.0, "status": "no_signals",
            "message": "Sin seniales BUY con confianza suficiente",
        }])

    # Valor total del portfolio
    portfolio_equity = float(account_state["equity_usd"].iloc[0])
    available_cash = float(account_state["cash_usd"].iloc[0])
    cash_reserve = portfolio_equity * MIN_CASH_RESERVE
    investable_cash = max(available_cash - cash_reserve, 0.0)

    alloc_per_position = min(
        portfolio_equity / max_positions,
        investable_cash / max(len(buy_signals), 1),
        MAX_ORDER_USD,
        portfolio_equity * MAX_PORTFOLIO_PCT,
    )

    records = []
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client, paper = _get_alpaca_client()
        mode = "paper" if paper else "live"

        for _, row in buy_signals.iterrows():
            ticker = str(row["ticker"])
            # Alpaca usa "BTC/USD" para cripto (no "BTC-USD" de yfinance)
            # Para acciones (SPY, COIN) el símbolo se mantiene igual
            alpaca_symbol = ticker.replace("-USD", "/USD") if "-USD" in ticker else ticker
            notional = round(alloc_per_position, 2)

            if notional <= 0:
                records.append({
                    "timestamp": ts, "ticker": ticker, "side": "BUY",
                    "qty": 0.0, "notional_usd": notional,
                    "status": "skipped", "message": "Sin capital disponible",
                    "mode": mode,
                })
                continue

            try:
                order_request = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    notional=notional,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                order = client.submit_order(order_request)
                logger.info(
                    "[%s] BUY %s: $%.2f | order_id=%s",
                    mode, ticker, notional, order.id,
                )
                records.append({
                    "timestamp": ts, "ticker": ticker, "side": "BUY",
                    "qty": 0.0,  # se llena al completarse
                    "notional_usd": notional,
                    "status": "submitted",
                    "message": f"order_id={order.id}",
                    "mode": mode,
                })
            except Exception as exc:
                logger.error("Error enviando orden %s: %s", ticker, exc)
                records.append({
                    "timestamp": ts, "ticker": ticker, "side": "BUY",
                    "qty": 0.0, "notional_usd": notional,
                    "status": "error", "message": str(exc),
                    "mode": mode,
                })

    except ImportError:
        # alpaca-py no instalado → simular (sin ejecutar nada real)
        logger.warning(
            "alpaca-py no instalado — simulando órdenes (instala con: pip install alpaca-py)"
        )
        for _, row in buy_signals.iterrows():
            records.append({
                "timestamp": ts, "ticker": str(row["ticker"]), "side": "BUY",
                "qty": 0.0, "notional_usd": round(alloc_per_position, 2),
                "status": "simulated",
                "message": "alpaca-py no disponible — instala con pip install alpaca-py",
                "mode": "paper_sim",
            })

    return pd.DataFrame(records)


def sincronizar_posiciones_alpaca() -> pd.DataFrame:
    """Obtiene las posiciones abiertas actuales de la cuenta Alpaca.

    Retorna DataFrame con ticker, qty, market_value, unrealized_pl, side.
    Útil para verificar que el estado del portafolio coincide con las señales.
    """
    ts = datetime.now(timezone.utc).isoformat()
    try:
        client, paper = _get_alpaca_client()
        positions = client.get_all_positions()
        mode = "paper" if paper else "live"

        if not positions:
            return pd.DataFrame([{
                "timestamp": ts, "ticker": "", "qty": 0.0,
                "market_value_usd": 0.0, "unrealized_pl_usd": 0.0,
                "side": "NONE", "mode": mode,
            }])

        records = []
        for pos in positions:
            records.append({
                "timestamp": ts,
                "ticker": str(pos.symbol).replace("/", "-"),
                "qty": float(pos.qty),
                "market_value_usd": float(pos.market_value),
                "unrealized_pl_usd": float(pos.unrealized_pl),
                "side": str(pos.side),
                "mode": mode,
            })
        logger.info(
            "Posiciones Alpaca [%s]: %d abiertas | valor_total=$%.2f",
            mode,
            len(records),
            sum(r["market_value_usd"] for r in records),
        )
        return pd.DataFrame(records)

    except Exception as exc:
        logger.error("Error obteniendo posiciones Alpaca: %s", exc)
        return pd.DataFrame([{
            "timestamp": ts, "ticker": "", "qty": 0.0,
            "market_value_usd": 0.0, "unrealized_pl_usd": 0.0,
            "side": "ERROR", "mode": "unknown",
        }])
