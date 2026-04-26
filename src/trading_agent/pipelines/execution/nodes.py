# src/trading_agent/pipelines/execution/nodes.py

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


def verificar_riesgo(signal_df: pd.DataFrame, parameters: dict) -> dict:
    """Filtra el ranking multi-ticker y aprueba solo las señales BUY con
    confianza suficiente.

    Retorna un dict con los tickers aprobados y el contexto de riesgo.
    """
    confidence_threshold = float(parameters["llm"]["confidence_threshold"])
    max_position_pct = float(parameters["risk"]["max_position_pct"])

    buy_signals = signal_df[signal_df["signal"] == "BUY"].copy()
    approved_tickers = buy_signals[
        buy_signals["confidence"] >= confidence_threshold
    ]["ticker"].tolist()

    if not approved_tickers:
        reason = (
            "Sin seniales BUY con confianza suficiente "
            f"(umbral={confidence_threshold:.2f})"
        )
        logger.info("Sin órdenes aprobadas: %s", reason)
        return {
            "approved": False,
            "reason": reason,
            "approved_tickers": [],
            "max_position_pct": max_position_pct,
        }

    logger.info(
        "Señales aprobadas: %s | umbral=%.2f | max_posicion=%.0f%%",
        approved_tickers,
        confidence_threshold,
        max_position_pct * 100,
    )
    return {
        "approved": True,
        "reason": "Controles de riesgo superados",
        "approved_tickers": approved_tickers,
        "max_position_pct": max_position_pct,
    }


def enviar_orden(risk_result: dict, signal_df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Registra órdenes paper trading para cada ticker aprobado.

    Retorna un DataFrame con una fila por ticker procesado.
    """
    ts = datetime.now(timezone.utc).isoformat()
    initial_capital = float(parameters["backtesting"]["initial_capital"])
    max_positions = int(parameters["risk"]["max_positions"])

    if not risk_result.get("approved", False):
        logger.info("Órdenes rechazadas: %s", risk_result.get("reason", ""))
        return pd.DataFrame(
            [
                {
                    "status": "REJECTED",
                    "reason": risk_result.get("reason", ""),
                    "signal": "HOLD",
                    "ticker": "",
                    "order_size_usd": 0.0,
                    "confidence": 0.0,
                    "timestamp": ts,
                    "mode": "paper",
                }
            ]
        )

    approved_tickers = risk_result["approved_tickers"]
    max_position_pct = float(risk_result.get("max_position_pct", 0.33))
    # Cada posición recibe 1/max_positions del capital total
    order_size = initial_capital / max_positions

    records = []
    for ticker in approved_tickers:
        row = signal_df[signal_df["ticker"] == ticker]
        confidence = float(row["confidence"].iloc[0]) if not row.empty else 0.0
        logger.info(
            "PAPER TRADE: BUY %s | tamaño=$%.2f | confianza=%.2f",
            ticker,
            order_size,
            confidence,
        )
        records.append(
            {
                "status": "FILLED",
                "reason": "Orden ejecutada en modo paper",
                "signal": "BUY",
                "ticker": ticker,
                "order_size_usd": round(order_size, 2),
                "confidence": confidence,
                "timestamp": ts,
                "mode": "paper",
            }
        )

    return pd.DataFrame(records)


def actualizar_portafolio(execution_record: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Genera el estado del portafolio multi-posición tras las últimas órdenes.

    Retorna una fila por posición activa (o una fila REJECTED si no hay órdenes).
    """
    initial_capital = float(parameters["backtesting"]["initial_capital"])
    stop_loss_atr_mult = float(parameters["risk"]["stop_loss_atr_mult"])
    ts = datetime.now(timezone.utc).isoformat()

    filled = execution_record[execution_record["status"] == "FILLED"]

    if filled.empty:
        order = execution_record.iloc[0].to_dict()
        logger.info("Portafolio sin cambios: %s", order.get("reason", ""))
        return pd.DataFrame(
            [
                {
                    "ticker": "",
                    "cash": initial_capital,
                    "position_value": 0.0,
                    "total_value": initial_capital,
                    "last_signal": order.get("signal", "HOLD"),
                    "last_order_status": "REJECTED",
                    "stop_loss_atr_mult": stop_loss_atr_mult,
                    "timestamp": ts,
                    "mode": "paper",
                }
            ]
        )

    total_invested = float(filled["order_size_usd"].sum())
    cash_remaining = max(initial_capital - total_invested, 0.0)
    total_value = cash_remaining + total_invested

    records = []
    for _, order in filled.iterrows():
        records.append(
            {
                "ticker": order["ticker"],
                "cash": round(cash_remaining, 2),
                "position_value": round(float(order["order_size_usd"]), 2),
                "total_value": round(total_value, 2),
                "last_signal": order["signal"],
                "last_order_status": "FILLED",
                "stop_loss_atr_mult": stop_loss_atr_mult,
                "timestamp": order["timestamp"],
                "mode": "paper",
            }
        )

    logger.info(
        "Portafolio actualizado: %d posiciones | total=$%.2f | cash=$%.2f",
        len(records),
        total_value,
        cash_remaining,
    )
    return pd.DataFrame(records)
