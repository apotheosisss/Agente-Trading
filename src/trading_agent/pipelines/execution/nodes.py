# src/trading_agent/pipelines/execution/nodes.py

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


def verificar_riesgo(signal_df: pd.DataFrame, parameters: dict) -> dict:
    """Evalúa si la señal supera los controles de riesgo configurados.

    Retorna dict con campo 'approved' (bool) y 'reason' (str).
    """
    signal = signal_df.iloc[0].to_dict()
    signal_type = str(signal.get("signal", "HOLD"))
    confidence = float(signal.get("confidence", 0.0))

    confidence_threshold = float(parameters["llm"]["confidence_threshold"])
    max_position_pct = float(parameters["risk"]["max_position_pct"])

    if signal_type == "HOLD":
        logger.info("Senial HOLD - sin accion requerida")
        return {
            "approved": False,
            "reason": "Senial HOLD - sin accion requerida",
            "signal": signal_type,
            "confidence": confidence,
        }

    if confidence < confidence_threshold:
        logger.info(
            f"Senial bloqueada: confianza {confidence:.2f} < umbral {confidence_threshold:.2f}"
        )
        return {
            "approved": False,
            "reason": f"Confianza insuficiente ({confidence:.2f} < {confidence_threshold:.2f})",
            "signal": signal_type,
            "confidence": confidence,
        }

    logger.info(
        f"Senial aprobada: {signal_type} | confianza={confidence:.2f} | "
        f"max_posicion={max_position_pct:.0%}"
    )
    return {
        "approved": True,
        "reason": "Controles de riesgo superados",
        "signal": signal_type,
        "confidence": confidence,
        "max_position_pct": max_position_pct,
    }


def enviar_orden(risk_result: dict, signal_df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Registra la orden en modo paper trading si el riesgo fue aprobado.

    No conecta con broker real — toda orden queda en modo simulacion.
    """
    ts = datetime.now(timezone.utc).isoformat()

    if not risk_result.get("approved", False):
        logger.info(f"Orden no ejecutada: {risk_result.get('reason', '')}")
        return pd.DataFrame(
            [
                {
                    "status": "REJECTED",
                    "reason": risk_result.get("reason", ""),
                    "signal": risk_result.get("signal", ""),
                    "ticker": "",
                    "order_size_usd": 0.0,
                    "confidence": risk_result.get("confidence", 0.0),
                    "timestamp": ts,
                    "mode": "paper",
                }
            ]
        )

    signal = signal_df.iloc[0].to_dict()
    ticker = str(signal.get("ticker", ""))
    signal_type = risk_result["signal"]
    confidence = risk_result["confidence"]
    max_position_pct = risk_result.get("max_position_pct", parameters["risk"]["max_position_pct"])
    initial_capital = float(parameters["backtesting"]["initial_capital"])
    order_size = initial_capital * max_position_pct

    logger.info(
        f"PAPER TRADE: {signal_type} {ticker} | tamano=${order_size:,.2f} | "
        f"confianza={confidence:.2f}"
    )

    return pd.DataFrame(
        [
            {
                "status": "FILLED",
                "reason": "Orden ejecutada en modo paper",
                "signal": signal_type,
                "ticker": ticker,
                "order_size_usd": round(order_size, 2),
                "confidence": confidence,
                "timestamp": ts,
                "mode": "paper",
            }
        ]
    )


def actualizar_portafolio(execution_record: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Genera el estado actualizado del portafolio tras la ultima orden.

    Para paper trading: calcula posiciones y capital basado en la orden registrada.
    """
    order = execution_record.iloc[0].to_dict()
    initial_capital = float(parameters["backtesting"]["initial_capital"])
    stop_loss_pct = float(parameters["risk"]["stop_loss_pct"])
    take_profit_pct = float(parameters["risk"]["take_profit_pct"])

    if order["status"] == "REJECTED":
        portfolio_state = {
            "cash": initial_capital,
            "position": 0.0,
            "position_value": 0.0,
            "total_value": initial_capital,
            "last_signal": order["signal"],
            "last_order_status": "REJECTED",
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "timestamp": order["timestamp"],
            "mode": "paper",
        }
    else:
        order_size = float(order["order_size_usd"])
        signal = order["signal"]

        if signal == "BUY":
            cash = initial_capital - order_size
            position_value = order_size
        else:  # SELL
            cash = initial_capital
            position_value = 0.0

        portfolio_state = {
            "cash": round(cash, 2),
            "position": round(order_size, 2) if signal == "BUY" else 0.0,
            "position_value": round(position_value, 2),
            "total_value": round(cash + position_value, 2),
            "last_signal": signal,
            "last_order_status": "FILLED",
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "timestamp": order["timestamp"],
            "mode": "paper",
        }

    logger.info(
        f"Portafolio actualizado: total=${portfolio_state['total_value']:,.2f} | "
        f"cash=${portfolio_state['cash']:,.2f} | posicion=${portfolio_state['position_value']:,.2f}"
    )

    return pd.DataFrame([portfolio_state])
