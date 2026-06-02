"""Nodos Kedro del pipeline TSMOM long/short (producción).

Flujo: clean_ohlcv -> pesos objetivo (con signo) -> órdenes long/short.
Reemplaza la lógica long-only/top-N por la estrategia validada (ver
OPTIMIZACION_RESULTADOS.md). Larga y corta sobre universo multi-activo.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trading_agent.pipelines.tsmom import strategy

logger = logging.getLogger(__name__)


def _cfg_from_params(parameters: dict) -> dict:
    """Lee la sección 'tsmom' de parameters.yml sobre los valores por defecto."""
    return dict(parameters.get("tsmom", {}))


def calcular_pesos_tsmom(clean_ohlcv: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Calcula los pesos objetivo (con signo) por activo.

    Devuelve la fila del ÚLTIMO día: una fila por ticker con el peso objetivo
    (positivo=largo, negativo=corto) y su lectura de señal. Es lo que consume
    la ejecución para hoy.
    """
    cfg = _cfg_from_params(parameters)
    px = strategy.to_wide_close(clean_ohlcv).ffill()
    w = strategy.weights(px, cfg)
    sig = strategy.signal(px, cfg)

    last = w.iloc[-1]
    last_sig = sig.iloc[-1]
    date = w.index[-1]
    rows = []
    for ticker in w.columns:
        wt = float(last[ticker])
        rows.append({
            "ticker": ticker,
            "target_weight": round(wt, 4),
            "direction": "LONG" if wt > 1e-6 else "SHORT" if wt < -1e-6 else "FLAT",
            "signal": round(float(last_sig[ticker]), 3),
            "date": pd.Timestamp(date).normalize(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    df = pd.DataFrame(rows).sort_values("target_weight", ascending=False).reset_index(drop=True)
    gross = float(df["target_weight"].abs().sum())
    n_long = int((df["target_weight"] > 0).sum())
    n_short = int((df["target_weight"] < 0).sum())
    logger.info(
        "TSMOM pesos @ %s: %d largos | %d cortos | exposición bruta %.2fx",
        pd.Timestamp(date).date(), n_long, n_short, gross,
    )
    return df


def generar_reporte_tsmom(target_weights: pd.DataFrame) -> str:
    lines = ["=== TSMOM long/short — pesos objetivo de hoy ==="]
    gross = float(target_weights["target_weight"].abs().sum())
    net = float(target_weights["target_weight"].sum())
    lines.append(f"  Exposición bruta: {gross:.2f}x | neta: {net:+.2f}x")
    for _, r in target_weights.iterrows():
        if abs(r["target_weight"]) < 1e-6:
            continue
        lines.append(
            f"    [{r['ticker']:>8}] {r['direction']:>5} peso={r['target_weight']:+.3f} "
            f"(señal {r['signal']:+.2f})"
        )
    return "\n".join(lines)


def generar_ordenes_tsmom(
    target_weights: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """Convierte pesos objetivo en órdenes (paper) hacia la posición objetivo.

    target_position_usd = capital * target_weight  (negativo = corto).
    Asume rebalanceo completo a objetivo (sin estado previo de cartera): cada
    fila es la posición objetivo a alcanzar. En vivo, el adaptador Alpaca debe
    enviar la orden delta = objetivo - posición_actual.
    """
    capital = float(parameters["backtesting"]["initial_capital"])
    ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, r in target_weights.iterrows():
        wt = float(r["target_weight"])
        if abs(wt) < 1e-6:
            continue
        target_usd = capital * wt
        rows.append({
            "ticker": r["ticker"],
            "side": "BUY" if wt > 0 else "SELL_SHORT",
            "target_position_usd": round(target_usd, 2),
            "target_weight": round(wt, 4),
            "mode": "paper",
            "timestamp": ts,
        })
    if not rows:
        rows = [{"ticker": "", "side": "HOLD", "target_position_usd": 0.0,
                 "target_weight": 0.0, "mode": "paper", "timestamp": ts}]
    df = pd.DataFrame(rows)
    logger.info(
        "TSMOM órdenes: %d posiciones objetivo | bruto $%.0f",
        len(df[df["side"] != "HOLD"]),
        float(df["target_position_usd"].abs().sum()),
    )
    return df


def validar_tsmom(clean_ohlcv: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Backtest de validación de la estrategia (full + OOS). No opera; informa."""
    cfg = _cfg_from_params(parameters)
    split = pd.Timestamp(str(parameters["backtesting"].get("walk_forward_split", "2022-01-01")))
    px = strategy.to_wide_close(clean_ohlcv).ffill()
    r = strategy.backtest(px, cfg)
    full = strategy.perf_metrics(r)
    oos = strategy.perf_metrics(r[r.index >= split])
    out = pd.DataFrame([
        {"periodo": "FULL", **full},
        {"periodo": "OOS", **oos},
    ])
    logger.info(
        "TSMOM validación | FULL Sharpe=%.2f CAGR=%.1f%% MaxDD=%.1f%% | OOS Sharpe=%.2f",
        full["sharpe"], full["cagr_pct"], full["maxdd_pct"], oos["sharpe"],
    )
    return out
