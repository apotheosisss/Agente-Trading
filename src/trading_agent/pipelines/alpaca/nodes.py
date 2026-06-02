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

    # Todos los activos del universo son ejecutables en Polymarket
    buy_signals = signal_df[
        (signal_df["signal"] == "BUY")
        & (signal_df["confidence"] >= confidence_threshold)
    ].sort_values("confidence", ascending=False).head(max_positions).copy()

    sell_signals = signal_df[
        signal_df["signal"] == "SELL"
    ]["ticker"].tolist()

    records = []
    try:
        from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client, paper = _get_alpaca_client()
        mode = "paper" if paper else "live"

        # ── Normalizar posiciones abiertas ────────────────────────────────
        open_positions = client.get_all_positions()
        # Mapa: ticker_yfinance -> objeto posición Alpaca
        held_map = {}
        for p in open_positions:
            sym = str(p.symbol)
            variants = {sym, sym.replace("/", "-")}
            if sym.endswith("USD") and "/" not in sym and "-" not in sym:
                variants.add(sym[:-3] + "-USD")
            for v in variants:
                held_map[v] = p
        held_symbols = set(held_map.keys())
        logger.info(
            "Posiciones en cartera: %d | %s",
            len(open_positions), {p.symbol for p in open_positions} or "ninguna"
        )

        # Top tickers objetivo hoy
        target_tickers = set(buy_signals["ticker"].tolist())

        # ── 1. VENTAS: SELL explícito + rotación (fuera del top-N) ──────────
        tickers_a_vender = set(sell_signals)
        for ticker, pos in held_map.items():
            if ticker not in held_symbols:
                continue
            # Rotar: si la posición ya no está en el top-N de hoy, venderla
            if ticker not in target_tickers and ticker not in tickers_a_vender:
                tickers_a_vender.add(ticker)
                logger.info("%s salio del top-%d — rotando", ticker, max_positions)

        for ticker in tickers_a_vender:
            if ticker not in held_map:
                continue
            pos = held_map[ticker]
            alpaca_symbol = ticker.replace("-USD", "/USD") if "-USD" in ticker else ticker
            razon = "Señal SELL" if ticker in sell_signals else f"Rotacion — fuera del top-{max_positions}"
            try:
                client.close_position(alpaca_symbol)
                logger.info("[%s] SELL %s — %s", mode, ticker, razon)
                records.append({
                    "timestamp": ts, "ticker": ticker, "side": "SELL",
                    "qty": float(pos.qty), "notional_usd": float(pos.market_value),
                    "status": "submitted", "message": razon,
                    "mode": mode,
                })
                held_symbols.discard(ticker)
            except Exception as exc:
                logger.error("Error cerrando posición %s: %s", ticker, exc)
                records.append({
                    "timestamp": ts, "ticker": ticker, "side": "SELL",
                    "qty": 0.0, "notional_usd": 0.0,
                    "status": "error", "message": str(exc),
                    "mode": mode,
                })

        # ── 2. COMPRAS: slots libres tras ventas ──────────────────────────
        sold_tickers = {r["ticker"] for r in records
                        if r["side"] == "SELL" and r["status"] == "submitted"}
        n_open = len(open_positions) - len(sold_tickers)
        slots_available = max(max_positions - n_open, 0)

        if buy_signals.empty or slots_available == 0:
            if not buy_signals.empty and slots_available == 0:
                logger.info("Portfolio lleno (%d/%d) — sin compras.", n_open, max_positions)
            else:
                logger.info("Sin seniales BUY aprobadas para Alpaca.")
            if not records:
                records.append({
                    "timestamp": ts, "ticker": "", "side": "HOLD",
                    "qty": 0.0, "notional_usd": 0.0, "status": "no_signals",
                    "message": f"Portfolio {n_open}/{max_positions} — sin accion",
                })
            return pd.DataFrame(records)

        # Valor total del portfolio
        portfolio_equity = float(account_state["equity_usd"].iloc[0])
        available_cash = float(account_state["cash_usd"].iloc[0])
        cash_reserve = portfolio_equity * MIN_CASH_RESERVE
        investable_cash = max(available_cash - cash_reserve, 0.0)

        # Solo comprar los slots disponibles
        buy_signals = buy_signals.head(slots_available)
        alloc_per_position = min(
            portfolio_equity / max_positions,
            investable_cash / max(len(buy_signals), 1),
            MAX_ORDER_USD,
            portfolio_equity * MAX_PORTFOLIO_PCT,
        )

        for _, row in buy_signals.iterrows():
            ticker = str(row["ticker"])
            alpaca_symbol = ticker.replace("-USD", "/USD") if "-USD" in ticker else ticker
            notional = round(alloc_per_position, 2)

            # Saltar si ya tenemos posición abierta en este activo
            if ticker in held_symbols:
                logger.info("Ya existe posición en %s — orden omitida.", ticker)
                records.append({
                    "timestamp": ts, "ticker": ticker, "side": "BUY",
                    "qty": 0.0, "notional_usd": 0.0,
                    "status": "skipped", "message": "Posición ya abierta",
                    "mode": mode,
                })
                continue

            if notional <= 0:
                records.append({
                    "timestamp": ts, "ticker": ticker, "side": "BUY",
                    "qty": 0.0, "notional_usd": notional,
                    "status": "skipped", "message": "Sin capital disponible",
                    "mode": mode,
                })
                continue

            try:
                # Cripto usa GTC (mercado 24/7), acciones usan DAY
                is_crypto = "/" in alpaca_symbol
                tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY
                order_request = MarketOrderRequest(
                    symbol=alpaca_symbol,
                    notional=notional,
                    side=OrderSide.BUY,
                    time_in_force=tif,
                )
                order = client.submit_order(order_request)
                logger.info(
                    "[%s] BUY %s: $%.2f | order_id=%s",
                    mode, ticker, notional, order.id,
                )
                records.append({
                    "timestamp": ts, "ticker": ticker, "side": "BUY",
                    "qty": 0.0,
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


def _to_alpaca_symbol(ticker: str) -> tuple[str, bool]:
    """yfinance -> símbolo Alpaca. Devuelve (símbolo, es_cripto)."""
    if ticker.endswith("-USD"):
        return ticker.replace("-USD", "/USD"), True
    return ticker, False


def ejecutar_tsmom_alpaca(
    tsmom_weights: pd.DataFrame,
    account_state: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Ejecuta la estrategia TSMOM long/short en Alpaca mediante órdenes DELTA.

    Para cada activo calcula la posición objetivo en USD (equity * peso, normalizado
    a una exposición bruta segura) y envía la orden necesaria para pasar de la
    posición ACTUAL a la OBJETIVO. Soporta largos y cortos.

    Seguridad:
    - Siempre paper salvo paper_trading:false explícito (heredado de _get_alpaca_client).
    - Exposición bruta objetivo configurable (``tsmom.live_gross``, def. 1.0 = sin
      apalancamiento) para empezar conservador en paper.
    - Cap por posición (``tsmom.live_max_position_pct``, def. 0.30).
    - Órdenes por debajo de ``tsmom.min_order_usd`` (def. $25) se omiten.
    - Cripto no admite corto en Alpaca: si el peso es negativo, se cierra a 0.
    - Cada orden va en su propio try/except: un fallo no detiene el resto.
    """
    ts = datetime.now(timezone.utc).isoformat()
    cfg = parameters.get("tsmom", {})
    live_gross = float(cfg.get("live_gross", 1.0))
    max_pos_pct = float(cfg.get("live_max_position_pct", 0.30))
    min_order = float(cfg.get("min_order_usd", 25.0))

    records: list[dict] = []

    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client, paper = _get_alpaca_client()
        mode = "paper" if paper else "live"
        equity = float(account_state["equity_usd"].iloc[0])
        if equity <= 0:
            logger.error("Equity no disponible (%.2f) — abortando ejecución TSMOM", equity)
            return pd.DataFrame([{
                "timestamp": ts, "ticker": "", "side": "ABORT",
                "target_usd": 0.0, "delta_usd": 0.0, "status": "no_equity",
                "message": "account_state sin equity", "mode": mode,
            }])

        # ── Normalizar pesos a exposición bruta segura y cap por posición ────
        w = tsmom_weights.set_index("ticker")["target_weight"].astype(float)
        gross_raw = w.abs().sum()
        scale = (live_gross / gross_raw) if gross_raw > 0 else 0.0
        w = (w * scale).clip(-max_pos_pct, max_pos_pct)
        target_usd = (w * equity)

        # ── Posiciones actuales (market_value firmado) ───────────────────────
        cur_val: dict[str, float] = {}
        for p in client.get_all_positions():
            sym = str(p.symbol)               # p.ej. "AAPL" o "BTC/USD"
            yf = sym.replace("/USD", "-USD") if "/USD" in sym else sym
            cur_val[yf] = float(p.market_value)  # negativo si corto

        symbols = set(target_usd.index) | set(cur_val.keys())
        logger.info(
            "TSMOM Alpaca [%s]: equity=$%.0f | bruto objetivo=%.2fx | %d símbolos",
            mode, equity, live_gross, len(symbols),
        )

        for yf_sym in sorted(symbols):
            alpaca_sym, is_crypto = _to_alpaca_symbol(yf_sym)
            desired = float(target_usd.get(yf_sym, 0.0))
            if is_crypto and desired < 0:
                desired = 0.0   # sin cortos en cripto
            current = float(cur_val.get(yf_sym, 0.0))

            flip = (current != 0.0) and (desired * current < 0)
            tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY

            try:
                # Caso 1: cambio de signo (de largo a corto o viceversa) → cerrar primero
                if flip:
                    client.close_position(alpaca_sym)
                    records.append({
                        "timestamp": ts, "ticker": yf_sym, "side": "CLOSE",
                        "target_usd": round(desired, 2), "delta_usd": round(-current, 2),
                        "status": "submitted", "message": "cierre previo a flip", "mode": mode,
                    })
                    current = 0.0

                # Caso 2: objetivo ~0 y hay posición → cerrar
                if abs(desired) < min_order and abs(current) >= min_order and not flip:
                    client.close_position(alpaca_sym)
                    records.append({
                        "timestamp": ts, "ticker": yf_sym, "side": "CLOSE",
                        "target_usd": 0.0, "delta_usd": round(-current, 2),
                        "status": "submitted", "message": "objetivo cero", "mode": mode,
                    })
                    continue

                delta = desired - current
                if abs(delta) < min_order:
                    records.append({
                        "timestamp": ts, "ticker": yf_sym, "side": "HOLD",
                        "target_usd": round(desired, 2), "delta_usd": round(delta, 2),
                        "status": "skipped", "message": "delta < min_order", "mode": mode,
                    })
                    continue

                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                order = client.submit_order(MarketOrderRequest(
                    symbol=alpaca_sym,
                    notional=round(abs(delta), 2),
                    side=side,
                    time_in_force=tif,
                ))
                logger.info(
                    "[%s] %s %s: $%.2f (objetivo $%.0f) id=%s",
                    mode, side.value.upper(), yf_sym, abs(delta), desired, order.id,
                )
                records.append({
                    "timestamp": ts, "ticker": yf_sym, "side": side.value.upper(),
                    "target_usd": round(desired, 2), "delta_usd": round(delta, 2),
                    "status": "submitted", "message": f"order_id={order.id}", "mode": mode,
                })
            except Exception as exc:
                logger.error("Orden TSMOM %s falló: %s", yf_sym, exc)
                records.append({
                    "timestamp": ts, "ticker": yf_sym, "side": "ERROR",
                    "target_usd": round(desired, 2), "delta_usd": 0.0,
                    "status": "error", "message": str(exc), "mode": mode,
                })

    except ImportError:
        logger.warning("alpaca-py no instalado — instala con: pip install alpaca-py")
        records.append({
            "timestamp": ts, "ticker": "", "side": "HOLD", "target_usd": 0.0,
            "delta_usd": 0.0, "status": "simulated", "message": "alpaca-py no disponible",
            "mode": "paper_sim",
        })
    except Exception as exc:
        logger.error("Error en ejecución TSMOM Alpaca: %s", exc)
        records.append({
            "timestamp": ts, "ticker": "", "side": "ERROR", "target_usd": 0.0,
            "delta_usd": 0.0, "status": "error", "message": str(exc), "mode": "unknown",
        })

    n_sub = sum(1 for r in records if r["status"] == "submitted")
    logger.info("TSMOM Alpaca: %d órdenes enviadas | %d registros", n_sub, len(records))
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
