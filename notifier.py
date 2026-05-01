"""
Notificaciones via Telegram.

Configurar en conf/local/credentials.yml:
    telegram:
        bot_token: "123456:ABC-DEF..."
        chat_id: "987654321"

Para obtener estas credenciales:
    1. Abre Telegram y busca @BotFather
    2. Envia /newbot y sigue las instrucciones -> obtienes el bot_token
    3. Busca @userinfobot y envia /start -> obtienes tu chat_id
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _load_telegram_config() -> dict | None:
    cred_path = Path("conf/local/credentials.yml")
    if not cred_path.exists():
        return None
    with open(cred_path, encoding="utf-8") as f:
        creds = yaml.safe_load(f) or {}
    cfg = creds.get("telegram", {})
    token = cfg.get("bot_token", "")
    chat_id = cfg.get("chat_id", "")
    if not token or not chat_id or "PONER" in token:
        return None
    return {"token": token, "chat_id": chat_id}


def send(message: str) -> bool:
    cfg = _load_telegram_config()
    if cfg is None:
        logger.info("Telegram no configurado — notificacion omitida.")
        return False

    url = TELEGRAM_API.format(token=cfg["token"])
    payload = {
        "chat_id": cfg["chat_id"],
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Notificacion Telegram enviada.")
        return True
    except Exception as exc:
        logger.warning("Error enviando notificacion Telegram: %s", exc)
        return False


def notify_signals(strategy: str, report: str, n_buy: int, n_sell: int) -> bool:
    emoji_buy  = "🟢" if n_buy > 0 else "⚪"
    emoji_sell = "🔴" if n_sell > 0 else "⚪"

    lines = [
        f"<b>Trading Agent — {strategy}</b>",
        "",
        f"{emoji_buy} BUY:  {n_buy}",
        f"{emoji_sell} SELL: {n_sell}",
        "",
        "<pre>",
        report[:3000],
        "</pre>",
    ]
    return send("\n".join(lines))


def notify_alpaca_orders(orders_csv_path: Path) -> bool:
    try:
        import pandas as pd
        df = pd.read_csv(orders_csv_path)
        submitted = df[df["status"] == "submitted"]
        errors    = df[df["status"] == "error"]

        if submitted.empty and errors.empty:
            return send("<b>Alpaca</b> — Sin ordenes ejecutadas hoy.")

        lines = ["<b>Alpaca Paper Trading</b>"]
        for _, row in submitted.iterrows():
            lines.append(f"  BUY {row['ticker']}  ${row['notional_usd']:,.0f}  ✅")
        for _, row in errors.iterrows():
            lines.append(f"  ERROR {row['ticker']}  ❌  {row['message'][:60]}")

        return send("\n".join(lines))
    except Exception as exc:
        logger.warning("Error leyendo ordenes para notificacion: %s", exc)
        return False
