"""
Scheduler diario de señales de trading.

Uso:
    uv run python scheduler.py            # ejecutar hoy
    uv run python scheduler.py --force    # ignorar verificacion de dia de mercado
    uv run python scheduler.py --dry-run  # mostrar que haria sin ejecutar

El script:
  1. Verifica si hoy es dia de mercado en EE.UU.
  2. Descarga datos recientes (ultimos 400 dias)
  3. Ejecuta el pipeline de señales (sin backtesting — ~30s)
  4. Imprime las recomendaciones del dia
  5. Guarda un log en data/08_reporting/daily_log/
"""
import argparse
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Forzar UTF-8 en stdout para evitar errores de encoding en Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Festivos mercado EE.UU. (NYSE) 2025-2027 ─────────────────────────────────
US_MARKET_HOLIDAYS = {
    # 2025
    "2025-01-01",  # Año Nuevo
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents' Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Navidad
    # 2026
    "2026-01-01",  # Año Nuevo
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observado)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Navidad
    # 2027
    "2027-01-01",  # Año Nuevo
    "2027-01-18",  # MLK Day
    "2027-02-15",  # Presidents' Day
    "2027-03-26",  # Good Friday
    "2027-05-31",  # Memorial Day
    "2027-06-18",  # Juneteenth (observado)
    "2027-07-05",  # Independence Day (observado)
    "2027-09-06",  # Labor Day
    "2027-11-25",  # Thanksgiving
    "2027-12-24",  # Navidad (observado)
}

LOG_DIR = Path("data/08_reporting/daily_log")


def is_trading_day(date: datetime) -> bool:
    if date.weekday() >= 5:  # sabado=5, domingo=6
        return False
    return date.strftime("%Y-%m-%d") not in US_MARKET_HOLIDAYS


def find_latest_signal() -> Path | None:
    signal_dir = Path("data/07_model_output/signal.json")
    if not signal_dir.exists():
        return None
    versions = sorted(signal_dir.glob("*/signal.json"), key=os.path.getmtime)
    return versions[-1] if versions else None


def format_report(signal_path: Path, today_str: str) -> str:
    import pandas as pd

    lines = []
    lines.append("=" * 60)
    lines.append(f"  SEÑALES DE TRADING — {today_str}")
    lines.append("=" * 60)

    try:
        df = pd.read_json(signal_path)

        buy_df  = df[df["signal"] == "BUY"].head(5)
        sell_df = df[df["signal"] == "SELL"].head(3)
        hold_df = df[df["signal"] == "HOLD"].head(3)

        lines.append(f"\n{'Ticker':<10} {'Señal':<6} {'Score':>6} {'Poly':>6}  Razonamiento")
        lines.append("-" * 60)

        for _, row in buy_df.iterrows():
            ticker  = str(row.get("ticker", ""))
            score   = float(row.get("score", 0))
            poly    = float(row.get("poly_boost", 0))
            reason  = str(row.get("reasoning", ""))[:45]
            lines.append(f"  {ticker:<8}  BUY   {score:>5.1f}  {poly:>+5.2f}  {reason}")

        if not sell_df.empty:
            lines.append("")
            for _, row in sell_df.iterrows():
                ticker = str(row.get("ticker", ""))
                score  = float(row.get("score", 0))
                lines.append(f"  {ticker:<8}  SELL  {score:>5.1f}")

        if not hold_df.empty:
            lines.append("")
            for _, row in hold_df.iterrows():
                ticker = str(row.get("ticker", ""))
                score  = float(row.get("score", 0))
                lines.append(f"  {ticker:<8}  HOLD  {score:>5.1f}")

        n_buy  = (df["signal"] == "BUY").sum()
        n_sell = (df["signal"] == "SELL").sum()
        n_hold = (df["signal"] == "HOLD").sum()

        lines.append("\n" + "-" * 60)
        lines.append(f"  Total: {len(df)} activos | BUY={n_buy}  HOLD={n_hold}  SELL={n_sell}")

        # Recomendacion ejecutable
        top_buys = buy_df["ticker"].tolist()
        if top_buys:
            lines.append(f"\n  ACCION RECOMENDADA:")
            lines.append(f"  Comprar/mantener: {', '.join(top_buys[:2])}")
            sells = sell_df["ticker"].tolist()
            if sells:
                lines.append(f"  Vender:           {', '.join(sells)}")
        else:
            lines.append(f"\n  ACCION RECOMENDADA: Mantener posiciones (sin BUY claros)")

    except Exception as e:
        lines.append(f"  [Error al leer señal: {e}]")

    lines.append("=" * 60)
    return "\n".join(lines)


def run_pipeline(start_date: str, end_date: str, dry_run: bool) -> bool:
    cmd = [
        "uv", "run", "kedro", "run",
        "--pipeline=signals",
        f"--params=start_date={start_date},end_date={end_date}",
    ]
    print(f"  Ejecutando: {' '.join(cmd)}")
    if dry_run:
        print("  [DRY RUN] Pipeline no ejecutado.")
        return True

    result = subprocess.run(cmd, capture_output=True, text=True)
    ok = "Pipeline execution completed" in result.stdout + result.stderr
    if not ok:
        print("  ERROR: Pipeline fallo.")
        print(result.stderr[-1500:])
    return ok


def save_log(report: str, today_str: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{today_str}.txt"
    log_file.write_text(report, encoding="utf-8")
    return log_file


def main():
    parser = argparse.ArgumentParser(description="Scheduler diario de señales")
    parser.add_argument("--force",   action="store_true", help="Ignorar verificacion de dia de mercado")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar acciones sin ejecutar pipeline")
    args = parser.parse_args()

    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    print(f"\n[{today_str}] Trading Agent Scheduler")
    print("-" * 40)

    # Verificar dia de mercado
    if not args.force and not is_trading_day(today):
        day_name = ["Lunes","Martes","Miercoles","Jueves","Viernes","Sabado","Domingo"][today.weekday()]
        reason = "fin de semana" if today.weekday() >= 5 else "festivo NYSE"
        print(f"  {day_name} — {reason}. No se generan señales.")
        print("  Usa --force para ejecutar de todas formas.")
        return

    # Calcular ventana de datos (400 dias = suficiente para EMA-200 + features)
    end_date   = today_str
    start_date = (today - timedelta(days=400)).strftime("%Y-%m-%d")
    print(f"  Periodo de datos: {start_date} a {end_date}")

    # Ejecutar pipeline
    ok = run_pipeline(start_date, end_date, dry_run=args.dry_run)
    if not ok:
        sys.exit(1)

    if args.dry_run:
        print("  [DRY RUN] Fin.")
        return

    # Leer y mostrar señales
    signal_path = find_latest_signal()
    if signal_path is None:
        print("  No se encontro archivo de señales.")
        sys.exit(1)

    report = format_report(signal_path, today_str)
    print(report)

    # Guardar log diario
    log_file = save_log(report, today_str)
    print(f"\n  Log guardado en: {log_file}")

    # Notificacion Telegram
    try:
        import pandas as pd
        import notifier
        sig = pd.read_json(signal_path)
        notifier.notify_signals(
            strategy="Polymarket",
            report=report,
            n_buy=int((sig["signal"] == "BUY").sum()),
            n_sell=int((sig["signal"] == "SELL").sum()),
        )
    except Exception as e:
        print(f"  [Telegram] {e}")


if __name__ == "__main__":
    main()
