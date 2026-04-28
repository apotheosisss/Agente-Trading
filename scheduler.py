"""
Scheduler diario de señales crypto.

Crypto opera 24/7 — no hay restriccion de dias de mercado.

Uso:
    uv run python scheduler.py            # ejecutar hoy
    uv run python scheduler.py --dry-run  # mostrar que haria sin ejecutar

El script:
  1. Descarga datos recientes (ultimos 400 dias)
  2. Ejecuta el pipeline de señales crypto (sin backtesting — ~30s)
  3. Imprime las recomendaciones del dia
  4. Guarda un log en data/08_reporting/daily_log/
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

LOG_DIR = Path("data/08_reporting/daily_log")


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
    lines.append(f"  SEÑALES CRYPTO — {today_str}")
    lines.append("=" * 60)

    try:
        df = pd.read_json(signal_path)

        buy_df  = df[df["signal"] == "BUY"].head(5)
        sell_df = df[df["signal"] == "SELL"].head(3)
        hold_df = df[df["signal"] == "HOLD"].head(3)

        lines.append(f"\n{'Ticker':<10} {'Señal':<6} {'Score':>6} {'Poly':>6}  Razonamiento")
        lines.append("-" * 60)

        for _, row in buy_df.iterrows():
            ticker = str(row.get("ticker", ""))
            score  = float(row.get("score", 0))
            poly   = float(row.get("poly_boost", 0))
            reason = str(row.get("reasoning", ""))[:45]
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
    parser = argparse.ArgumentParser(description="Scheduler diario crypto")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar acciones sin ejecutar pipeline")
    args = parser.parse_args()

    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    print(f"\n[{today_str}] Trading Agent Scheduler — CRYPTO")
    print("-" * 40)

    # Crypto opera 24/7 — siempre ejecutar
    end_date   = today_str
    start_date = (today - timedelta(days=400)).strftime("%Y-%m-%d")
    print(f"  Periodo de datos: {start_date} a {end_date}")

    ok = run_pipeline(start_date, end_date, dry_run=args.dry_run)
    if not ok:
        sys.exit(1)

    if args.dry_run:
        print("  [DRY RUN] Fin.")
        return

    signal_path = find_latest_signal()
    if signal_path is None:
        print("  No se encontro archivo de señales.")
        sys.exit(1)

    report = format_report(signal_path, today_str)
    print(report)

    log_file = save_log(report, today_str)
    print(f"\n  Log guardado en: {log_file}")


if __name__ == "__main__":
    main()
