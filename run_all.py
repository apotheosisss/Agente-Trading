"""
Lanzador unificado — ejecuta ambas estrategias en secuencia.

Uso:
    uv run python run_all.py              # ejecutar ambas ramas
    uv run python run_all.py --dry-run   # mostrar acciones sin ejecutar
    uv run python run_all.py --only poly # solo rama polymarket
    uv run python run_all.py --only crypto # solo rama crypto

Estructura esperada:
    trading-agent/
        run_all.py                         <- este archivo
        .claude/worktrees/
            polymarket-work/               <- feature/polymarket
                scheduler.py
            wonderful-villani-fa15cf/      <- feature/crypto
                scheduler.py
"""
import argparse
import io
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).parent

BRANCHES = {
    "poly": {
        "name":    "Polymarket (largo plazo)",
        "path":    BASE / ".claude/worktrees/polymarket-work",
        "script":  "scheduler.py",
        "always":  False,  # respeta dias de mercado NYSE
    },
    "crypto": {
        "name":    "Crypto (corto plazo)",
        "path":    BASE / ".claude/worktrees/wonderful-villani-fa15cf",
        "script":  "scheduler.py",
        "always":  True,   # crypto opera 24/7
    },
}

# Festivos NYSE (para informar si polymarket no corre)
US_MARKET_HOLIDAYS = {
    "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26",
    "2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25",
    "2026-01-01","2026-01-19","2026-02-16","2026-04-03","2026-05-25",
    "2026-06-19","2026-07-03","2026-09-07","2026-11-26","2026-12-25",
    "2027-01-01","2027-01-18","2027-02-15","2027-03-26","2027-05-31",
    "2027-06-18","2027-07-05","2027-09-06","2027-11-25","2027-12-24",
}


def is_trading_day(date: datetime) -> bool:
    if date.weekday() >= 5:
        return False
    return date.strftime("%Y-%m-%d") not in US_MARKET_HOLIDAYS


def run_branch(key: str, cfg: dict, dry_run: bool) -> bool:
    path   = cfg["path"]
    script = path / cfg["script"]
    today  = datetime.now()

    print(f"\n{'='*60}")
    print(f"  {cfg['name']}")
    print(f"{'='*60}")

    if not path.exists():
        print(f"  [OMITIDO] Directorio no encontrado: {path}")
        return False

    if not script.exists():
        print(f"  [OMITIDO] scheduler.py no encontrado en {path}")
        return False

    # Polymarket solo corre en dias de mercado
    if not cfg["always"] and not is_trading_day(today):
        day = ["Lun","Mar","Mie","Jue","Vie","Sab","Dom"][today.weekday()]
        reason = "fin de semana" if today.weekday() >= 5 else "festivo NYSE"
        print(f"  [{day}] {reason} — polymarket no genera señales hoy.")
        return True

    cmd = ["uv", "run", "python", str(script)]
    if dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, cwd=str(path), text=True, encoding="utf-8",
                            errors="replace", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    print(result.stdout)

    ok = result.returncode == 0
    if not ok:
        print(f"  [ERROR] {cfg['name']} termino con codigo {result.returncode}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Lanzador unificado poly + crypto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", choices=["poly", "crypto"],
                        help="Ejecutar solo una rama")
    args = parser.parse_args()

    today_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n[{today_str}] Trading Agent — Sistema Unificado")
    print(f"  Ramas activas: {', '.join(BRANCHES.keys())}")

    targets = [args.only] if args.only else list(BRANCHES.keys())
    results = {}

    for key in targets:
        cfg = BRANCHES[key]
        results[key] = run_branch(key, cfg, dry_run=args.dry_run)

    # Resumen final
    print(f"\n{'='*60}")
    print("  RESUMEN")
    print(f"{'='*60}")
    for key, ok in results.items():
        status = "OK" if ok else "ERROR"
        print(f"  {BRANCHES[key]['name']:<30} [{status}]")
    print()


if __name__ == "__main__":
    main()
