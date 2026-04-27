"""Barrido de rebalance_interval — corto plazo crypto.
Config fija: max_positions=2, CB=0.40, ATR=4.5, score=1.0
Intervalos: 1 (diario), 2, 3, 5 (semanal), 10, 15, 21 (mensual)
"""
import subprocess, csv, re
from pathlib import Path

PARAMS_PATH  = Path("conf/base/parameters.yml")
METRICS_PATH = Path("data/08_reporting/metrics.csv")
WF_PATH      = Path("data/08_reporting/walk_forward.csv")

# Intervalos a probar (en días de trading)
INTERVALS = [1, 2, 3, 5, 10, 15, 21]
LABELS = {
    1:  "Diario        (1 día)",
    2:  "Cada 2 días",
    3:  "Cada 3 días",
    5:  "Semanal       (5 días)",
    10: "Quincenal     (10 días)",
    15: "Cada 3 sem.   (15 días)",
    21: "Mensual       (21 días)",
}

def load_yaml():
    with open(PARAMS_PATH, encoding="utf-8") as f:
        return f.read()

def save_yaml(content):
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        f.write(content)

def patch_interval(content, value):
    return re.sub(
        r"(  rebalance_interval:\s*)\d+",
        lambda m: m.group(1) + str(value),
        content
    )

def read_metrics():
    with open(METRICS_PATH, encoding="utf-8") as f:
        row = list(csv.DictReader(f))[0]
    with open(WF_PATH, encoding="utf-8") as f:
        wf = list(csv.DictReader(f))
    oos = next(r for r in wf if "Out-of-sample" in r["periodo"])
    return {
        "cagr":       float(row["cagr_pct"]),
        "sharpe":     float(row["sharpe_ratio"]),
        "maxdd":      float(row["max_drawdown_pct"]),
        "equity":     float(row["final_equity_usd"]),
        "win_rate":   float(row["win_rate_pct"]),
        "n_trades":   int(row["n_trades"]),
        "trades_yr":  float(row["trades_per_year"]),
        "cb_events":  int(row["circuit_break_events"]),
        "oos_cagr":   float(oos["cagr_pct"]),
        "oos_sharpe": float(oos["sharpe"]),
    }

def run_kedro():
    r = subprocess.run(["uv", "run", "kedro", "run"], capture_output=True, text=True)
    return "Pipeline execution completed" in r.stdout + r.stderr

# ── Correr experimentos ───────────────────────────────
original_yaml = load_yaml()
results = {}

print(f"\n{'Intervalo':<26} {'Trades/yr':>9} {'WinRate':>8} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Equity':>12} {'OOS':>7}")
print("-" * 90)

for interval in INTERVALS:
    content = patch_interval(original_yaml, interval)
    save_yaml(content)
    ok = run_kedro()
    if ok:
        m = read_metrics()
        results[interval] = m
        label = LABELS[interval]
        print(f"  {label:<24} {m['trades_yr']:>8.1f}  {m['win_rate']:>7.1f}%  "
              f"{m['cagr']:>6.1f}%  {m['sharpe']:>7.3f}  "
              f"{m['maxdd']:>7.1f}%  ${m['equity']:>10,.0f}  {m['oos_cagr']:>6.1f}%")
    else:
        print(f"  {LABELS[interval]:<24} FALLO")

# ── Restaurar ─────────────────────────────────────────
save_yaml(original_yaml)

# ── Tabla final ───────────────────────────────────────
print("\n\n" + "=" * 95)
print("RESULTADOS COMPLETOS — ordenados por intervalo")
print(f"{'Intervalo':<26} {'Trades/yr':>9} {'WinRate':>8} {'CAGR':>7} {'Sharpe':>7} "
      f"{'MaxDD':>8} {'Equity':>12} {'OOS CAGR':>9} {'OOS Sh':>7}")
print("-" * 95)

best_cagr   = max(results.items(), key=lambda x: x[1]["cagr"])
best_sharpe = max(results.items(), key=lambda x: x[1]["sharpe"])
best_wr     = max(results.items(), key=lambda x: x[1]["win_rate"])
best_oos    = max(results.items(), key=lambda x: x[1]["oos_cagr"])
best_maxdd  = min(results.items(), key=lambda x: abs(x[1]["maxdd"]))

for interval, m in results.items():
    tags = []
    if interval == best_cagr[0]:   tags.append("CAGR")
    if interval == best_sharpe[0]: tags.append("SHARPE")
    if interval == best_wr[0]:     tags.append("WINRATE")
    if interval == best_oos[0]:    tags.append("OOS")
    if interval == best_maxdd[0]:  tags.append("MENOR DD")
    tag = "  *** " + "+".join(tags) if tags else ""
    label = LABELS[interval]
    print(f"  {label:<24} {m['trades_yr']:>8.1f}  {m['win_rate']:>7.1f}%  "
          f"{m['cagr']:>6.1f}%  {m['sharpe']:>7.3f}  {m['maxdd']:>7.1f}%  "
          f"${m['equity']:>10,.0f}  {m['oos_cagr']:>8.1f}%  {m['oos_sharpe']:>6.2f}{tag}")

print("=" * 95)
clp_rate = 940
inv_clp  = 100_000
print(f"\n  Proyeccion sobre inversion de {inv_clp:,} CLP ({inv_clp/clp_rate:.0f} USD aprox):")
print(f"  {'Intervalo':<24} {'Final CLP':>14} {'Ganancia CLP':>14}")
print(f"  {'-'*55}")
for interval, m in results.items():
    mult  = m["equity"] / 10000
    final = inv_clp * mult
    print(f"  {LABELS[interval]:<24} {final:>13,.0f}  {final-inv_clp:>13,.0f}")
print(f"\nYAML restaurado. Tipo de cambio usado: {clp_rate} CLP/USD")
