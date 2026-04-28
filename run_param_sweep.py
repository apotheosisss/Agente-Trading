"""Sweep de parametros para universo ampliado (37 activos).
Prueba combinaciones de rebalance_interval x min_entry_score.
"""
import subprocess, csv, re
from pathlib import Path

PARAMS_PATH  = Path("conf/base/parameters.yml")
METRICS_PATH = Path("data/08_reporting/metrics.csv")
WF_PATH      = Path("data/08_reporting/walk_forward.csv")

# Combinaciones a probar
INTERVALS   = [5, 10, 15, 21]       # dias entre rebalanceos
SCORES      = [1.5, 2.0, 2.5, 3.0]  # min_entry_score

def load_yaml():
    with open(PARAMS_PATH, encoding="utf-8") as f:
        return f.read()

def save_yaml(content):
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        f.write(content)

def patch(content, interval, score):
    # Agrega o reemplaza rebalance_interval
    if "rebalance_interval:" in content:
        content = re.sub(
            r"(  rebalance_interval:\s*)\d+",
            lambda m: m.group(1) + str(interval),
            content
        )
    else:
        content = content.replace(
            "  rebalance_day:",
            f"  rebalance_interval: {interval}\n  rebalance_day:"
        )
    # min_entry_score
    content = re.sub(
        r"(  min_entry_score:\s*)[\d.]+",
        lambda m: m.group(1) + str(score),
        content
    )
    return content

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
        "oos_cagr":   float(oos["cagr_pct"]),
        "oos_sharpe": float(oos["sharpe"]),
    }

def run_kedro():
    r = subprocess.run(["uv", "run", "kedro", "run"], capture_output=True, text=True)
    return "Pipeline execution completed" in r.stdout + r.stderr

# ── Correr sweep ──────────────────────────────────────
original_yaml = load_yaml()
results = {}

print(f"\n{'Interval':>9} {'Score':>6} {'Trades/yr':>10} {'WR%':>6} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Equity':>12} {'OOS%':>7}")
print("-" * 82)

for interval in INTERVALS:
    for score in SCORES:
        content = patch(original_yaml, interval, score)
        save_yaml(content)
        ok = run_kedro()
        if ok:
            m = read_metrics()
            results[(interval, score)] = m
            print(f"  {interval:>7}d  {score:>5.1f}  {m['trades_yr']:>9.1f}  {m['win_rate']:>5.1f}%  "
                  f"{m['cagr']:>6.1f}%  {m['sharpe']:>7.3f}  {m['maxdd']:>7.1f}%  "
                  f"${m['equity']:>10,.0f}  {m['oos_cagr']:>6.1f}%")
        else:
            print(f"  {interval:>7}d  {score:>5.1f}  FALLO")

save_yaml(original_yaml)

# ── Tabla final ordenada por Sharpe ──────────────────
if not results:
    print("Sin resultados.")
else:
    print("\n\n" + "=" * 90)
    print("TOP CONFIGURACIONES — ordenadas por Sharpe")
    print(f"{'Interval':>9} {'Score':>6} {'Trades/yr':>10} {'WR%':>6} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Equity':>12} {'OOS CAGR':>9}")
    print("-" * 90)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    for (interval, score), m in sorted_results[:8]:
        print(f"  {interval:>7}d  {score:>5.1f}  {m['trades_yr']:>9.1f}  {m['win_rate']:>5.1f}%  "
              f"{m['cagr']:>6.1f}%  {m['sharpe']:>7.3f}  {m['maxdd']:>7.1f}%  "
              f"${m['equity']:>10,.0f}  {m['oos_cagr']:>8.1f}%")
    print("=" * 90)

    best = sorted_results[0]
    print(f"\nMejor configuracion: interval={best[0][0]}d, score={best[0][1]}")
    print(f"  CAGR={best[1]['cagr']:.1f}%, Sharpe={best[1]['sharpe']:.3f}, MaxDD={best[1]['maxdd']:.1f}%, Equity=${best[1]['equity']:,.0f}")
    print(f"\nYAML restaurado.")
