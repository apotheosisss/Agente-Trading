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
    # Mediana de Sharpe entre folds (métrica de selección anti-overfit)
    resumen = next((r for r in wf if "RESUMEN" in r["periodo"]), None)
    median_fold_sharpe = float(resumen["sharpe"]) if resumen else float(oos["sharpe"])
    worst_fold_sharpe = float(resumen["max_drawdown_pct"]) if resumen else float(oos["sharpe"])
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
        "median_fold_sharpe": median_fold_sharpe,
        "worst_fold_sharpe":  worst_fold_sharpe,
    }


# ── Criterio de selección anti-overfit ───────────────────────────────────────
# Rankea por la MEDIANA de Sharpe entre folds (consistencia temporal), no por
# el resultado full (contaminado por in-sample). Descarta configs con drawdown
# inaceptable.
MAX_ACCEPTABLE_DD = -25.0   # MaxDD peor que esto = descartada

def selection_key(m: dict) -> float:
    if m["maxdd"] < MAX_ACCEPTABLE_DD:
        return -999.0
    return m["median_fold_sharpe"]

def run_kedro():
    # Solo el pipeline de backtesting: reutiliza feature_vector/vix cacheados.
    # Evita re-descargar de yfinance en cada iteración (rate-limiting → tickers
    # omitidos → métricas corruptas) y es ~15x más rápido.
    r = subprocess.run(
        ["uv", "run", "kedro", "run", "--pipeline", "backtesting"],
        capture_output=True, text=True,
    )
    return "Pipeline execution completed" in r.stdout + r.stderr

# ── Correr sweep ──────────────────────────────────────
original_yaml = load_yaml()
results = {}

print(f"\n{'Interval':>9} {'Score':>6} {'Trd/yr':>7} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'OOS-CAGR':>9} {'MedFold-Sh':>11}")
print("-" * 80)

for interval in INTERVALS:
    for score in SCORES:
        content = patch(original_yaml, interval, score)
        save_yaml(content)
        ok = run_kedro()
        if ok:
            m = read_metrics()
            results[(interval, score)] = m
            flag = " X" if m["maxdd"] < MAX_ACCEPTABLE_DD else ""
            print(f"  {interval:>7}d  {score:>5.1f}  {m['trades_yr']:>6.1f}  "
                  f"{m['cagr']:>6.1f}%  {m['sharpe']:>7.3f}  {m['maxdd']:>7.1f}%  "
                  f"{m['oos_cagr']:>8.1f}%  {m['median_fold_sharpe']:>10.2f}{flag}")
        else:
            print(f"  {interval:>7}d  {score:>5.1f}  FALLO")

save_yaml(original_yaml)

# ── Tabla final ordenada por MEDIANA de Sharpe entre folds (anti-overfit) ──
if not results:
    print("Sin resultados.")
else:
    print("\n\n" + "=" * 92)
    print("TOP CONFIGURACIONES — ordenadas por MEDIANA Sharpe folds (descarta MaxDD<-25%)")
    print(f"{'Interval':>9} {'Score':>6} {'Trd/yr':>7} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'OOS-CAGR':>9} {'MedFold-Sh':>11}")
    print("-" * 92)

    sorted_results = sorted(results.items(), key=lambda x: selection_key(x[1]), reverse=True)
    for (interval, score), m in sorted_results[:8]:
        flag = " X" if m["maxdd"] < MAX_ACCEPTABLE_DD else ""
        print(f"  {interval:>7}d  {score:>5.1f}  {m['trades_yr']:>6.1f}  "
              f"{m['cagr']:>6.1f}%  {m['sharpe']:>7.3f}  {m['maxdd']:>7.1f}%  "
              f"{m['oos_cagr']:>8.1f}%  {m['median_fold_sharpe']:>10.2f}{flag}")
    print("=" * 92)

    best = sorted_results[0]
    bm = best[1]
    print(f"\nMejor config (anti-overfit): interval={best[0][0]}d, min_entry_score={best[0][1]}")
    print(f"  MedianFoldSharpe={bm['median_fold_sharpe']:.2f} | OOS-CAGR={bm['oos_cagr']:.1f}% | "
          f"CAGR-full={bm['cagr']:.1f}% | MaxDD={bm['maxdd']:.1f}%")
    if selection_key(bm) <= -999.0:
        print("  ADVERTENCIA: ninguna config pasó el filtro de MaxDD. Revisar estrategia base.")
    print("\nYAML restaurado.")
