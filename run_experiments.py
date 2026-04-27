"""Barrido de min_entry_score — mejora de win rate en feature/crypto.
Config fija: max_positions=2, CB=0.40, ATR=4.5 (configuracion optima actual).
Objetivo: ver tradeoff entre win rate y CAGR al exigir señales mas fuertes.
"""
import subprocess, csv, re
from pathlib import Path

PARAMS_PATH  = Path("conf/base/parameters.yml")
METRICS_PATH = Path("data/08_reporting/metrics.csv")
WF_PATH      = Path("data/08_reporting/walk_forward.csv")

# Baseline conocido
BASELINE = {
    "score": 2.0,
    "cagr": 39.4, "sharpe": 0.862, "maxdd": -79.1,
    "equity": 329541, "win_rate": 39.28, "n_trades": 293,
    "cb_events": 7, "oos_cagr": 5.6, "oos_sharpe": 0.35,
}

# 8 valores de min_entry_score a probar (evitamos 2.0 que ya conocemos)
SCORE_VALUES = [1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

def load_yaml():
    with open(PARAMS_PATH, encoding="utf-8") as f:
        return f.read()

def save_yaml(content):
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        f.write(content)

def patch_param(content, key, value):
    patterns = {
        "score": r"(  min_entry_score:\s*)[\d.]+",
    }
    return re.sub(patterns[key], lambda m: m.group(1) + str(value), content)

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
        "cb_events":  int(row["circuit_break_events"]),
        "oos_cagr":   float(oos["cagr_pct"]),
        "oos_sharpe": float(oos["sharpe"]),
    }

def run_kedro():
    result = subprocess.run(
        ["uv", "run", "kedro", "run"],
        capture_output=True, text=True
    )
    return "Pipeline execution completed" in result.stdout + result.stderr

# ── Correr experimentos ───────────────────────────────
original_yaml = load_yaml()
all_results = {BASELINE["score"]: BASELINE}

print(f"\n{'Score':>6}  {'WinRate':>8}  {'Trades':>7}  {'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>8}  {'Equity':>12}")
print("-" * 75)

for score in SCORE_VALUES:
    content = patch_param(original_yaml, "score", score)
    save_yaml(content)
    ok = run_kedro()
    if ok:
        m = read_metrics()
        all_results[score] = m
        print(f"  {score:>4}  {m['win_rate']:>7.1f}%  {m['n_trades']:>7}  "
              f"{m['cagr']:>6.1f}%  {m['sharpe']:>7.3f}  "
              f"{m['maxdd']:>7.1f}%  ${m['equity']:>10,.0f}")
    else:
        print(f"  {score:>4}  FALLO")

# ── Restaurar ─────────────────────────────────────────
save_yaml(original_yaml)

# ── Tabla completa ────────────────────────────────────
print("\n\n" + "=" * 90)
print("TABLA COMPLETA — ordenada por min_entry_score")
print(f"{'Score':>6}  {'WinRate':>8}  {'Trades':>7}  {'CAGR':>7}  {'Sharpe':>7}  "
      f"{'MaxDD':>8}  {'Equity':>12}  {'OOS CAGR':>9}")
print("-" * 90)

best_wr     = max(all_results.items(), key=lambda x: x[1]["win_rate"])
best_sharpe = max(all_results.items(), key=lambda x: x[1]["sharpe"])
best_cagr   = max(all_results.items(), key=lambda x: x[1]["cagr"])
best_oos    = max(all_results.items(), key=lambda x: x[1]["oos_cagr"])

for score in sorted(all_results):
    m = all_results[score]
    tags = []
    if score == best_wr[0]:     tags.append("WIN RATE")
    if score == best_sharpe[0]: tags.append("SHARPE")
    if score == best_cagr[0]:   tags.append("CAGR")
    if score == best_oos[0]:    tags.append("OOS")
    tag = "  *** " + "+".join(tags) if tags else ""
    baseline_mark = "  <- ACTUAL" if score == 2.0 else ""
    print(f"  {score:>4}  {m['win_rate']:>7.1f}%  {m['n_trades']:>7}  "
          f"{m['cagr']:>6.1f}%  {m['sharpe']:>7.3f}  "
          f"{m['maxdd']:>7.1f}%  ${m['equity']:>10,.0f}  "
          f"{m['oos_cagr']:>8.1f}%{tag}{baseline_mark}")

print("=" * 90)
print(f"\n  Mayor Win Rate -> score={best_wr[0]}  ({best_wr[1]['win_rate']:.1f}%,  "
      f"CAGR={best_wr[1]['cagr']:.1f}%,  Equity=${best_wr[1]['equity']:,.0f})")
print(f"  Mejor Sharpe   -> score={best_sharpe[0]}  ({best_sharpe[1]['sharpe']:.3f},  "
      f"CAGR={best_sharpe[1]['cagr']:.1f}%)")
print(f"  Mejor CAGR     -> score={best_cagr[0]}  ({best_cagr[1]['cagr']:.1f}%)")
print(f"  Mejor OOS      -> score={best_oos[0]}  ({best_oos[1]['oos_cagr']:.1f}%)")
print("\nYAML restaurado al estado original.")
