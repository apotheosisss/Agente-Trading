"""Microajuste de stop_loss_atr_mult — rama feature/crypto.
Config fija: max_positions=2, max_drawdown_circuit=0.40 (maximo retorno).
"""
import subprocess, csv, re
from pathlib import Path

PARAMS_PATH  = Path("conf/base/parameters.yml")
METRICS_PATH = Path("data/08_reporting/metrics.csv")
WF_PATH      = Path("data/08_reporting/walk_forward.csv")

# Valores de ATR ya conocidos (no repetir)
KNOWN = {
    2.0: {"cagr": 35.3, "sharpe": 0.82,  "maxdd": -73.5, "equity": 241426, "cb_events": 7, "oos_cagr": 4.8, "oos_sharpe": 0.32},
    2.5: {"cagr": 35.9, "sharpe": 0.82,  "maxdd": -74.2, "equity": 253778, "cb_events": 6, "oos_cagr": 4.5, "oos_sharpe": 0.32},
    4.0: {"cagr": 38.8, "sharpe": 0.85,  "maxdd": -79.1, "equity": 315272, "cb_events": 7, "oos_cagr": 5.6, "oos_sharpe": 0.35},
}

# 10 microajustes: exploramos 3.0-5.7 a pasos de 0.3
ATR_VALUES = [3.0, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7]

def load_yaml():
    with open(PARAMS_PATH, encoding="utf-8") as f:
        return f.read()

def save_yaml(content):
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        f.write(content)

def patch_param(content, key, value):
    patterns = {
        "atr": r"(  stop_loss_atr_mult:\s*)[\d.]+",
        "cb":  r"(  max_drawdown_circuit:\s*)[\d.]+",
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

# ── Guardar original y fijar CB=0.40 ─────────────────
original_yaml = load_yaml()
base_yaml = patch_param(original_yaml, "cb", 0.40)

all_results = dict(KNOWN)

print(f"\n{'ATR':>6}  {'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>8}  {'Equity':>12}  {'CB#':>4}  {'OOS':>7}")
print("-" * 70)

for atr in ATR_VALUES:
    content = patch_param(base_yaml, "atr", atr)
    save_yaml(content)
    ok = run_kedro()
    if ok:
        m = read_metrics()
        all_results[atr] = m
        print(f"  {atr:>4}  {m['cagr']:>6.1f}%  {m['sharpe']:>7.3f}  "
              f"{m['maxdd']:>7.1f}%  ${m['equity']:>10,.0f}  {m['cb_events']:>4}  "
              f"{m['oos_cagr']:>6.1f}%")
    else:
        print(f"  {atr:>4}  FALLO")

# ── Restaurar YAML original ───────────────────────────
save_yaml(original_yaml)

# ── Tabla completa ordenada por ATR ──────────────────
print("\n\n" + "=" * 82)
print("TABLA COMPLETA — ordenada por ATR  (*** = mejor en su categoría)")
print(f"{'ATR':>6}  {'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>8}  {'Equity':>12}  {'CB#':>4}  {'OOS CAGR':>9}  {'OOS Sharpe':>11}")
print("-" * 82)

best_sharpe = max(all_results.items(), key=lambda x: x[1]["sharpe"])
best_cagr   = max(all_results.items(), key=lambda x: x[1]["cagr"])
best_oos    = max(all_results.items(), key=lambda x: x[1]["oos_cagr"])

for atr in sorted(all_results):
    m = all_results[atr]
    tags = []
    if atr == best_sharpe[0]: tags.append("SHARPE")
    if atr == best_cagr[0]:   tags.append("CAGR")
    if atr == best_oos[0]:    tags.append("OOS")
    tag = "  *** " + "+".join(tags) if tags else ""
    print(f"  {atr:>4}  {m['cagr']:>6.1f}%  {m['sharpe']:>7.3f}  "
          f"{m['maxdd']:>7.1f}%  ${m['equity']:>10,.0f}  {m['cb_events']:>4}  "
          f"{m['oos_cagr']:>8.1f}%  {m['oos_sharpe']:>10.2f}{tag}")

print("=" * 82)
print(f"\n  Mejor CAGR   -> ATR={best_cagr[0]}   ({best_cagr[1]['cagr']:.1f}%,  Equity ${best_cagr[1]['equity']:,.0f})")
print(f"  Mejor Sharpe -> ATR={best_sharpe[0]}   ({best_sharpe[1]['sharpe']:.3f})")
print(f"  Mejor OOS    -> ATR={best_oos[0]}   ({best_oos[1]['oos_cagr']:.1f}%)")
print("\nYAML restaurado al estado original.")
