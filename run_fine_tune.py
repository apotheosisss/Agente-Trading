"""10 casos de afinamiento fino — universo original 18 activos.
Baseline: weekly rebalance, score=2.0, ATR=3.0, CB=0.25, pos=2
"""
import subprocess, csv, re
from pathlib import Path

PARAMS_PATH  = Path("conf/base/parameters.yml")
METRICS_PATH = Path("data/08_reporting/metrics.csv")
WF_PATH      = Path("data/08_reporting/walk_forward.csv")

# (label, rebalance_interval, min_entry_score, atr_mult, max_positions, cb)
CASES = [
    ("BASE  weekly  sc=2.0  ATR=3.0  pos=2  CB=0.25",  0, 2.0, 3.0, 2, 0.25),
    ("C01   int=10  sc=2.0  ATR=3.0  pos=2  CB=0.25", 10, 2.0, 3.0, 2, 0.25),
    ("C02   int=15  sc=2.0  ATR=3.0  pos=2  CB=0.25", 15, 2.0, 3.0, 2, 0.25),
    ("C03   weekly  sc=1.5  ATR=3.0  pos=2  CB=0.25",  0, 1.5, 3.0, 2, 0.25),
    ("C04   int=10  sc=1.5  ATR=3.0  pos=2  CB=0.25", 10, 1.5, 3.0, 2, 0.25),
    ("C05   int=15  sc=1.5  ATR=3.0  pos=2  CB=0.25", 15, 1.5, 3.0, 2, 0.25),
    ("C06   weekly  sc=2.0  ATR=4.0  pos=2  CB=0.25",  0, 2.0, 4.0, 2, 0.25),
    ("C07   int=10  sc=2.0  ATR=4.0  pos=2  CB=0.25", 10, 2.0, 4.0, 2, 0.25),
    ("C08   int=10  sc=1.5  ATR=4.0  pos=2  CB=0.35", 10, 1.5, 4.0, 2, 0.35),
    ("C09   int=15  sc=1.5  ATR=4.0  pos=2  CB=0.35", 15, 1.5, 4.0, 2, 0.35),
    ("C10   int=10  sc=2.0  ATR=3.0  pos=3  CB=0.25", 10, 2.0, 3.0, 3, 0.25),
]

def load_yaml():
    with open(PARAMS_PATH, encoding="utf-8") as f:
        return f.read()

def save_yaml(content):
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        f.write(content)

def patch(content, interval, score, atr, positions, cb):
    # rebalance_interval
    if "rebalance_interval:" in content:
        if interval == 0:
            content = re.sub(r"  rebalance_interval:\s*\d+\n", "", content)
        else:
            content = re.sub(
                r"(  rebalance_interval:\s*)\d+",
                lambda m: m.group(1) + str(interval), content
            )
    else:
        if interval > 0:
            content = content.replace(
                "  rebalance_day:",
                f"  rebalance_interval: {interval}\n  rebalance_day:"
            )
    # min_entry_score
    content = re.sub(
        r"(  min_entry_score:\s*)[\d.]+",
        lambda m: m.group(1) + str(score), content
    )
    # stop_loss_atr_mult
    content = re.sub(
        r"(  stop_loss_atr_mult:\s*)[\d.]+",
        lambda m: m.group(1) + str(atr), content
    )
    # max_positions
    content = re.sub(
        r"(  max_positions:\s*)\d+",
        lambda m: m.group(1) + str(positions), content
    )
    # max_drawdown_circuit
    content = re.sub(
        r"(  max_drawdown_circuit:\s*)[\d.]+",
        lambda m: m.group(1) + str(cb), content
    )
    return content

def read_metrics():
    with open(METRICS_PATH, encoding="utf-8") as f:
        row = list(csv.DictReader(f))[0]
    with open(WF_PATH, encoding="utf-8") as f:
        wf = list(csv.DictReader(f))
    oos = next(r for r in wf if "Out-of-sample" in r["periodo"])
    ins = next(r for r in wf if "In-sample" in r["periodo"])
    return {
        "cagr":      float(row["cagr_pct"]),
        "sharpe":    float(row["sharpe_ratio"]),
        "maxdd":     float(row["max_drawdown_pct"]),
        "equity":    float(row["final_equity_usd"]),
        "win_rate":  float(row["win_rate_pct"]),
        "trades_yr": float(row["trades_per_year"]),
        "oos_cagr":  float(oos["cagr_pct"]),
        "oos_sh":    float(oos["sharpe"]),
        "is_cagr":   float(ins["cagr_pct"]),
    }

def run_kedro():
    r = subprocess.run(["uv", "run", "kedro", "run"], capture_output=True, text=True)
    return "Pipeline execution completed" in r.stdout + r.stderr

# ── Correr experimentos ────────────────────────────────
original_yaml = load_yaml()
results = {}

print(f"\n{'Caso':<50} {'T/yr':>5} {'WR%':>5} {'CAGR':>6} {'Sharpe':>7} {'MaxDD':>7} {'Equity':>12} {'OOS%':>6}")
print("-" * 105)

for case in CASES:
    label, interval, score, atr, positions, cb = case
    content = patch(original_yaml, interval, score, atr, positions, cb)
    save_yaml(content)
    ok = run_kedro()
    if ok:
        m = read_metrics()
        results[label] = m
        marker = " <-- BASE" if label.startswith("BASE") else ""
        print(f"  {label:<48} {m['trades_yr']:>4.0f}  {m['win_rate']:>4.1f}%  "
              f"{m['cagr']:>5.1f}%  {m['sharpe']:>7.3f}  {m['maxdd']:>6.1f}%  "
              f"${m['equity']:>10,.0f}  {m['oos_cagr']:>5.1f}%{marker}")
    else:
        print(f"  {label:<48} FALLO")

save_yaml(original_yaml)

# ── Ranking final ─────────────────────────────────────
if not results:
    print("Sin resultados.")
else:
    print("\n\n" + "=" * 100)
    print("RANKING POR SHARPE — top 5")
    print(f"{'Caso':<50} {'CAGR':>6} {'Sharpe':>7} {'MaxDD':>7} {'Equity':>12} {'OOS%':>6} {'IS%':>6}")
    print("-" * 100)
    sorted_r = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    for label, m in sorted_r[:5]:
        marker = " ***" if label.startswith("BASE") else ""
        print(f"  {label:<48}  {m['cagr']:>5.1f}%  {m['sharpe']:>7.3f}  {m['maxdd']:>6.1f}%  "
              f"${m['equity']:>10,.0f}  {m['oos_cagr']:>5.1f}%  {m['is_cagr']:>5.1f}%{marker}")
    print("=" * 100)

    best = sorted_r[0]
    baseline = results.get(CASES[0][0])
    print(f"\nMejor: {best[0].strip()}")
    if baseline and best[0] != CASES[0][0]:
        delta_eq  = best[1]['equity'] - baseline['equity']
        delta_sh  = best[1]['sharpe'] - baseline['sharpe']
        delta_cagr = best[1]['cagr'] - baseline['cagr']
        print(f"  vs BASE -> Equity: ${delta_eq:+,.0f} | Sharpe: {delta_sh:+.3f} | CAGR: {delta_cagr:+.1f}%")
    print("\nYAML restaurado.")
