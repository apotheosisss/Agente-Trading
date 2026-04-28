"""Sweep de subgrupos — universo base (18) + bloques especificos.
Parametros fijos optimos: rebalance_interval=15, min_entry_score=1.5
"""
import subprocess, csv, re
from pathlib import Path

PARAMS_PATH  = Path("conf/base/parameters.yml")
METRICS_PATH = Path("data/08_reporting/metrics.csv")
WF_PATH      = Path("data/08_reporting/walk_forward.csv")

BASE_UNIVERSE = [
    '  - "SPY"     # S&P 500 — referencia / ancla',
    '  - "QQQ"     # Nasdaq 100 — tecnologia',
    '  - "NVDA"    # Nvidia — semiconductor / AI lider',
    '  - "META"    # Meta — mega-cap growth',
    '  - "AMZN"    # Amazon — cloud + e-commerce',
    '  - "GOOGL"   # Alphabet — diversificado tech',
    '  - "AAPL"    # Apple — large cap quality',
    '  - "MSFT"    # Microsoft — cloud + AI',
    '  - "BTC-USD" # Bitcoin — alta volatilidad / alto potencial',
    '  - "ETH-USD" # Ethereum — cripto #2',
    '  - "XLK"     # Tech ETF — exposicion sectorial amplia',
    '  - "SOXX"    # Semiconductores ETF — ciclo AI',
    '  - "CVX"     # Chevron — Venezuela',
    '  - "SLB"     # Schlumberger — servicios petroleros',
    '  - "HAL"     # Halliburton — perforacion',
    '  - "VLO"     # Valero — refineria',
    '  - "OXY"     # Occidental Petroleum',
    '  - "XLE"     # Energy ETF',
]

BLOCKS = {
    "AI pura": [
        '  - "PLTR"    # Palantir — IA gobierno/empresas',
        '  - "AMD"     # AMD — chips IA',
        '  - "AVGO"    # Broadcom — chips IA personalizados',
        '  - "ARM"     # Arm Holdings — arquitectura chip IA',
        '  - "ORCL"    # Oracle — cloud + IA',
        '  - "CRM"     # Salesforce — IA empresarial',
        '  - "SNOW"    # Snowflake — infraestructura datos IA',
        '  - "SMCI"    # Super Micro — servidores IA',
    ],
    "Electricidad": [
        '  - "NEE"     # NextEra Energy — renovables',
        '  - "CEG"     # Constellation Energy — nuclear',
        '  - "VST"     # Vistra — generacion electrica',
        '  - "ETN"     # Eaton — gestion energia',
        '  - "PWR"     # Quanta Services — red electrica',
    ],
    "EV + Urbano": [
        '  - "TSLA"    # Tesla — lider EV',
        '  - "ALB"     # Albemarle — litio',
        '  - "CHPT"    # ChargePoint — carga EV',
        '  - "DRIV"    # Global X EV ETF',
        '  - "CAT"     # Caterpillar — construccion',
        '  - "URI"     # United Rentals — equipos construccion',
    ],
    "AI + Electricidad": [
        '  - "PLTR"    # Palantir — IA gobierno/empresas',
        '  - "AMD"     # AMD — chips IA',
        '  - "AVGO"    # Broadcom — chips IA personalizados',
        '  - "ORCL"    # Oracle — cloud + IA',
        '  - "NEE"     # NextEra Energy — renovables',
        '  - "CEG"     # Constellation Energy — nuclear',
        '  - "VST"     # Vistra — generacion electrica',
        '  - "PWR"     # Quanta Services — red electrica',
    ],
    "AI + Electricidad + EV": [
        '  - "PLTR"    # Palantir — IA gobierno/empresas',
        '  - "AMD"     # AMD — chips IA',
        '  - "AVGO"    # Broadcom — chips IA personalizados',
        '  - "ORCL"    # Oracle — cloud + IA',
        '  - "NEE"     # NextEra Energy — renovables',
        '  - "CEG"     # Constellation Energy — nuclear',
        '  - "VST"     # Vistra — generacion electrica',
        '  - "TSLA"    # Tesla — lider EV',
        '  - "ALB"     # Albemarle — litio',
    ],
}

def load_yaml():
    with open(PARAMS_PATH, encoding="utf-8") as f:
        return f.read()

def save_yaml(content):
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        f.write(content)

def build_universe_block(extra_tickers):
    lines = ["universe:"] + BASE_UNIVERSE + extra_tickers
    return "\n".join(lines)

def patch_yaml(content, universe_block, interval=15, score=1.5):
    # Reemplazar bloque universe
    content = re.sub(
        r"universe:.*?(?=\nticker:)",
        universe_block + "\n",
        content,
        flags=re.DOTALL
    )
    # rebalance_interval
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
    ins = next(r for r in wf if "In-sample" in r["periodo"])
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
        "is_cagr":    float(ins["cagr_pct"]),
    }

def run_kedro():
    r = subprocess.run(["uv", "run", "kedro", "run"], capture_output=True, text=True)
    return "Pipeline execution completed" in r.stdout + r.stderr

# ── Baseline: universo original sin extras ────────────────────────────────────
original_yaml = load_yaml()
results = {}

subgroups = {"BASE (18 activos — referencia)": []} | BLOCKS

print(f"\n{'Subgrupo':<35} {'N':>3} {'T/yr':>6} {'WR%':>6} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Equity':>12} {'OOS%':>7} {'IS%':>7}")
print("-" * 100)

for name, extra in subgroups.items():
    universe_block = build_universe_block(extra)
    content = patch_yaml(original_yaml, universe_block)
    save_yaml(content)
    ok = run_kedro()
    if ok:
        m = read_metrics()
        results[name] = m
        n = len(BASE_UNIVERSE) + len(extra)
        tag = " ***" if name == "BASE (18 activos — referencia)" else ""
        print(f"  {name:<33} {n:>3}  {m['trades_yr']:>5.1f}  {m['win_rate']:>5.1f}%  "
              f"{m['cagr']:>6.1f}%  {m['sharpe']:>7.3f}  {m['maxdd']:>7.1f}%  "
              f"${m['equity']:>10,.0f}  {m['oos_cagr']:>6.1f}%  {m['is_cagr']:>6.1f}%{tag}")
    else:
        print(f"  {name:<33} FALLO")

save_yaml(original_yaml)

# ── Ranking final ─────────────────────────────────────────────────────────────
if results:
    print("\n\n" + "=" * 85)
    print("RANKING POR SHARPE")
    print(f"{'Subgrupo':<35} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Equity':>12} {'OOS%':>7}")
    print("-" * 85)
    for name, m in sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True):
        print(f"  {name:<33}  {m['cagr']:>6.1f}%  {m['sharpe']:>7.3f}  {m['maxdd']:>7.1f}%  "
              f"${m['equity']:>10,.0f}  {m['oos_cagr']:>6.1f}%")
    print("=" * 85)
    print("\nYAML restaurado.")
