"""Ablation test A/B/C para medir el valor incremental de Polymarket.

Corre el pipeline de backtesting tres veces sobre los mismos datos cacheados:
    B = base                 (use_polymarket=false)
    A = base + Polymarket    (use_polymarket=true,  shuffle=false)
    C = base + Poly barajado (use_polymarket=true,  shuffle=true)

Interpretación (en OOS / mediana de folds):
    A > B y A > C  → Polymarket aporta señal real. Conservar.
    A ≈ B          → Polymarket no aporta nada. Borrar/flag off.
    A ≈ C          → el "efecto" es ruido con suerte. Borrar.

NOTA: requiere que data/01_raw/polymarket_history.parquet solape la ventana de
backtest. Hoy el histórico arranca vacío y se llena hacia adelante (1 fila/día),
así que hasta acumular meses el resultado será A==B==C (0 solape). El harness
queda listo para cuando los datos existan.
"""
import csv
import re
import subprocess
from pathlib import Path

PARAMS_PATH = Path("conf/base/parameters.yml")
METRICS_PATH = Path("data/08_reporting/metrics.csv")
WF_PATH = Path("data/08_reporting/walk_forward.csv")

VARIANTS = {
    "B  base":            (False, False),
    "A  base+poly":       (True,  False),
    "C  base+poly_shuf":  (True,  True),
}


def load_yaml() -> str:
    return PARAMS_PATH.read_text(encoding="utf-8")


def save_yaml(content: str) -> None:
    PARAMS_PATH.write_text(content, encoding="utf-8")


def _set_flag(content: str, key: str, value: bool) -> str:
    val = "true" if value else "false"
    if re.search(rf"  {key}:\s*\w+", content):
        return re.sub(rf"(  {key}:\s*)\w+", lambda m: m.group(1) + val, content)
    # insertar bajo la sección backtesting (tras 'commission:')
    return re.sub(
        r"(  commission:.*\n)",
        lambda m: m.group(1) + f"  {key}: {val}\n",
        content,
        count=1,
    )


def run_backtest() -> bool:
    r = subprocess.run(
        ["uv", "run", "kedro", "run", "--pipeline", "backtesting"],
        capture_output=True, text=True,
    )
    return "Pipeline execution completed" in r.stdout + r.stderr


def read_metrics() -> dict:
    row = list(csv.DictReader(METRICS_PATH.open(encoding="utf-8")))[0]
    wf = list(csv.DictReader(WF_PATH.open(encoding="utf-8")))
    oos = next(r for r in wf if "Out-of-sample" in r["periodo"])
    resumen = next((r for r in wf if "RESUMEN" in r["periodo"]), None)
    return {
        "cagr": float(row["cagr_pct"]),
        "sharpe": float(row["sharpe_ratio"]),
        "maxdd": float(row["max_drawdown_pct"]),
        "oos_cagr": float(oos["cagr_pct"]),
        "oos_sharpe": float(oos["sharpe"]),
        "median_fold_sharpe": float(resumen["sharpe"]) if resumen else 0.0,
    }


original = load_yaml()
results = {}

print(f"\n{'Variante':<18} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'OOS-CAGR':>9} {'OOS-Sh':>7} {'MedFold':>8}")
print("-" * 70)
for name, (use_poly, shuffle) in VARIANTS.items():
    content = _set_flag(original, "use_polymarket", use_poly)
    content = _set_flag(content, "polymarket_shuffle", shuffle)
    save_yaml(content)
    if run_backtest():
        m = read_metrics()
        results[name] = m
        print(f"{name:<18} {m['cagr']:>6.1f}% {m['sharpe']:>7.3f} {m['maxdd']:>7.1f}% "
              f"{m['oos_cagr']:>8.1f}% {m['oos_sharpe']:>7.2f} {m['median_fold_sharpe']:>8.2f}")
    else:
        print(f"{name:<18} FALLO")
save_yaml(original)

print("\n" + "=" * 70)
if {"A  base+poly", "B  base", "C  base+poly_shuf"} <= results.keys():
    a, b, c = results["A  base+poly"], results["B  base"], results["C  base+poly_shuf"]
    da = a["oos_cagr"] - b["oos_cagr"]
    dc = a["oos_cagr"] - c["oos_cagr"]
    print(f"A vs B (OOS-CAGR): {da:+.2f} pp   |   A vs C (OOS-CAGR): {dc:+.2f} pp")
    if abs(da) < 0.05:
        print("VEREDICTO: A ~= B -> Polymarket NO aporta valor medible. Recomendado: flag off / borrar.")
    elif abs(dc) < 0.05:
        print("VEREDICTO: A ~= C -> el efecto es indistinguible de ruido. Recomendado: borrar.")
    elif da > 0 and dc > 0:
        print("VEREDICTO: A > B y A > C -> Polymarket aporta senal real. Conservar.")
    else:
        print("VEREDICTO: Polymarket degrada el resultado. Recomendado: flag off.")
print("=" * 70)
print("YAML restaurado.")
