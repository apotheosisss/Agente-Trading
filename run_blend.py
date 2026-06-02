"""Combina el stream TSMOM (crisis-alpha, descorrelacionado) con SPY.

La forma realista de "batir a tener solo SPY" no es un sistema standalone, sino
AÑADIR un retorno descorrelacionado: una cartera w*SPY + (1-w)*TSMOM tiene mejor
Sharpe y menor drawdown que SPY al 100%, precisamente porque el TSMOM gana cuando
las acciones se hunden.
"""
import numpy as np
import pandas as pd

from run_tsmom import UNIVERSES, backtest, fetch_prices, SPLIT


def stats(r: pd.Series):
    r = r.dropna()
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (252 / len(r)) - 1
    sharpe = r.mean() * 252 / (r.std() * np.sqrt(252))
    dd = (eq / eq.cummax() - 1).min()
    return cagr * 100, sharpe, dd * 100


def main():
    px = fetch_prices(UNIVERSES["r2crypto"], "r2crypto").dropna(how="all").ffill()
    ts = backtest(px)
    spy = px["SPY"].reindex(ts.index).ffill().pct_change().fillna(0.0)
    # Escalar TSMOM a vol de SPY para que el "peso" sea comparable en riesgo
    ts_scaled = ts * (spy.std() / ts.std())

    for label, idx in [("FULL", ts.index), ("OOS (2022+)", ts.index[ts.index >= SPLIT])]:
        s = spy.loc[idx]
        t = ts_scaled.loc[idx]
        print(f"\n=== {label} ===")
        print(f"{'cartera':>18} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8}")
        for w in [1.0, 0.8, 0.7, 0.6, 0.5]:
            blend = w * s + (1 - w) * t
            cg, sh, dd = stats(blend)
            tag = "100% SPY" if w == 1.0 else f"{int(w*100)}% SPY/{int((1-w)*100)}% TSMOM"
            print(f"{tag:>18} {cg:>6.1f}% {sh:>7.2f} {dd:>7.1f}%")


if __name__ == "__main__":
    main()
