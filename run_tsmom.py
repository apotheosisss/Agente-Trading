"""Backtester Time-Series Momentum (TSMOM) long/short multi-activo.

Reforma (PLAN_REFORMA.md). Edge: trend-following diversificado, el de mejor
evidencia OOS. Cada mercado se opera LARGO si su propia tendencia es alcista,
CORTO si bajista; tamaño por volatility targeting (cada activo aporta riesgo
similar). El Sharpe nace de diversificar trends descorrelacionados.

Self-contained: descarga su universo, cachea, y reporta walk-forward + correlación
con SPY + comportamiento en bears (2020, 2022). No usa el motor Kedro long-only.

Uso:  uv run python run_tsmom.py <universo>
      universos: r1 | r2 | crypto | r2crypto
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Universos ────────────────────────────────────────────────────────────────
UNIVERSES = {
    # R1: multi-activo básico (equity + bonos + oro + materias primas)
    "r1": ["SPY", "QQQ", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "VNQ"],
    # R2: clases de activo reales y descorrelacionadas (FX, commodities, rates)
    "r2": ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ",
           "IEF", "TLT", "LQD", "SHY",
           "GLD", "DBC", "USO", "DBA", "SLV",
           "UUP", "FXE", "FXY"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
               "ADA-USD", "DOGE-USD", "LTC-USD"],
    "r2crypto": ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ",
                 "IEF", "TLT", "LQD", "GLD", "DBC", "USO", "DBA", "SLV",
                 "UUP", "FXE", "FXY", "BTC-USD", "ETH-USD"],
    # Breadth: ~31 mercados en muchos subsectores (proxy ETF de "más futuros")
    "broad": ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "FXI", "VNQ",
              "XLE", "XLF", "XLK", "XLU", "XLV",
              "SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB",
              "GLD", "SLV", "DBC", "USO", "UNG", "DBA", "CPER",
              "UUP", "FXE", "FXY",
              "BTC-USD", "ETH-USD"],
}

START = "2015-01-01"
END = "2026-04-27"
SPLIT = pd.Timestamp("2022-01-01")   # out-of-sample desde aquí
N_FOLDS = 5

# ── Parámetros de estrategia (override por env para barridos) ─────────────────
import os
LOOKBACKS = [int(x) for x in os.environ.get("TS_LOOKBACKS", "63,126,252").split(",")]
VOL_WINDOW = int(os.environ.get("TS_VOLWIN", "63"))
PER_POS_VOL = float(os.environ.get("TS_POSVOL", "0.10"))
MAX_W = float(os.environ.get("TS_MAXW", "0.20"))
MAX_GROSS = float(os.environ.get("TS_GROSS", "1.5"))
REBAL = int(os.environ.get("TS_REBAL", "21"))     # mensual (clásico TSMOM)
COST_BPS = float(os.environ.get("TS_COST", "0.0010"))
NOTRADE = float(os.environ.get("TS_BAND", "0.0"))  # banda no-trade (anti-whipsaw)
SIGNAL_MODE = os.environ.get("TS_SIGNAL", "strength")  # sign | strength (V3 = strength)
PVT = float(os.environ.get("TS_PVT", "0.12"))      # vol objetivo anual de cartera (risk parity); V3 default
MAX_LEV = float(os.environ.get("TS_MAXLEV", "3.0"))
REV_W = float(os.environ.get("TS_REV", "0.2"))     # peso reversión/value (0.2 = óptimo disciplinado)
REV_LB = int(os.environ.get("TS_REVLB", "504"))    # lookback reversión (~2 años)


def fetch_prices(tickers: list[str], tag: str) -> pd.DataFrame:
    cache = Path(f"data/01_raw/tsmom_{tag}.parquet")
    if cache.exists():
        df = pd.read_parquet(cache)
        if set(tickers) <= set(df.columns):
            return df[tickers].dropna(how="all")
    import yfinance as yf
    cols = {}
    for t in tickers:
        d = yf.download(t, start=START, end=END, auto_adjust=True, progress=False)
        if d.empty:
            print(f"  WARN sin datos: {t}")
            continue
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0].lower() for c in d.columns]
        else:
            d.columns = [c.lower() for c in d.columns]
        cols[t] = d["close"]
    px = pd.DataFrame(cols).sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    px.to_parquet(cache)
    return px


def tsmom_signal(px: pd.DataFrame) -> pd.DataFrame:
    """Conviccion en [-1,1] sobre varios lookbacks.

    - mode 'sign'    : media del SIGNO del momentum (TSMOM clásico, ±1 buckets).
    - mode 'strength': media de tanh(momentum normalizado por volatilidad) →
                       conviccion continua (más peso a tendencias fuertes).
    """
    if SIGNAL_MODE == "strength":
        ann_vol = px.pct_change().rolling(VOL_WINDOW).std() * np.sqrt(252)
        sig = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        for L in LOOKBACKS:
            mom = px / px.shift(L) - 1.0
            horizon_vol = ann_vol * np.sqrt(L / 252.0)      # vol esperada del retorno a L días
            z = mom / horizon_vol.replace(0, np.nan)
            sig = sig + np.tanh(z)
        trend = (sig / len(LOOKBACKS)).fillna(0.0)
        return _maybe_add_reversal(px, trend)
    # 'sign' (clásico)
    sig = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for L in LOOKBACKS:
        sig = sig + np.sign(px / px.shift(L) - 1.0)
    trend = (sig / len(LOOKBACKS)).fillna(0.0)
    return _maybe_add_reversal(px, trend)


def _maybe_add_reversal(px: pd.DataFrame, trend: pd.DataFrame) -> pd.DataFrame:
    """Combina trend con un sleeve de REVERSIÓN de largo plazo (value cross-asset).

    Reversión: el activo que más ha caído en ~REV_LB días tiende a rebotar →
    señal = -z-score del retorno de largo plazo (cross-sectional). Diversifica
    el trend porque paga en regímenes distintos. Si REV_W=0, devuelve solo trend.
    """
    if REV_W <= 0:
        return trend
    long_ret = px / px.shift(REV_LB) - 1.0
    # z-score cross-sectional por fila (entre activos), señal contraria
    mu = long_ret.mean(axis=1)
    sd = long_ret.std(axis=1).replace(0, np.nan)
    rev = (-(long_ret.sub(mu, axis=0)).div(sd, axis=0)).clip(-1, 1).fillna(0.0)
    return ((1 - REV_W) * trend + REV_W * rev)


def backtest(px: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.Series:
    rets = px.pct_change().fillna(0.0)
    ann_vol = rets.rolling(VOL_WINDOW).std() * np.sqrt(252)
    sig = tsmom_signal(px)

    # Peso base por inverse-vol (cada posición ~PER_POS_VOL de vol), cap por pos.
    raw_w = sig * (PER_POS_VOL / ann_vol.replace(0, np.nan))
    raw_w = raw_w.clip(-MAX_W, MAX_W).fillna(0.0)
    rw = raw_w.to_numpy()
    rmat = rets.to_numpy()

    # Rebalanceo cada REBAL días, con escalado de riesgo + banda no-trade.
    held_arr = np.zeros_like(rw)
    cur = np.zeros(rw.shape[1])
    for i in range(len(rw)):
        if i % REBAL == 0:
            tgt = rw[i].copy()
            if PVT > 0 and i > VOL_WINDOW:
                # Volatility targeting de CARTERA (risk parity): escalar el libro
                # entero para que su vol ex-ante ≈ PVT, usando la covarianza
                # reciente (captura correlaciones entre mercados).
                win = rmat[i - VOL_WINDOW:i]
                cov = np.cov(win, rowvar=False) * 252.0
                pv = float(np.sqrt(max(tgt @ cov @ tgt, 1e-12)))
                s = min(MAX_LEV, PVT / pv) if pv > 0 else 0.0
                tgt = tgt * s
            else:
                # Cap de exposición bruta (modo base).
                g = np.abs(tgt).sum()
                if g > MAX_GROSS:
                    tgt = tgt * (MAX_GROSS / g)
            if NOTRADE > 0:
                move = np.abs(tgt - cur) > NOTRADE
                cur = np.where(move, tgt, cur)
            else:
                cur = tgt
        held_arr[i] = cur
    held = pd.DataFrame(held_arr, index=raw_w.index, columns=raw_w.columns)

    # Posición aplicada a los retornos del DIA SIGUIENTE (sin look-ahead).
    pos = held.shift(1).fillna(0.0)
    port_ret = (pos * rets).sum(axis=1)

    # Coste por turnover en cada rebalanceo.
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    port_ret = port_ret - turnover * cost_bps

    # Empezar cuando hay señal válida (tras el lookback más largo + vol window).
    start_i = max(LOOKBACKS) + VOL_WINDOW
    return port_ret.iloc[start_i:]


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    n = len(r)
    cagr = eq.iloc[-1] ** (252 / n) - 1 if n else 0
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() * 252 / (r.std() * np.sqrt(252)) if r.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": cagr * 100, "vol": vol * 100, "sharpe": sharpe, "maxdd": dd * 100, "eq": eq}


def fold_sharpes(r: pd.Series) -> list[float]:
    r = r.dropna()
    bounds = np.linspace(0, len(r), N_FOLDS + 1, dtype=int)
    out = []
    for i in range(N_FOLDS):
        seg = r.iloc[bounds[i]:bounds[i + 1]]
        out.append(round(seg.mean() * 252 / (seg.std() * np.sqrt(252)), 2) if seg.std() > 0 else 0.0)
    return out


def year_return(r: pd.Series, year: int) -> float:
    seg = r[(r.index >= f"{year}-01-01") & (r.index <= f"{year}-12-31")]
    return ((1 + seg).prod() - 1) * 100 if len(seg) else float("nan")


def main():
    uni_name = sys.argv[1] if len(sys.argv) > 1 else "r1"
    tickers = UNIVERSES[uni_name]
    print(f"\n=== TSMOM long/short | universo '{uni_name}' ({len(tickers)} activos) ===")
    px = fetch_prices(tickers, uni_name)
    px = px.dropna(how="all").ffill()
    print(f"Datos: {px.shape[1]} activos, {len(px)} días, {px.index.min().date()} -> {px.index.max().date()}")

    r = backtest(px)
    m = metrics(r)
    rin = r[r.index < SPLIT]
    roos = r[r.index >= SPLIT]
    moos = metrics(roos)

    # SPY buy&hold benchmark (mismo rango que r)
    spy = px["SPY"] if "SPY" in px.columns else None
    if spy is not None:
        spy = spy.reindex(r.index).ffill()
        spy_ret = spy.pct_change().fillna(0.0)
        corr = r.corr(spy_ret)
        spy_oos = spy_ret[spy_ret.index >= SPLIT]
        spy_oos_cagr = ((1 + spy_oos).prod()) ** (252 / len(spy_oos)) - 1
    else:
        corr, spy_oos_cagr = float("nan"), float("nan")

    print(f"\nFULL:  CAGR {m['cagr']:.1f}% | Vol {m['vol']:.1f}% | Sharpe {m['sharpe']:.2f} | MaxDD {m['maxdd']:.1f}%")
    print(f"OOS:   CAGR {moos['cagr']:.1f}% | Sharpe {moos['sharpe']:.2f} | MaxDD {moos['maxdd']:.1f}%")
    print(f"Folds Sharpe: {fold_sharpes(r)}  (min {min(fold_sharpes(r)):.2f})")
    print(f"Correlacion con SPY (diaria): {corr:.2f}   [meta <= 0.3]")
    print(f"SPY buy&hold OOS CAGR: {spy_oos_cagr*100:.1f}%")
    print(f"Retorno en bears:  2020 {year_return(r,2020):+.1f}%  |  2022 {year_return(r,2022):+.1f}%")
    print(f"  (SPY: 2020 {year_return(spy_ret,2020):+.1f}%  2022 {year_return(spy_ret,2022):+.1f}%)" if spy is not None else "")

    # Gate R
    g_sharpe = moos["sharpe"] >= 0.7
    g_corr = abs(corr) <= 0.4
    g_2022 = year_return(r, 2022) > 0
    print(f"\nGATE  OOS-Sharpe>=0.7: {'OK' if g_sharpe else 'NO'} ({moos['sharpe']:.2f}) | "
          f"corr<=0.4: {'OK' if g_corr else 'NO'} ({corr:.2f}) | "
          f"2022>0: {'OK' if g_2022 else 'NO'} ({year_return(r,2022):+.1f}%)")


if __name__ == "__main__":
    main()
