"""Núcleo de la estrategia TSMOM long/short (fuente única de verdad).

Estrategia ganadora de la investigación (ver OPTIMIZACION_RESULTADOS.md):
time-series momentum multi-lookback con señal de fuerza, sizing por volatility
targeting de cartera (risk parity) y un sleeve de reversión de largo plazo.
Larga y corta, sobre un universo multi-activo descorrelacionado.

Funciones puras dirigidas por un dict de config — usadas tanto por los nodos
Kedro de producción como por el script de investigación run_tsmom.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Config por defecto = mejor configuración validada.
DEFAULT_CFG = {
    "lookbacks": [63, 126, 252],
    "vol_window": 63,
    "per_pos_vol": 0.10,
    "max_w": 0.20,
    "signal_mode": "strength",   # sign | strength
    "pvt": 0.12,                 # vol objetivo anual de cartera (0 = cap bruto)
    "max_gross": 1.5,
    "max_lev": 3.0,
    "rebal": 21,                 # días entre rebalanceos
    "cost_bps": 0.0010,
    "notrade": 0.0,
    "rev_w": 0.2,                # peso del sleeve de reversión/value
    "rev_lb": 504,               # lookback reversión (~2 años)
}


def _cfg(cfg: dict | None) -> dict:
    c = dict(DEFAULT_CFG)
    if cfg:
        c.update({k: v for k, v in cfg.items() if v is not None})
    return c


def to_wide_close(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Pasa un DataFrame largo (ticker,close, índice fecha) a precios wide."""
    df = ohlcv.reset_index()
    date_col = "date" if "date" in df.columns else df.columns[0]
    wide = df.pivot_table(index=date_col, columns="ticker", values="close")
    return wide.sort_index()


def signal(px: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Convicción [-1,1] por activo: trend (multi-lookback) + reversión opcional."""
    c = _cfg(cfg)
    lookbacks = c["lookbacks"]
    if c["signal_mode"] == "strength":
        ann_vol = px.pct_change().rolling(c["vol_window"]).std() * np.sqrt(252)
        sig = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        for L in lookbacks:
            mom = px / px.shift(L) - 1.0
            horizon_vol = ann_vol * np.sqrt(L / 252.0)
            sig = sig + np.tanh(mom / horizon_vol.replace(0, np.nan))
        trend = (sig / len(lookbacks)).fillna(0.0)
    else:
        sig = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        for L in lookbacks:
            sig = sig + np.sign(px / px.shift(L) - 1.0)
        trend = (sig / len(lookbacks)).fillna(0.0)

    if c["rev_w"] <= 0:
        return trend
    long_ret = px / px.shift(c["rev_lb"]) - 1.0
    mu = long_ret.mean(axis=1)
    sd = long_ret.std(axis=1).replace(0, np.nan)
    rev = (-(long_ret.sub(mu, axis=0)).div(sd, axis=0)).clip(-1, 1).fillna(0.0)
    return (1 - c["rev_w"]) * trend + c["rev_w"] * rev


def weights(px: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Pesos objetivo (con signo) por activo y fecha. Última fila = objetivo actual."""
    c = _cfg(cfg)
    rets = px.pct_change().fillna(0.0)
    ann_vol = rets.rolling(c["vol_window"]).std() * np.sqrt(252)
    sig = signal(px, c)

    raw_w = (sig * (c["per_pos_vol"] / ann_vol.replace(0, np.nan))).clip(
        -c["max_w"], c["max_w"]
    ).fillna(0.0)
    rw = raw_w.to_numpy()
    rmat = rets.to_numpy()
    vw, pvt, maxlev, maxg, rebal, band = (
        c["vol_window"], c["pvt"], c["max_lev"], c["max_gross"], c["rebal"], c["notrade"],
    )

    held = np.zeros_like(rw)
    cur = np.zeros(rw.shape[1])
    for i in range(len(rw)):
        if i % rebal == 0:
            tgt = rw[i].copy()
            if pvt > 0 and i > vw:
                cov = np.cov(rmat[i - vw:i], rowvar=False) * 252.0
                pv = float(np.sqrt(max(tgt @ cov @ tgt, 1e-12)))
                tgt = tgt * (min(maxlev, pvt / pv) if pv > 0 else 0.0)
            else:
                g = np.abs(tgt).sum()
                if g > maxg:
                    tgt = tgt * (maxg / g)
            if band > 0:
                cur = np.where(np.abs(tgt - cur) > band, tgt, cur)
            else:
                cur = tgt
        held[i] = cur
    return pd.DataFrame(held, index=raw_w.index, columns=raw_w.columns)


def backtest(px: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    """Retornos diarios de la estrategia (para validación)."""
    c = _cfg(cfg)
    rets = px.pct_change().fillna(0.0)
    held = weights(px, c)
    pos = held.shift(1).fillna(0.0)
    port_ret = (pos * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    port_ret = port_ret - turnover * c["cost_bps"]
    start_i = max(c["lookbacks"]) + c["vol_window"]
    return port_ret.iloc[start_i:]


def perf_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 5:
        return {"cagr_pct": 0, "vol_pct": 0, "sharpe": 0, "maxdd_pct": 0}
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() * 252 / (r.std() * np.sqrt(252)) if r.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    return {
        "cagr_pct": round(cagr * 100, 2),
        "vol_pct": round(vol * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "maxdd_pct": round(dd * 100, 2),
    }
