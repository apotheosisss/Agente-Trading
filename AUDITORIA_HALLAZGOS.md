# Auditoría rama `feature/polymarket` — Hallazgos

Fecha: 2026-06-01 · Backtest reproducido localmente (`kedro run --pipeline __default__`, 43.6s, datos Yahoo Finance descargados en vivo).

---

## TL;DR

1. **El 51% CAGR del README es la métrica IN-SAMPLE, no el resultado real.** El backtest completo da **23.4% CAGR, Sharpe 0.69, MaxDD -35.5%**.
2. **La estrategia está sobre-ajustada (overfit).** Out-of-sample (2022+) cae a **7.2% CAGR, Sharpe 0.38** — y **pierde contra SPY buy & hold (11.3%)**.
3. **Polymarket no aporta NADA hoy:** las señales salieron **todas en 0.0** (0 mercados relevantes para los 18 tickers). Y por diseño **nunca entró en el backtest**.

---

## 1. Métricas reales vs. lo que dice el README

| Métrica | README | Backtest real (full) | In-sample (2019–2021) | Out-of-sample (2022+) |
|---|---|---|---|---|
| CAGR | **51.1%** | 23.4% | **51.2%** | **7.2%** |
| Sharpe | 1.13 | 0.69 | 1.11 | 0.38 |
| MaxDD | -15.9% | **-35.5%** | -34.2% | -35.5% |
| SPY B&H (CAGR) | — | 16.3% | 23.8% | 11.3% |

**Conclusión:** los números del README (CAGR 51%, Sharpe ~1.1) coinciden exactamente con la ventana **in-sample**. Es cherry-picking de la mejor mitad. El MaxDD -15.9% del README no se reproduce en ninguna ventana (el real es más del doble, -35.5%).

## 2. Sobre-ajuste (la señal más grave)

El propio `calcular_walk_forward` está pensado para detectar esto, y lo detecta:

```
In-sample  (hasta 2022): Sharpe 1.11 | CAGR 51.2% | MaxDD -34.2%
Out-sample (desde 2022): Sharpe 0.38 | CAGR  7.2% | MaxDD -35.5%
```

- Sharpe se desploma de **1.11 → 0.38**. Un sistema robusto mantiene métricas similares; esta caída es la firma clásica de overfitting al periodo 2019–2021 (rally tech + recuperación COVID).
- **Out-of-sample la estrategia rinde 7.2% vs. 11.3% de comprar y mantener SPY.** Es decir: con toda la maquinaria (momentum + LLM + TradingAgents + Polymarket), 40 trades/año y drawdowns del 35%, **se gana menos que comprando SPY y olvidándose.**
- El "23.4% full > 16.3% SPY full" que parece victoria es un artefacto: todo el exceso viene del tramo in-sample.

## 3. Polymarket: dead weight verificado empíricamente

`data/01_raw/polymarket_signals.csv` tras el run real:

```
ticker, poly_score, n_markets, top_market
SPY,    0.0, 0, Sin mercados relevantes
NVDA,   0.0, 0, Sin mercados relevantes
... (los 18 + _macro, TODOS 0.0)
```

Dos hechos independientes lo invalidan como fuente de señal:

1. **No está en el backtest.** Docstring de `polymarket/nodes.py`: *"El backtest historico NO usa estos datos."* La señal Polymarket solo modifica la decisión **de hoy** (`agente_decision`), sumando un `poly_boost` ∈ [-1.5, +1.5] al score técnico. Por tanto, **el 51%/23% no tiene ninguna atribución a Polymarket.** Es imposible saber si ayuda.
2. **Tampoco funciona en vivo.** El matching por keywords (`_KEYWORD_CATEGORIES`) sobre mercados activos con volumen >$5k devolvió **0 coincidencias para los 18 tickers**. Es exactamente el "mapping problem": Polymarket lista eventos (cripto, geopolítica, elecciones), no el precio de NVDA/CVX, y las keywords no cazan nada útil hoy.

Diseño no-backtesteable de raíz: la Gamma API solo expone mercados **activos** (`active=true, closed=false`), no la serie histórica de probabilidades, así que no se puede reconstruir la señal para fechas pasadas.

## 4. Otros hallazgos

- **Incoherencia backtest vs. ejecución en vivo:** el backtest usa `max_positions: 2` (concentración top-2), pero la ejecución paper compró **9 posiciones de $5.000 c/u ($45.000)**. La lógica de sizing/posiciones del path live no respeta la del backtest → lo que se valida no es lo que se opera.
- **El README también difiere en MaxDD** (-15.9% vs -35.5% real): la documentación no corresponde a esta configuración del código.
- El stop-loss es fijo en entrada (no trailing) y el sizing es peso igualitario — decisiones razonables y bien documentadas en el código, pero no rescatan el problema de fondo.

---

## Respuesta a "¿borrar Polymarket?"

**Sí, en su forma actual.** No por la teoría, sino por la evidencia: contribuye exactamente 0.0, no es backtesteable, y añade una dependencia externa que falla silenciosamente (fail-safe a neutral). El ablation test que se planteó **no se puede ni correr** con este diseño, porque la señal no existe históricamente.

Opciones, de menos a más trabajo:
1. **Borrarla / dejarla tras un flag `use_polymarket: false`.** Recupera simplicidad sin perder nada medible.
2. Si se quiere conservar la tesis, hay que **rediseñarla para ser backtesteable**: capturar y persistir snapshots diarios de probabilidades Polymarket a futuro, y solo entonces medir si aportan alfa. Es un proyecto de captura de datos de meses, no un módulo.

## Lo que de verdad necesita atención antes que Polymarket

El problema central **no es Polymarket** — es que **el motor de momentum no tiene edge out-of-sample.** Prioridades:
1. Re-optimizar/validar la estrategia base contra SPY en walk-forward **antes** de añadir capas. Si OOS no bate buy & hold, nada construido encima importa.
2. Corregir el README para reportar la métrica full y OOS, no la in-sample.
3. Reconciliar sizing backtest (2 pos.) vs. ejecución (9 pos.).
4. Recién entonces evaluar señales externas.
