# Implementación TSMOM en el pipeline (producción)

La estrategia ganadora (ver `OPTIMIZACION_RESULTADOS.md`) está integrada en Kedro
como pipeline de producción. Larga/corta, multi-activo, vol-targeting + reversión.

## Qué se añadió

| Componente | Ruta |
|---|---|
| Núcleo de estrategia (fuente única) | `src/trading_agent/pipelines/tsmom/strategy.py` |
| Nodos Kedro | `src/trading_agent/pipelines/tsmom/nodes.py` |
| Pipeline | `src/trading_agent/pipelines/tsmom/pipeline.py` |
| Config | sección `tsmom:` en `conf/base/parameters.yml` |
| Universo | 20 activos descorrelacionados (`universe:` en parameters.yml) |
| Datasets | `tsmom_weights`, `tsmom_orders`, `tsmom_report`, `tsmom_validation` en `catalog.yml` |
| Pipeline registrado | `tsmom_live` (ingestión + tsmom) en `pipeline_registry.py` |

## Cómo se ejecuta

```bash
# Pipeline de producción completo (descarga universo → pesos → órdenes → validación)
uv run kedro run --pipeline tsmom_live

# Solo la estrategia con datos ya cacheados (rápido)
uv run kedro run --pipeline tsmom
```

**Salidas:**
- `data/07_model_output/tsmom_weights.csv` — pesos objetivo (con signo) por activo, hoy.
- `data/07_model_output/tsmom_orders.csv` — posiciones objetivo en USD (versionado).
- `data/08_reporting/tsmom_report.txt` — reporte legible.
- `data/08_reporting/tsmom_validation.csv` — métricas backtest (full + OOS) para monitoreo.

## Flujo

```
ingestion (yfinance, 20 activos)
   └─ clean_ohlcv
        └─ calcular_pesos_tsmom  → tsmom_weights  (largo/corto, vol-targeting + reversión)
             ├─ generar_reporte_tsmom → tsmom_report
             ├─ generar_ordenes_tsmom → tsmom_orders (posición objetivo USD)
             └─ validar_tsmom        → tsmom_validation (autochequeo)
```

## Configuración (parameters.yml → `tsmom:`)
Config validada por defecto. Para ajustar (con cuidado, revalidando):
`lookbacks`, `signal_mode` (strength), `pvt` (vol cartera 0.12), `rev_w` (0.2), `rebal` (21d).

## Validación reproducida en el pipeline
`tsmom_validation.csv`: FULL Sharpe **0.84**, CAGR 11.6%, MaxDD -25% | OOS Sharpe **0.68**.
Coincide con la investigación → el port a producción es fiel.

## PENDIENTE antes de operar con dinero real (NO omitir)

1. **Adaptador de ejecución Alpaca long/short.** `tsmom_orders` da la posición OBJETIVO.
   Falta un nodo que lea la posición ACTUAL en Alpaca y envíe la orden delta
   (objetivo − actual), soportando cortos (o ETFs inversos donde no haya borrow).
   ⚠️ El pipeline `alpaca` existente es **long-only y NO es compatible** — no usarlo con TSMOM.
2. **Validación de robustez:** walk-forward purgado, costes pesimistas (borrow/financiación
   reales), test de universo perturbado.
3. **Paper trading 60-90 días** midiendo *tracking error* vs backtest antes de capital real.
4. **Apalancamiento:** el blend recomendado es 50/50 con SPY a 1.0–1.3×. Dimensionar con
   cuidado (margin/gap). El stream TSMOM solo es el overlay.

## Congelado (no aporta alfa, confirmado)
LLM, TradingAgents y Polymarket quedan fuera de la ruta de producción TSMOM.
El pipeline legacy (`signals`, `backtesting`) permanece para referencia, no para operar.
