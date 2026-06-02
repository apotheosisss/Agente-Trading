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

## Ejecución en Alpaca (paper) — YA IMPLEMENTADA

Adaptador long/short por **órdenes delta** (lleva la cuenta de la posición actual a
la objetivo). Pipeline `alpaca_tsmom` / `tsmom_trade`:

```bash
# Calcular señales Y ejecutar en Alpaca paper (ingestión + tsmom + ejecución)
uv run kedro run --pipeline tsmom_trade

# Solo ejecutar (con tsmom_weights ya calculado)
uv run kedro run --pipeline alpaca_tsmom
```

- `nodes.ejecutar_tsmom_alpaca`: para cada activo calcula posición objetivo USD
  (`equity * peso`), normaliza a exposición bruta segura, y envía la orden delta
  (BUY/SELL/CLOSE) hacia el objetivo. Soporta **largos y cortos** (cripto solo largo).
- Salida: `data/07_model_output/tsmom_alpaca_log.csv`.
- **Seguridad:** paper salvo `paper_trading:false` explícito; `live_gross=1.0`
  (sin apalancamiento por defecto); cap por posición `live_max_position_pct=0.30`;
  órdenes < `min_order_usd` ($25) omitidas; cada orden en try/except aislado.
- Config en `parameters.yml → tsmom:` (`live_gross`, `live_max_position_pct`, `min_order_usd`).

⚠️ El pipeline `alpaca` viejo (long-only) sigue existiendo pero **no debe usarse** con TSMOM.

## PENDIENTE antes de subir apalancamiento / dinero real (NO omitir)

1. **Verificar en paper:** correr `tsmom_trade` unos días y revisar `tsmom_alpaca_log.csv`
   y las posiciones (algunos ETFs pueden no ser shortables en paper → la orden se
   registra como error y se continúa; revisar cuáles).
2. **Validación de robustez:** walk-forward purgado, costes pesimistas (borrow/financiación).
3. **Paper trading 60-90 días** midiendo *tracking error* vs backtest.
4. **Apalancamiento:** subir `live_gross` (p.ej. 1.3-1.5) solo tras validar; dimensionar
   con cuidado (margin/gap). El blend recomendado es 50/50 con SPY.

## Congelado (no aporta alfa, confirmado)
LLM, TradingAgents y Polymarket quedan fuera de la ruta de producción TSMOM.
El pipeline legacy (`signals`, `backtesting`) permanece para referencia, no para operar.
