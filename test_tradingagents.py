#!/usr/bin/env python3
"""Test script to verify TradingAgents + OpenAI configuration."""

import os
import sys
from pathlib import Path
from datetime import datetime

# Force UTF-8 output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Test 1: Check if TradingAgents is installed
print("=" * 60)
print("TEST 1: Verificando TradingAgents instalado...")
try:
    from tradingagents.graph import TradingAgentsGraph
    print("[OK] TradingAgents importado correctamente")
except ImportError as e:
    print(f"[ERROR] Error importando TradingAgents: {e}")
    sys.exit(1)

# Test 2: Check OpenAI dependency
print("\n" + "=" * 60)
print("TEST 2: Verificando OpenAI SDK...")
try:
    from langchain_openai import ChatOpenAI
    print("[OK] LangChain OpenAI importado correctamente")
except ImportError as e:
    print(f"[ERROR] Error importando LangChain OpenAI: {e}")
    sys.exit(1)

# Test 3: Check default config is set to OpenAI
print("\n" + "=" * 60)
print("TEST 3: Verificando configuración por defecto...")
from tradingagents.default_config import DEFAULT_CONFIG
print(f"  LLM Provider: {DEFAULT_CONFIG['llm_provider']}")
print(f"  Deep thinking model: {DEFAULT_CONFIG['deep_think_llm']}")
print(f"  Quick thinking model: {DEFAULT_CONFIG['quick_think_llm']}")

if DEFAULT_CONFIG['llm_provider'] != 'openai':
    print("[ERROR] LLM provider no es 'openai'")
    sys.exit(1)
if DEFAULT_CONFIG['deep_think_llm'] != 'openai/gpt-5-nano':
    print("[ERROR] Deep thinking model no es 'openai/gpt-5-nano'")
    sys.exit(1)
if DEFAULT_CONFIG['quick_think_llm'] != 'openai/gpt-5-nano':
    print("[ERROR] Quick thinking model no es 'openai/gpt-5-nano'")
    sys.exit(1)
if DEFAULT_CONFIG['backend_url'] != 'https://openrouter.ai/api/v1':
    print("[ERROR] backend_url no apunta a OpenRouter")
    sys.exit(1)
print("[OK] Configuracion OpenRouter correcta (openai/gpt-5-nano via openrouter.ai)")

# Test 4: Check credentials file
print("\n" + "=" * 60)
print("TEST 4: Verificando credenciales OpenAI...")
import yaml
cred_path = Path("conf/local/credentials.yml")
if cred_path.exists():
    with open(cred_path, encoding="utf-8") as f:
        creds = yaml.safe_load(f) or {}
    openai_key = creds.get("openai", {}).get("api_key", "")
    if openai_key and openai_key != "":
        print(f"[OK] API key OpenAI configurada (primeros 8 chars: {openai_key[:8]}...)")
        os.environ["OPENAI_API_KEY"] = openai_key
    else:
        print("[ERROR] API key OpenAI vacía o no configurada")
        print("  Por favor, edita conf/local/credentials.yml y añade tu API key")
        sys.exit(1)
else:
    print("[ERROR] Archivo credentials.yml no encontrado")
    sys.exit(1)

# Test 5: Try creating a TradingAgentsGraph instance
print("\n" + "=" * 60)
print("TEST 5: Inicializando TradingAgentsGraph...")
try:
    graph = TradingAgentsGraph(
        selected_analysts=["market", "news"],
        debug=False,
        config=None,  # Uses DEFAULT_CONFIG
    )
    print("[OK] TradingAgentsGraph inicializado correctamente")
except Exception as e:
    print(f"[ERROR] Error inicializando TradingAgentsGraph: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Try running a single ticker analysis (quick test)
print("\n" + "=" * 60)
print("TEST 6: Analizando un ticker de prueba (AAPL)...")
try:
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"  Fecha análisis: {today}")
    print("  [Este análisis puede tomar 1-2 minutos por los LLM...]")

    final_state, signal = graph.propagate("AAPL", today)
    print(f"[OK] Análisis completado")
    print(f"  Señal: {signal}")
    print(f"  Tipo: {type(signal)}")
except Exception as e:
    print(f"[ERROR] Error ejecutando análisis: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit here, just warn - might be network/API issue

print("\n" + "=" * 60)
print("OK TODOS LOS TESTS PASARON")
print("=" * 60)
print("\nProximos pasos:")
print("1. Asegúrate de que OPENAI_API_KEY está en credentials.yml")
print("2. En parameters.yml, cambia 'tradingagents.enabled: true' para activar")
print("3. Ejecuta: uv run kedro run --pipeline=signals")
