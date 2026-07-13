"""
relaunch_targeted.py
====================
Relanza ÚNICAMENTE los 9 casos fallidos identificados tras revisar los
raw_logs del primer lanzamiento de GPT-OSS-120B (→EN).

Fallos y causa:
  Hakhun        M1 + M3   Error HTTP — call_model lanzó excepción antes de _log_raw
  Ngemba        M1        Bucle de repetición en sección DICTIONARY
  Yonggom       M1        Bucle de repetición en sección DICTIONARY
  Coastal Marind M1 + M4  Bucle de repetición (M1) / respuesta truncada (M4)
  Warlpiri      M5        El turno 2 tradujo al warlpiri en vez de al inglés

Correcciones ya aplicadas en experiment_runner_v5.py:
  - _has_repetition() + reintento con temperature=0.2 (reemplaza frequency_penalty)
  - Instrucción de dirección explícita ("translate into English") en M5 turno 2
  - Detección del fallo y reintento automático

Como siempre: _log_raw() hace APPEND, nunca sobreescribe.
Los intentos fallidos anteriores se conservan como registro histórico.

Uso:
    export OPENROUTER_API_KEY="tu_clave"
    python3 relaunch_targeted.py
"""
import os
from experiment_runner_row import run_experiment, HUMAN_RESOLUTION_HAKHUN

API_KEYS = {
    "groq":       os.getenv("GROQ_API_KEY", ""),
    "cerebras":   os.getenv("CEREBRAS_API_KEY", ""),
    "sambanova":  os.getenv("SAMBANOVA_API_KEY", ""),
    "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
}

MODEL = "gpt-oss-120b-openrouter"

# Cada entrada: (ruta_puzzle, tupla_estrategias, es_hakhun)
# is_hakhun=True → M4 se omite (evitar data leakage, es el puzzle fuente)
TARGETS = [
    # Hakhun: M1 (baseline=1) + M3 (step_by_step=3)
    # M4 omitido: Hakhun es el puzzle cuyo razonamiento se usa como guía
    ("puzzles/linguini_012018020100.json", (1,), True),

    # Ngemba: solo M1
    #("puzzles/linguini_012022030100.json", (1,), False),

    # Yonggom: solo M1
    #("puzzles/linguini_012019010100.json", (1,), False),

    # Coastal Marind: M1 + M4
    #("puzzles/linguini_012023030100.json", (1, 4), False),

    # Warlpiri: solo M5 (self_correction=5)
    #("puzzles/linguini_012015040100.json", (5,), False),
]

for puzzle_path, strategies, is_hakhun in TARGETS:
    run_experiment(
        puzzle_path      = puzzle_path,
        model_key        = MODEL,
        api_keys         = API_KEYS,
        strategies       = strategies,
        output_dir       = "results",
        human_resolution = "" if is_hakhun else HUMAN_RESOLUTION_HAKHUN,
    )