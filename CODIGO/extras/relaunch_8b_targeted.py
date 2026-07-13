"""
relaunch_8b_final.py
====================
Último relanzamiento del Llama-3.1-8B: 7 casos restantes.

Casos:
  Ngemba        M1  — bucle de repetición persistente
  Yonggom       M4  — bucle de repetición persistente
  Warlpiri      M4  — bucle de repetición persistente
  Inuktitut     M4  — bucle de repetición persistente
  Dyirbal       M4  — ausente (error HTTP en el primer lanzamiento)
  Tzeltal       M4  — ausente (error HTTP en el primer lanzamiento)
  Hakhun        M5  — T1 agotó tokens analizando sin llegar a TRANSLATIONS

Los nuevos intentos se añaden (append) a los .jsonl existentes en raw_logs/.
rescore_from_raw.py usará siempre el intento más reciente.

Uso:
    export GROQ_API_KEY="tu_clave"
    python3 relaunch_8b_final.py
"""
import os
from experiment_runner_row import run_experiment, HUMAN_RESOLUTION_HAKHUN

API_KEYS = {
    "groq":       os.getenv("GROQ_API_KEY", ""),
    "cerebras":   os.getenv("CEREBRAS_API_KEY", ""),
    "sambanova":  os.getenv("SAMBANOVA_API_KEY", ""),
    "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
}

MODEL = "llama-3.1-8b-groq"

TARGETS = [
    # Ngemba M1 — bucle en sección DICTIONARY
    ("puzzles/linguini_012022030100.json", (1,), False),

    # M4 ausentes / bucle — 5 puzzles
    ("puzzles/linguini_012019010100.json", (4,), False),  # Yonggom
    ("puzzles/linguini_012015040100.json", (4,), False),  # Warlpiri
    ("puzzles/linguini_012008050100.json", (4,), False),  # Inuktitut
    ("puzzles/linguini_012012010200.json", (4,), False),  # Dyirbal
    # ("puzzles/linguini_012005010100.json", (4,), False),  # Tzeltal

    # Hakhun M5 — T1 agotó tokens sin llegar a TRANSLATIONS
    # (is_hakhun=True → M4 se omite; M5 sí se ejecuta)
    # ("puzzles/linguini_012018020100.json", (5,), True),
]

print(f"Relanzando {len(TARGETS)} casos finales sobre Llama-3.1-8B...\n")

for puzzle_path, strategies, is_hakhun in TARGETS:
    run_experiment(
        puzzle_path      = puzzle_path,
        model_key        = MODEL,
        api_keys         = API_KEYS,
        strategies       = strategies,
        output_dir       = "results",
        human_resolution = "" if is_hakhun else HUMAN_RESOLUTION_HAKHUN,
    )