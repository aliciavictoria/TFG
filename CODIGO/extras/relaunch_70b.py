"""
relaunch_70b_last2.py
=====================
Últimos 2 puzzles del Llama-3.3-70B: Kunuz Nubian y Lakota.
Usa SambaNova en vez de Groq para aprovechar su cuota independiente.

Estimado: ~40 llamadas (M0+M1+M2×15+M3×4+M4+M5×2 por puzzle).

Uso:
    export SAMBANOVA_API_KEY="tu_clave"
    python3 relaunch_70b_last2.py
"""
import os
import experiment_runner_row as runner

runner.RAW_LOG_DIR = "raw_logs"   # append al directorio común

API_KEYS = {
    "groq":       os.getenv("GROQ_API_KEY", ""),
    "cerebras":   os.getenv("CEREBRAS_API_KEY", ""),
    "sambanova":  os.getenv("SAMBANOVA_API_KEY", ""),
    "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
}

MODEL = "llama-3.3-70b-sambanova"

TARGETS = [
    # ("puzzles/linguini_012016030100.json", (0,1,2,3,4,5), False),  # Kunuz Nubian
    ("puzzles/linguini_012006010100.json", (4,), False),  # Lakota
]

print(f"Llama-3.3-70B vía SambaNova — 2 puzzles, ~40 llamadas\n")

for puzzle_path, strategies, is_hakhun in TARGETS:
    runner.run_experiment(
        puzzle_path      = puzzle_path,
        model_key        = MODEL,
        api_keys         = API_KEYS,
        strategies       = strategies,
        output_dir       = "results",
        human_resolution = "" if is_hakhun else runner.HUMAN_RESOLUTION_HAKHUN,
    )