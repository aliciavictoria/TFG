"""
run_en2lang.py
==============
Experimento en dirección inversa: inglés → lengua desconocida (EN→lang).

Puzzles seleccionados (3 familias lingüísticas distintas):
  Hakhun   (Sino-tibetana)  — comparación directa con los resultados lang2en
  Yonggom  (Trans-NG)       — ídem
  N|uuki   (Tuu)            — lengua de chasquidos, la más tipológicamente distinta

Estrategias: M0 (verificación) + M1 (baseline)
  Son solo 2 llamadas × 3 puzzles × 3 modelos = 18 llamadas en total.
  Rápido y suficiente para la sección "Otras pruebas" de la memoria.

Los logs se escriben en raw_logs/ junto con el resto del experimento (append).
Cuando terminen, subir raw_logs/ y recalcular métricas con rescore_from_raw.py.

Uso:
    export GROQ_API_KEY="..."
    export SAMBANOVA_API_KEY="..."
    export OPENROUTER_API_KEY="..."
    python3 run_en2lang.py
"""
import os
import experiment_runner_row as runner

runner.RAW_LOG_DIR = "raw_logs"   # append al directorio común

API_KEYS = {
    "groq":       os.getenv("GROQ_API_KEY", ""),
    "sambanova":  os.getenv("SAMBANOVA_API_KEY", ""),
    "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
    "cerebras":   os.getenv("CEREBRAS_API_KEY", ""),
}

PUZZLES = [
    "puzzles/linguini_012018020200.json",   # Hakhun  EN→lang  (Sino-tibetana)
    "puzzles/linguini_012019010200.json",   # Yonggom EN→lang  (Trans-New Guinea)
    "puzzles/linguini_012022030200.json",   # N|uuki  EN→lang  (Tuu / chasquidos)
]

# Los tres modelos del experimento principal — así la comparación es directa
MODELS = [
    "llama-3.1-8b-groq",
    "llama-3.3-70b-sambanova",
    "gpt-oss-120b-openrouter",
]

# M0 (verificación) + M1 (baseline)
# M4 EXCLUIDO deliberadamente: el razonamiento humano de Hakhun describe la
# dirección lang→EN, no tiene sentido como guía para producir morfología
# en la lengua desconocida. human_resolution se deja vacío por coherencia.
STRATEGIES = (0, 1)

available = {k: v for k, v in runner.MODELS.items()
             if API_KEYS.get(v["provider"])}
print("\nModelos disponibles:")
for k, v in available.items():
    print(f"  ✓ {k:<35} ({v['tier']})")

print(f"\nLanzando {len(PUZZLES)} puzzles × {len(MODELS)} modelos × "
      f"{len(STRATEGIES)} estrategias = "
      f"{len(PUZZLES)*len([m for m in MODELS if m in available])*len(STRATEGIES)} llamadas\n")

for puzzle_path in PUZZLES:
    for model_key in MODELS:
        if model_key not in available:
            print(f"  [SKIP] {model_key} — API key no disponible")
            continue
        runner.run_experiment(
            puzzle_path      = puzzle_path,
            model_key        = model_key,
            api_keys         = API_KEYS,
            strategies       = STRATEGIES,
            output_dir       = "results_en2lang",
            human_resolution = "",   # M4 no aplica en dirección en2lang
        )