"""
run_m22_combinaciones.py  —  M2.2: Exploración combinatoria por tamaños crecientes
====================================================================================
Implementación de la variante combinatoria descrita en la memoria (§ 4.3.5).

Diferencia con M2 (prefijos):
  M2:   (f1,f2) → (f1,f2,f3) → (f1,f2,f3,f4) → ...   [solo prefijos]
  M2.2: (f1,f2) → (f1,f3) → ... → (f2,f3) → ...       [TODAS las parejas]
        → (f1,f2,f3) → (f1,f2,f4) → ...                [TODOS los triples]
        → ...

Diseño del estado acumulado (decisión documentada):
  El diccionario y reglas se acumulan a través de TODAS las iteraciones, en orden.
  Cada nueva combinación puede confirmar, refinar o contradecir la hipótesis vigente.
  Esto es equivalente a "M2 exhaustivo": el modelo ve todas las combinaciones posibles
  de tamaño k antes de pasar a tamaño k+1.
  Esta decisión implica que el resultado depende del orden de procesamiento de combos,
  lo que constituye en sí mismo un resultado de interés (sensibilidad al orden).

Comparación directa con M2:
  M2 usa el PRIMER prefijo de cada tamaño (siempre f1,...,fk).
  M2.2 usa TODOS los subconjuntos de tamaño k. Si M2 ya encuentra el mejor subconjunto
  en su prefijo fijo, M2.2 no aporta mejora. Si el prefijo de M2 no es óptimo,
  M2.2 puede encontrar combinaciones mejores.

Coste computacional (Hakhun, n=10):
  MAX_K=2 →  45 iteraciones (~6 min)
  MAX_K=3 → 165 iteraciones (~22 min)  ← configuración por defecto
  MAX_K=4 → 375 iteraciones (~50 min)

Uso:
    export OPENROUTER_API_KEY="..."
    python3 run_m22_combinaciones.py
"""

import json, os, sys, time, itertools, datetime
from pathlib import Path

# Importamos todo lo que ya está verificado y corregido
import experiment_runner_row as runner

# ─── Configuración ─────────────────────────────────────────────────────────────
PUZZLE_PATH = "puzzles/linguini_012018020100.json"   # Hakhun →EN (n=10 train)
MODEL_KEY   = "gpt-oss-120b-openrouter"              # mismo modelo que el resto de experimentos
MAX_K       = 3                                       # k=2 (45 combos) + k=3 (120 combos) = 165 total
OUTPUT_DIR  = "results_m22_combinaciones"
RAW_LOG_DIR = "raw_logs_m22"                         # separado de raw_logs/ del experimento principal

API_KEYS = {
    "groq":       os.getenv("GROQ_API_KEY", ""),
    "sambanova":  os.getenv("SAMBANOVA_API_KEY", ""),
    "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
    "cerebras":   os.getenv("CEREBRAS_API_KEY", ""),
}

# ─── Log crudo (mismo mecanismo que experiment_runner_v5) ──────────────────────

def _log_raw(puzzle_id, model_key, k, combo, prompt, response):
    """Guarda cada llamada antes de procesar nada — si el extractor falla,
    la respuesta original sigue disponible."""
    Path(RAW_LOG_DIR).mkdir(exist_ok=True)
    fname = f"{RAW_LOG_DIR}/{puzzle_id}_{model_key}_m22.jsonl"
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "strategy":  "m2_combinaciones",
        "k":         k,
        "combo":     list(combo),
        "prompt":    prompt,
        "response":  response,
    }
    with open(fname, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ─── Runner M2.2 ───────────────────────────────────────────────────────────────

def run_m22(puzzle_path, model_key, api_keys, max_k=3, output_dir=OUTPUT_DIR):
    with open(puzzle_path) as f:
        puzzle = json.load(f)

    train      = puzzle["train_pairs"]
    test_items = puzzle["test_pairs"]
    refs       = [p["target"] for p in test_items]
    n          = len(train)
    direction  = puzzle.get("translation_direction", "lang2en")
    lang       = puzzle["language"]
    pid        = puzzle["id"]
    max_k      = min(max_k, n)

    # Número total de iteraciones
    total_iters = sum(
        len(list(itertools.combinations(range(n), k)))
        for k in range(2, max_k + 1)
    )

    trans_instr = runner.translation_instruction(direction, lang)
    tests       = "\n".join(
        f"  {i+1}. {item['source']}" for i, item in enumerate(test_items)
    )

    print(f"\n{'='*62}")
    print(f"  M2.2 COMBINACIONES — {lang} ({direction})")
    print(f"  Modelo:  {model_key}  ({runner.MODELS[model_key]['tier']})")
    print(f"  n={n} frases · k=2..{max_k} · {total_iters} iteraciones")
    print(f"  Puzzles seleccionado por comparabilidad: todos los demás")
    print(f"  modelos tienen M1/M2/M3/M4/M5 para este puzzle.")
    print(f"{'='*62}")

    steps         = []
    current_dict  = ""
    current_rules = ""
    best_chrf     = -1
    best_step     = None
    iter_count    = 0
    errors_count  = 0

    for k in range(2, max_k + 1):
        combos = list(itertools.combinations(range(n), k))
        print(f"\n  ── k={k}: {len(combos)} combinaciones ──")

        for combo in combos:
            iter_count += 1
            combo_pairs = [train[i] for i in combo]
            frase_ids   = [i + 1 for i in combo]
            examples    = "\n".join(
                f"  {p['source']} -> {p['target']}" for p in combo_pairs
            )

            # Hipótesis acumulada del paso anterior
            _d = runner._truncate_context(current_dict)  if current_dict  else "(empty)"
            _r = runner._truncate_context(current_rules) if current_rules else "(empty)"
            prev_block = f"""
--- YOUR CURRENT HYPOTHESIS (provisional — review critically) ---
DICTIONARY:
{_d}

GRAMMAR RULES:
{_r}

IMPORTANT: New examples may CONFIRM, REFINE or CONTRADICT entries above.
Delete proven-wrong entries; correct imprecise ones.
""" if (current_dict or current_rules) else "\n(No prior knowledge — first iteration.)\n"

            prompt = f"""{runner.METALINGUISTIC_INSTRUCTION}

ITERATION {iter_count}/{total_iters} — Subset of {k} examples (indices {frase_ids}):
{prev_block}
--- NEW EXAMPLES (this combination) ---
{examples}

--- YOUR TASK (in this order) ---

STEP A · ANALYSIS
Identify morphemes in each new example. Check for contradictions with your
current hypothesis. Think aloud. Do NOT translate the test sentences yet.

STEP B · REVISED DICTIONARY
One entry per line: morpheme = meaning (notes)
Mark deleted entries: [DELETED: entry — reason]

STEP C · REVISED GRAMMAR RULES
Numbered list. Mark deleted rules: [DELETED: rule — reason]

STEP D · TRANSLATIONS
{trans_instr.capitalize()} — write ONLY the translation on each line, nothing else:
{tests}

Format your response EXACTLY as:
ANALYSIS:
[reasoning]

DICTIONARY:
[entries]

GRAMMAR RULES:
[rules]

{runner._translations_format_block(len(test_items))}"""

            print(f"  [{iter_count:3d}/{total_iters}] k={k} frases={frase_ids}", end=" ", flush=True)
            t0 = time.time()

            try:
                # ── Llamada al modelo con detección de bucles ──────────────────
                response = runner.call_model(prompt, model_key, api_keys)
                latency  = round(time.time() - t0, 2)

                # ── Log crudo ANTES de extraer nada ───────────────────────────
                _log_raw(pid, model_key, k, combo, prompt, response)

                # ── Actualizar hipótesis acumulada ─────────────────────────────
                nd, nr = runner.extract_dict_and_rules(response)
                if nd: current_dict  = nd
                if nr: current_rules = nr

                # ── Extraer traducciones con el extractor verificado ──────────
                hyps    = runner.extract_translations(response, len(test_items))
                metrics = runner.compute_metrics(hyps, refs)  # penaliza ausentes con ""
                chrf    = metrics["corpus_chrfpp"]
                n_ok    = metrics["num_evaluated"]

                print(f"chrF={chrf:5.1f}  ({n_ok}/{len(test_items)} hyps)  {latency:.1f}s")

                step = {
                    "iter":       iter_count,
                    "k":          k,
                    "combo":      list(combo),
                    "frase_ids":  frase_ids,
                    "latency_s":  latency,
                    "metrics":    metrics,
                    "accumulated_dict":  current_dict,
                    "accumulated_rules": current_rules,
                }
                steps.append(step)

                if chrf > best_chrf:
                    best_chrf = chrf
                    best_step = step
                    print(f"      ★ nuevo mejor: chrF={chrf:.1f}  hyps={metrics.get('hypotheses', [])}")

            except Exception as e:
                print(f"ERR: {e}")
                errors_count += 1
                steps.append({
                    "iter": iter_count, "k": k,
                    "combo": list(combo), "frase_ids": frase_ids,
                    "error": str(e),
                })

            time.sleep(8)   # respetar rate limits

    # ── Guardar resultados ─────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().isoformat()

    # Estadísticas por k
    stats_by_k = {}
    import math
    for k in range(2, max_k + 1):
        k_steps = [s for s in steps if s.get("k") == k and "error" not in s]
        if k_steps:
            chrfs = [s["metrics"]["corpus_chrfpp"] for s in k_steps]
            mu    = sum(chrfs) / len(chrfs)
            sigma = math.sqrt(sum((x - mu) ** 2 for x in chrfs) / len(chrfs))
            stats_by_k[k] = {
                "n_combos":    len(k_steps),
                "mean_chrfpp": round(mu, 2),
                "std_chrfpp":  round(sigma, 2),
                "max_chrfpp":  round(max(chrfs), 2),
                "min_chrfpp":  round(min(chrfs), 2),
                "best_combo":  k_steps[chrfs.index(max(chrfs))]["frase_ids"],
                "worst_combo": k_steps[chrfs.index(min(chrfs))]["frase_ids"],
            }

    log = {
        "metadata": {
            "timestamp":     timestamp,
            "puzzle_id":     pid,
            "language":      lang,
            "direction":     direction,
            "model_key":     model_key,
            "model_tier":    runner.MODELS[model_key]["tier"],
            "n_train":       n,
            "max_k":         max_k,
            "total_iters":   total_iters,
            "completed":     len([s for s in steps if "error" not in s]),
            "errors":        errors_count,
            "strategy":      "m2_combinaciones",
            "design_note":   (
                "El diccionario acumulado persiste a través de TODAS las iteraciones "
                "en orden (k=2 completo → k=3 completo → ...). Cada combinación "
                "refina la hipótesis vigente. El orden de procesamiento es lexicográfico."
            ),
        },
        "stats_by_k":  stats_by_k,
        "best_chrf":   round(best_chrf, 2),
        "best_step":   best_step,
        "steps":       steps,
    }

    Path(output_dir).mkdir(exist_ok=True)
    fname = f"{output_dir}/m22_{pid}_{model_key}_{timestamp[:10]}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # ── Resumen final ──────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"✓ Guardado: {fname}")
    print(f"\nRESUMEN M2.2 — {lang}:")
    print(f"  Iteraciones completadas: {log['metadata']['completed']}/{total_iters}")
    print(f"  Mejor chrF++ global:     {best_chrf:.1f}")
    if best_step:
        print(f"  Mejor combinación:       k={best_step['k']}  frases={best_step['frase_ids']}")
    print(f"\n  Estadísticas por tamaño k:")
    for k, s in stats_by_k.items():
        print(f"    k={k}: μ={s['mean_chrfpp']:.1f}  σ={s['std_chrfpp']:.1f}  "
              f"max={s['max_chrfpp']:.1f} {s['best_combo']}  "
              f"min={s['min_chrfpp']:.1f} {s['worst_combo']}")
    return log


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    available = {k: v for k, v in runner.MODELS.items()
                 if API_KEYS.get(v["provider"])}
    print("\nModelos disponibles:")
    for k, v in available.items():
        print(f"  ✓ {k:<35} ({v['tier']})")

    if MODEL_KEY not in available:
        print(f"\nERROR: {MODEL_KEY} no disponible. Configura la API key.")
        sys.exit(1)

    if not Path(PUZZLE_PATH).exists():
        print(f"\nERROR: no encontrado {PUZZLE_PATH}")
        sys.exit(1)

    run_m22(
        puzzle_path = PUZZLE_PATH,
        model_key   = MODEL_KEY,
        api_keys    = API_KEYS,
        max_k       = MAX_K,
        output_dir  = OUTPUT_DIR,
    )