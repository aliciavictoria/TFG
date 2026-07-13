"""
run_otras_tareas.py
===================
Experimentos con tareas no-traducción del benchmark Linguini:

  1. Guazacapán Xinka (fill_blanks): dado form1 y significado, inferir form2.
     Morfología verbal de lengua amerindia aislante (Guatemala).

  2. Supyire text_to_num: convertir palabras numéricas a dígitos.
     Sistema numérico de lengua Gur (Malí).

  3. Supyire num_to_text: convertir dígitos a palabras numéricas (dirección inversa).

Métricas: chrF++ (para comparabilidad) + Exact Match (métrica principal para numerales).

Estrategia: M0 (verificación) + M1 (baseline).
Modelos: los tres del experimento principal.

Uso:
    export GROQ_API_KEY="..."
    export SAMBANOVA_API_KEY="..."
    export OPENROUTER_API_KEY="..."
    python3 run_otras_tareas.py
"""

import json, os, sys, time, datetime, re
from pathlib import Path
import experiment_runner_row as runner

runner.RAW_LOG_DIR = "raw_logs_otras_tareas"

API_KEYS = {
    "groq":       os.getenv("GROQ_API_KEY", ""),
    "sambanova":  os.getenv("SAMBANOVA_API_KEY", ""),
    "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
    "cerebras":   os.getenv("CEREBRAS_API_KEY", ""),
}

MODELS = [
    "llama-3.1-8b-groq",
    "llama-3.3-70b-sambanova",
    "gpt-oss-120b-openrouter",
]

OUTPUT_DIR = "results_otras_tareas"
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(runner.RAW_LOG_DIR).mkdir(exist_ok=True)

available = {k: v for k, v in runner.MODELS.items()
             if API_KEYS.get(v["provider"])}

print("\nModelos disponibles:")
for k, v in available.items():
    print(f"  ✓ {k:<35} ({v['tier']})")


# ─── Prompt builders por tipo de tarea ─────────────────────────────────────────

def prompt_fill_blanks(puzzle, model_key):
    """Guazacapán Xinka: dado form1 y significado → form2."""
    train = puzzle["train_pairs"]
    test  = puzzle["test_pairs"]
    lang  = puzzle["language"]

    examples = "\n".join(
        f"  {p['source'].split(' | ')[0]} → {p['target']}   ({p['source'].split(' | ')[1]})"
        for p in train
    )
    tests = "\n".join(
        f"  {i+1}. {p['source'].split(' | ')[0]}   ({p['source'].split(' | ')[1]})"
        for i, p in enumerate(test)
    )

    return f"""{runner.METALINGUISTIC_INSTRUCTION}

You are analyzing morphological alternations in {lang}, an unknown language.
Each example shows two different forms of the same verb plus its English meaning.

--- EXAMPLES: Form1 → Form2 (meaning) ---
{examples}

--- YOUR TASK ---
Identify the morphological pattern that transforms Form1 into Form2.
Then apply it to produce Form2 for each item below.

{tests}

Format your response EXACTLY as:
ANALYSIS:
[describe the morphological pattern you identified]

{runner._translations_format_block(len(test), label="Form2 only — no explanations")}"""


def prompt_text_to_num(puzzle, model_key):
    """Supyire: palabras numéricas → dígitos."""
    train = puzzle["train_pairs"]
    test  = puzzle["test_pairs"]
    lang  = puzzle["language"]

    examples = "\n".join(f"  {p['source']} = {p['target']}" for p in train)
    tests    = "\n".join(f"  {i+1}. {p['source']}" for i, p in enumerate(test))

    return f"""{runner.METALINGUISTIC_INSTRUCTION}

You are analyzing the numeral system of {lang}, an unknown language.
The examples show {lang} number words and their numerical values.

--- EXAMPLES ---
{examples}

--- YOUR TASK ---
Infer the numeral system (base, morphemes for each power, additive/multiplicative rules).
Then convert each expression below to digits.

{tests}

{runner._translations_format_block(len(test), label="digit value only (e.g. 810)")}"""


def prompt_num_to_text(puzzle, model_key):
    """Supyire: dígitos → palabras numéricas."""
    train = puzzle["train_pairs"]
    test  = puzzle["test_pairs"]
    lang  = puzzle["language"]

    examples = "\n".join(f"  {p['source']} = {p['target']}" for p in train)
    tests    = "\n".join(f"  {i+1}. {p['source']}" for i, p in enumerate(test))

    return f"""{runner.METALINGUISTIC_INSTRUCTION}

You are analyzing the numeral system of {lang}, an unknown language.
The examples show numerical values and their {lang} expressions.

--- EXAMPLES ---
{examples}

--- YOUR TASK ---
Infer the numeral system (morphemes for units, tens, hundreds; additive/multiplicative rules).
Then express each number below in {lang}.

{tests}

{runner._translations_format_block(len(test), label=f"{lang} numeral expression only")}"""


PROMPT_BUILDERS = {
    "fill2lang": prompt_fill_blanks,
    "text2num":  prompt_text_to_num,
    "num2text":  prompt_num_to_text,
}

# ─── M0 personalizado para tareas no-traducción ────────────────────────────────

def run_m0_task(puzzle, model_key):
    """Verificación: ¿conoce el modelo esta lengua/sistema?"""
    task_dir  = puzzle.get("translation_direction", "fill2lang")   # fill2lang / text2num / num2text
    test = puzzle["test_pairs"][0]
    lang = puzzle["language"]

    if task_dir == "text2num":
        prompt = (f"What is the numerical value of this expression in {lang}? "
                  f"Give your best guess even if uncertain.\n\n"
                  f"Expression: {test['source']}\n\nAnswer:")
    elif task_dir == "num2text":
        prompt = (f"How do you write the number {test['source']} in {lang}? "
                  f"Give your best guess even if uncertain.\n\nAnswer:")
    else:
        src_parts = test['source'].split(' | ')
        form1 = src_parts[0] if len(src_parts) > 0 else test['source']
        meaning = src_parts[1] if len(src_parts) > 1 else ""
        prompt = (f"In {lang}, the verb '{form1}' means '{meaning}'. "
                  f"What is its other morphological form? Give your best guess.\n\nAnswer:")

    print(f"  [M0] verificación")
    t0       = time.time()
    response = runner.call_model(prompt, model_key, API_KEYS)
    _log_raw(puzzle["id"], model_key, "verification", 1, prompt, response)
    latency  = round(time.time() - t0, 2)
    metrics  = runner.compute_metrics([response.strip()], [test["target"]])
    chrf     = metrics["corpus_chrfpp"]
    known    = chrf > 20
    print(f"    → '{response[:60]}'  chrF={chrf:.1f}  {'⚠ CONOCE' if known else '✓'}")
    return {"strategy": "verification", "metrics": metrics, "known": known}


def _log_raw(puzzle_id, model_key, strategy, step, prompt, response):
    fname = f"{runner.RAW_LOG_DIR}/{puzzle_id}_{model_key}.jsonl"
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "strategy": strategy, "step": step,
        "prompt": prompt, "response": response,
    }
    with open(fname, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─── M1 para tareas no-traducción ─────────────────────────────────────────────

def run_m1_task(puzzle, model_key):
    task_dir  = puzzle.get("translation_direction", "fill2lang")   # key for PROMPT_BUILDERS
    builder   = PROMPT_BUILDERS.get(task_dir, prompt_fill_blanks)
    test      = puzzle["test_pairs"]
    refs      = [p["target"] for p in test]

    prompt   = builder(puzzle, model_key)
    print(f"  [M1] baseline — {task_dir}")
    t0       = time.time()
    response = runner.call_model(prompt, model_key, API_KEYS)
    _log_raw(puzzle["id"], model_key, "baseline", 1, prompt, response)
    latency  = round(time.time() - t0, 2)

    hyps    = runner.extract_translations(response, len(test))
    metrics = runner.compute_metrics(hyps, refs)

    # Exact Match adicional (limpia puntuación y espacios)
    em_count = sum(
        1 for h, r in zip(hyps, refs)
        if h and re.sub(r'\s+', ' ', h.strip().lower()) == re.sub(r'\s+', ' ', r.strip().lower())
    )
    em_pct = round(em_count / len(refs) * 100, 1)

    print(f"    chrF={metrics['corpus_chrfpp']:.1f}  EM={em_pct}%  "
          f"({metrics['num_evaluated']}/{len(refs)} extraídas)")

    for i, (h, r) in enumerate(zip(hyps, refs)):
        match = "✓" if (h and h.strip().lower() == r.strip().lower()) else "✗"
        print(f"    {match} [{i+1}] pred='{h}'  ref='{r}'")

    metrics["exact_match_strict"] = em_pct
    return {
        "strategy": "baseline",
        "metrics": metrics,
        "hypotheses": hyps,
        "references": refs,
        "latency_s": latency,
    }


# ─── Runner principal ──────────────────────────────────────────────────────────

PUZZLES = [
    "puzzles/linguini_012023010100.json",  # Guazacapán Xinka — relanzar (fix extractor 8B)
    "puzzles/linguini_012023050100.json",  # Supyire text_to_num — nuevo
    "puzzles/linguini_012023050200.json",  # Supyire num_to_text — nuevo
]

for puzzle_path in PUZZLES:
    puzzle = json.load(open(puzzle_path))
    pid    = puzzle["id"]
    lang   = puzzle["language"]
    ttype  = puzzle.get("task_type", "?")

    for model_key in MODELS:
        if model_key not in available:
            print(f"  [SKIP] {model_key}")
            continue

        print(f"\n{'='*60}")
        print(f"  {lang} ({ttype}) — {model_key}")
        print(f"{'='*60}")

        results = {}
        try:
            results["verification"] = run_m0_task(puzzle, model_key)
            time.sleep(5)
            results["baseline"]     = run_m1_task(puzzle, model_key)
            time.sleep(8)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results["error"] = str(e)

        # Guardar
        timestamp = datetime.datetime.now().isoformat()
        log = {
            "metadata": {
                "timestamp": timestamp,
                "puzzle_id": pid,
                "language":  lang,
                "task_type": ttype,
                "model_key": model_key,
            },
            "results": results,
        }
        fname = f"{OUTPUT_DIR}/{pid}_{model_key}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Guardado: {fname}")