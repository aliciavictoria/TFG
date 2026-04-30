"""
experiment_runner.py
====================
Motor de experimentos para el TFG: "Estrategias para que LLMs resuelvan
olimpiadas lingüísticas en lenguas no vistas durante el entrenamiento"

Proveedores: Groq + Cerebras (ambos gratuitos, fiables, sin tarjeta)
  - Groq:     console.groq.com  → API Keys → Create
  - Cerebras: cloud.cerebras.ai → API Keys → Create

Estrategias:
  0 - Baseline: todas las frases de entrenamiento de golpe
  1 - Incremental por combinaciones (propuesta del tutor):
      C(n,2) → C(n,3) → ... → C(n,n), acumulando diccionario+reglas
  2 - Step-by-step ordenado (Zhu et al., 2025):
      léxico+orden → fonología → morfo-sintaxis → sintaxis completa
"""

import json, os, sys, time, itertools, datetime
from pathlib import Path
import requests

# ─────────────────────────────────────────────
# MODELOS Y PROVEEDORES
# ─────────────────────────────────────────────

MODELS = {
    #"llama-3.1-8b-groq": {
    #    "provider": "groq",
    #    "model_id": "llama-3.1-8b-instant",
    #    "tier": "weak",
    #    "description": "Llama 3.1 8B via Groq — modelo débil (14.400 req/día gratis)"
    #},
    "llama-3.3-70b-groq": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "tier": "medium",
        "description": "Llama 3.3 70B via Groq — modelo medio (1.000 req/día gratis)"
    },
    #"llama-3.3-70b-cerebras": {
    #    "provider": "cerebras",
    #    "model_id": "llama-3.3-70b",
    #    "tier": "strong",
    #    "description": "Llama 3.3 70B via Cerebras — modelo fuerte (14.400 req/día gratis)"
    #},
}

PROVIDER_URLS = {
    "groq":     "https://api.groq.com/openai/v1/chat/completions",
    "cerebras": "https://api.cerebras.ai/v1/chat/completions",
}

def call_model(prompt: str, model_key: str, api_keys: dict) -> str:
    """Llama al proveedor correcto (Groq o Cerebras, ambos OpenAI-compatible)."""
    model = MODELS[model_key]
    provider = model["provider"]
    key = api_keys.get(provider)
    if not key:
        raise ValueError(f"Falta API key para '{provider}'. Configura {provider.upper()}_API_KEY")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024
    }
    resp = requests.post(PROVIDER_URLS[provider], headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Error de {provider}: {data['error']}")
    return data["choices"][0]["message"]["content"].strip()

# ─────────────────────────────────────────────
# MÉTRICAS BLEU
# ─────────────────────────────────────────────

def compute_bleu(hypotheses: list, references: list) -> dict:
    """Calcula BLEU con sacrebleu."""
    try:
        import sacrebleu as sb
    except ImportError:
        print("  [WARN] sacrebleu no instalado. Ejecuta: pip install sacrebleu")
        return {"corpus_bleu": 0.0, "sentence_bleu": [], "note": "sacrebleu no disponible"}

    if not hypotheses or not references:
        return {"corpus_bleu": 0.0, "sentence_bleu": [], "note": "sin hipótesis"}

    min_len = min(len(hypotheses), len(references))
    hyps, refs = hypotheses[:min_len], references[:min_len]

    corpus_result = sb.corpus_bleu(hyps, [refs])
    sent_scores = [round(sb.sentence_bleu(h, [r]).score, 2) for h, r in zip(hyps, refs)]

    # chrF: más tolerante con morfología compleja (mejor para lenguas aglutinantes)
    chrf_result = sb.corpus_chrf(hyps, [refs], word_order=2)  # chrF++ (word_order=2)
    sent_chrf = [round(sb.sentence_chrf(h, [r], word_order=2).score, 2) for h, r in zip(hyps, refs)]

    return {
        "corpus_bleu": round(corpus_result.score, 2),
        "sentence_bleu": sent_scores,
        "corpus_chrfpp": round(chrf_result.score, 2),
        "sentence_chrfpp": sent_chrf,
        "num_evaluated": min_len,
        "hypotheses": hyps,
        "references": refs
    }

def extract_translations(response: str, num_expected: int) -> list:
    """
    Extrae traducciones de la respuesta del modelo.
    Primero busca la sección TRANSLATIONS: explícita.
    Si no la encuentra, usa líneas numeradas como fallback
    descartando entradas de diccionario.
    """
    import re
    lines = response.strip().split("\n")

    # Intento 1: buscar sección TRANSLATIONS explícita
    in_trans = False
    trans_lines = []
    for line in lines:
        u = line.upper().strip()
        if re.search(r"TRANSLATION[S]?\s*:", u):
            in_trans = True
            continue
        if in_trans and any(kw in u for kw in
                ["DICTIONARY:", "GRAMMAR RULE", "GRAMMAR:", "NOTE:", "EXPLANATION:", "ANALYSIS:"]):
            break
        if in_trans and line.strip():
            trans_lines.append(line.strip())

    if trans_lines:
        cleaned = []
        for line in trans_lines:
            m = re.match(r"^\d+[.)\s]\s*(.+)$", line)
            cleaned.append(m.group(1).strip() if m else line)
        if cleaned:
            # Quitar justificaciones del nuevo formato: "traducción — [notas morfológicas]"
            result = []
            for line in cleaned:
                # Descartar líneas explicativas entre paréntesis
                if line.startswith("(") and line.endswith(")"):
                    continue
                if " — " in line:
                    line = line.split(" — ")[0].strip()
                elif " - " in line and len(line.split(" - ")[0].split()) < 8:
                    line = line.split(" - ")[0].strip()
                # Descartar si después del split sigue siendo una nota
                if line.startswith("("):
                    continue
                # Quitar "source → " si el modelo repite la frase original
                if " → " in line:
                    line = line.split(" → ", 1)[1].strip()
                result.append(line)
            return result[:num_expected]

    # Intento 2 (fallback): líneas numeradas que NO sean diccionario ni notas
    numbered = []
    for line in lines:
        line = line.strip()
        m = re.match(r"^(\d+)[.)]\s+(.+)$", line)
        if m:
            text = m.group(2).strip()
            # Descartar entradas de diccionario
            if re.match(r"^[\w\'\u00c0-\u024f]+\s*[:=]\s*\w", text):
                continue
            # Descartar notas entre paréntesis
            if text.startswith("("):
                continue
            # Si el modelo escribe "source → translation", quedarse con translation
            if " → " in text:
                text = text.split(" → ", 1)[1].strip()
            if " -> " in text:
                text = text.split(" -> ", 1)[1].strip()
            # Quitar justificaciones morfológicas al final
            if " — " in text:
                text = text.split(" — ")[0].strip()
            if text.startswith("(") or not text:
                continue
            numbered.append(text)
    return numbered[:num_expected]

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

METALINGUISTIC_INSTRUCTION = """You are an expert linguistic analyst performing inductive reasoning on an unknown language.
You have never seen this language before. You must reason exclusively from the provided examples using your meta-linguistic knowledge of how human languages work in general (word order typology, morphological patterns, agreement systems, affixation, etc.).

Core principles:
- Do NOT use any prior knowledge of this specific language or its family
- Treat every hypothesis as provisional and subject to revision
- Always show your reasoning explicitly before reaching any conclusion
- A wrong rule confidently stated is worse than uncertainty acknowledged"""

def build_baseline_prompt(train_pairs, test_items):
    examples = "\n".join(f"  {p['source']} → {p['target']}" for p in train_pairs)
    tests = "\n".join(f"  {i+1}. {item['source']}" for i, item in enumerate(test_items))
    return f"""{METALINGUISTIC_INSTRUCTION}

--- TRAINING EXAMPLES ---
{examples}

--- YOUR TASK ---
Based on ALL examples above, write your inferred DICTIONARY and GRAMMAR RULES,
then translate:
{tests}

Format:
DICTIONARY:
[word mappings, one per line]

GRAMMAR RULES:
[rules, one per line]

TRANSLATIONS:
1. [translation]
2. [translation]
"""

def build_incremental_prompt(combo_pairs, current_dict, current_rules, test_items, step, total_steps):
    examples = "\n".join(f"  {p['source']} -> {p['target']}" for p in combo_pairs)
    tests = "\n".join(f"  {i+1}. {item['source']}" for i, item in enumerate(test_items))

    if current_dict or current_rules:
        prev = f"""
--- YOUR CURRENT HYPOTHESIS (treat as provisional — review critically) ---
DICTIONARY:
{current_dict or '(empty)'}

GRAMMAR RULES:
{current_rules or '(empty)'}

IMPORTANT: The new examples below may CONFIRM, REFINE or CONTRADICT entries above.
You MUST delete entries proven wrong and correct entries that were imprecise.
"""
    else:
        prev = "\n(No prior knowledge — this is your first set of examples.)\n"

    return f"""{METALINGUISTIC_INSTRUCTION}

STEP {step}/{total_steps} — New subset of examples to analyze:
{prev}
--- NEW EXAMPLES ---
{examples}

--- YOUR TASK: follow these steps IN ORDER ---

STEP A · ANALYSIS (reason before concluding):
For each NEW EXAMPLE above, identify every morpheme or word you can isolate.
Check explicitly: do any new examples CONTRADICT your current hypothesis?
If yes, state what was wrong and why. Think aloud — show your reasoning.
Do NOT translate the test sentences yet.

STEP B · REVISED DICTIONARY:
Write the updated dictionary. Format: one entry per line: morpheme = meaning (notes)
- ADD new morpheme/word mappings discovered in this step
- CORRECT entries that were imprecise or partially wrong
- DELETE entries proven incorrect — write: [DELETED: old_entry — reason]

STEP C · REVISED GRAMMAR RULES:
Write the updated numbered rule set.
- ADD new rules supported by evidence from this step
- REFINE rules that were too broad or too narrow  
- DELETE rules proven wrong — write: [DELETED: old_rule — reason]

STEP D · TRANSLATIONS:
Now translate ONLY these sentences into English (these are different from the training examples above):
{tests}
Write only the English translation on each line, nothing else before the dash.

Format your entire response EXACTLY as:
ANALYSIS:
[your reasoning about the new examples]

DICTIONARY:
[entries, one per line]

GRAMMAR RULES:
[rules, one per line]

TRANSLATIONS:
1. [English translation only] — [brief morpheme breakdown]
2. [English translation only] — [brief morpheme breakdown]
"""


def build_stepbystep_prompt(stage, stage_pairs, current_dict, current_rules, test_items, step, total_steps):
    stage_focus = {
        "lexical":     "Focus on vocabulary and basic word order. Simple SVO/SOV sentences, no morphological variation.",
        "phonology":   "Focus on phonological patterns: vowel changes, consonant alternations, affixes, allomorphs.",
        "morphosyntax":"Focus on agreement, tense, number, person, gender markings.",
        "syntax":      "Combine all rules: negation, questions, complex clauses."
    }
    examples = "\n".join(f"  {p['source']} → {p['target']}" for p in stage_pairs)
    tests = "\n".join(f"  {i+1}. {item['source']}" for i, item in enumerate(test_items))
    prev = ""
    if current_dict or current_rules:
        prev = f"""
--- ACCUMULATED KNOWLEDGE ---
DICTIONARY:
{current_dict or '(empty)'}

GRAMMAR RULES:
{current_rules or '(empty)'}
"""
    return f"""{METALINGUISTIC_INSTRUCTION}

STEP {step}/{total_steps} — Stage: {stage.upper()}
{stage_focus.get(stage, '')}
{prev}
--- EXAMPLES FOR THIS STAGE ---
{examples}

--- YOUR TASK ---
1. Refine DICTIONARY with new vocabulary
2. Add/update GRAMMAR RULES for stage: {stage}
3. Translate using ALL accumulated knowledge

Format:
DICTIONARY:
[word mappings]

GRAMMAR RULES:
[rules]

TRANSLATIONS:
1. [translation]
2. [translation]
"""

# ─────────────────────────────────────────────
# EXTRACCIÓN DE DICT Y REGLAS
# ─────────────────────────────────────────────

def extract_dict_and_rules(response: str):
    lines = response.split("\n")
    section = None
    dict_l, rule_l = [], []
    for line in lines:
        u = line.upper().strip()
        if "DICTIONARY" in u and ":" in u: section = "dict"; continue
        elif "GRAMMAR" in u and "RULE" in u: section = "rules"; continue
        elif "TRANSLATION" in u: section = "trans"; continue
        if section == "dict" and line.strip(): dict_l.append(line.strip())
        elif section == "rules" and line.strip(): rule_l.append(line.strip())
    return "\n".join(dict_l), "\n".join(rule_l)

# ─────────────────────────────────────────────
# ESTRATEGIAS
# ─────────────────────────────────────────────

def run_strategy_0(puzzle, model_key, api_keys):
    print(f"  [S0-Baseline] {len(puzzle['train_pairs'])} frases de golpe...")
    test_items = puzzle["test_pairs"]
    prompt = build_baseline_prompt(puzzle["train_pairs"], test_items)
    t0 = time.time()
    response = call_model(prompt, model_key, api_keys)
    latency = round(time.time() - t0, 2)
    refs = [p["target"] for p in test_items]
    hyps = extract_translations(response, len(test_items))
    bleu = compute_bleu(hyps, refs)
    print(f"    BLEU: {bleu['corpus_bleu']:.2f} | chrF: {bleu.get('corpus_chrfpp',0):.2f} | latencia: {latency}s")
    return {
        "strategy": "baseline", "strategy_id": 0,
        "steps": [{"step": 1, "num_train_pairs": len(puzzle["train_pairs"]),
                   "prompt": prompt, "response": response,
                   "latency_s": latency, "bleu": bleu}],
        "final_bleu": bleu["corpus_bleu"], "best_bleu": bleu["corpus_bleu"]
    }

def run_strategy_1(puzzle, model_key, api_keys, combo_sizes=None):
    train = puzzle["train_pairs"]
    n = len(train)
    test_items = puzzle["test_pairs"]
    refs = [p["target"] for p in test_items]
    if combo_sizes is None:
        combo_sizes = list(range(2, n + 1))

    steps, current_dict, current_rules, best_bleu = [], "", "", 0.0

    for idx, k in enumerate(combo_sizes):
        combo = list(itertools.combinations(range(n), k))[0]
        combo_pairs = [train[i] for i in combo]
        print(f"  [S1-Incremental] Paso {idx+1}/{len(combo_sizes)}: {k} frases (C({n},{k}))")
        prompt = build_incremental_prompt(combo_pairs, current_dict, current_rules,
                                          test_items, idx+1, len(combo_sizes))
        t0 = time.time()
        response = call_model(prompt, model_key, api_keys)
        latency = round(time.time() - t0, 2)
        nd, nr = extract_dict_and_rules(response)
        if nd: current_dict = nd
        if nr: current_rules = nr
        hyps = extract_translations(response, len(test_items))
        bleu = compute_bleu(hyps, refs)
        if bleu["corpus_bleu"] > best_bleu: best_bleu = bleu["corpus_bleu"]
        print(f"    BLEU: {bleu['corpus_bleu']:.2f} | chrF: {bleu.get('corpus_chrfpp',0):.2f} | latencia: {latency}s")
        steps.append({"step": idx+1, "combo_size": k, "combo_indices": list(combo),
                      "prompt": prompt, "response": response, "latency_s": latency,
                      "accumulated_dict": current_dict, "accumulated_rules": current_rules,
                      "bleu": bleu})
        time.sleep(10)  # respetar rate limit de Groq (6000 tokens/min)

    return {"strategy": "incremental_combinations", "strategy_id": 1,
            "combo_sizes": combo_sizes, "steps": steps,
            "final_bleu": steps[-1]["bleu"]["corpus_bleu"], "best_bleu": best_bleu}

def run_strategy_2(puzzle, model_key, api_keys):
    train = puzzle["train_pairs"]
    n = len(train)
    test_items = puzzle["test_pairs"]
    refs = [p["target"] for p in test_items]

    # Partición heurística por tipo lingüístico
    lex_end  = max(1, int(n * 0.35))
    phon_end = max(lex_end + 1, int(n * 0.50))
    morp_end = max(phon_end + 1, int(n * 0.80))
    stages_data = {
        "lexical":      train[:lex_end],
        "phonology":    train[lex_end:phon_end] or train[:1],
        "morphosyntax": train[phon_end:morp_end] or train[:1],
        "syntax":       train[morp_end:] or train[-1:]
    }
    stage_order = ["lexical", "phonology", "morphosyntax", "syntax"]
    steps, current_dict, current_rules, best_bleu = [], "", "", 0.0

    for idx, stage in enumerate(stage_order):
        stage_pairs = stages_data[stage]
        print(f"  [S2-StepByStep] Etapa {idx+1}/4: {stage} ({len(stage_pairs)} frases)")
        prompt = build_stepbystep_prompt(stage, stage_pairs, current_dict, current_rules,
                                         test_items, idx+1, 4)
        t0 = time.time()
        response = call_model(prompt, model_key, api_keys)
        latency = round(time.time() - t0, 2)
        nd, nr = extract_dict_and_rules(response)
        if nd: current_dict = nd
        if nr: current_rules = nr
        hyps = extract_translations(response, len(test_items))
        bleu = compute_bleu(hyps, refs)
        if bleu["corpus_bleu"] > best_bleu: best_bleu = bleu["corpus_bleu"]
        print(f"    BLEU: {bleu['corpus_bleu']:.2f} | chrF: {bleu.get('corpus_chrfpp',0):.2f} | latencia: {latency}s")
        steps.append({"step": idx+1, "stage": stage, "num_stage_pairs": len(stage_pairs),
                      "prompt": prompt, "response": response, "latency_s": latency,
                      "accumulated_dict": current_dict, "accumulated_rules": current_rules,
                      "bleu": bleu})
        time.sleep(10)

    return {"strategy": "step_by_step_linguistic", "strategy_id": 2,
            "stage_order": stage_order, "steps": steps,
            "final_bleu": steps[-1]["bleu"]["corpus_bleu"], "best_bleu": best_bleu}

# ─────────────────────────────────────────────
# RUNNER PRINCIPAL
# ─────────────────────────────────────────────

def run_experiment(puzzle_path, model_key, api_keys, strategies=[0, 1, 2], output_dir="results"):
    with open(puzzle_path) as f:
        puzzle = json.load(f)

    model_info = MODELS[model_key]
    timestamp = datetime.datetime.now().isoformat()

    print(f"\n{'='*60}")
    print(f"  Idioma:  {puzzle['language']}")
    print(f"  Modelo:  {model_key}  ({model_info['tier']})")
    print(f"  Tiempo:  {timestamp[:19]}")
    print(f"{'='*60}")

    log = {
        "metadata": {
            "timestamp": timestamp,
            "puzzle_id": puzzle["id"],
            "language": puzzle["language"],
            "language_family": puzzle.get("language_family", ""),
            "source": puzzle.get("source", ""),
            "model_key": model_key,
            "model_info": model_info,
            "num_train_pairs": len(puzzle["train_pairs"]),
            "num_test_pairs": len(puzzle["test_pairs"]),
            "strategies_run": strategies
        },
        "puzzle": puzzle,
        "results": {}
    }

    runners = {
        0: lambda: run_strategy_0(puzzle, model_key, api_keys),
        1: lambda: run_strategy_1(puzzle, model_key, api_keys),
        2: lambda: run_strategy_2(puzzle, model_key, api_keys),
    }
    names = {0: "baseline", 1: "incremental_combinations", 2: "step_by_step"}

    for s_id in strategies:
        print(f"\n→ Estrategia {s_id}: {names[s_id]}")
        try:
            result = runners[s_id]()
            log["results"][names[s_id]] = result
        except Exception as e:
            print(f"  ✗ Error: {e}")
            log["results"][names[s_id]] = {"error": str(e)}

    # Resumen
    summary = {}
    for name, res in log["results"].items():
        if "error" not in res:
            # Calcular mejor chrF entre todos los pasos
            best_chrf = max(
                (s.get("bleu", {}).get("corpus_chrfpp",
                 s.get("bleu", {}).get("corpus_chrf", 0)) for s in res.get("steps", [])),
                default=0
            )
            summary[name] = {"final_bleu": res.get("final_bleu", 0),
                             "best_bleu": res.get("best_bleu", 0),
                             "best_chrf": best_chrf,
                             "num_steps": len(res.get("steps", []))}
    log["summary"] = summary

    Path(output_dir).mkdir(exist_ok=True)
    safe = model_key.replace("/", "-")
    fname = f"{output_dir}/{puzzle['id']}_{safe}_{timestamp[:10]}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Log guardado: {fname}")
    print("\nRESUMEN FINAL:")
    for name, s in summary.items():
        print(f"  {name:35s} BLEU: {s['final_bleu']:5.2f}  best: {s['best_bleu']:5.2f}  chrF: {s.get('best_chrf',0):5.2f}")

    return log

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    API_KEYS = {
        "groq":     os.getenv("GROQ_API_KEY", ""),
        "cerebras": os.getenv("CEREBRAS_API_KEY", ""),
    }

    # Filtrar solo los modelos para los que hay key disponible
    available = {k: v for k, v in MODELS.items() if API_KEYS.get(v["provider"])}
    if not available:
        print("\nERROR: No se encontró ninguna API key.")
        print("Necesitas al menos una de estas (ambas gratuitas):")
        print()
        print("  GROQ (14.400 req/día):     console.groq.com → API Keys → Create")
        print("    export GROQ_API_KEY='gsk_...'")
        print()
        print("  CEREBRAS (14.400 req/día): cloud.cerebras.ai → API Keys → Create")
        print("    export CEREBRAS_API_KEY='csk_...'")
        sys.exit(1)

    print(f"\nModelos disponibles con las keys configuradas:")
    for k, v in available.items():
        print(f"  ✓ {k:35s} ({v['tier']})")

    # ── Puzzles disponibles ──
    # Cambia esta lista para seleccionar qué puzzles ejecutar
    PUZZLES = [
        #"puzzles/ayutla_mixe.json",                        # piloto (8 train)
        #"puzzles/linguini_012023020100.json",               # Apurinã — 16 train
        "puzzles/linguini_012018020100.json",               # Hakhun — 10 train
        #"puzzles/linguini_012008040100.json",               # Copainalá Zoque — 20 train
        #"puzzles/linguini_012013010200.json",               # Yidiny — 23 train
    ]

    for puzzle_path in PUZZLES:
        for model_key in available:
            run_experiment(
                puzzle_path=puzzle_path,
                model_key=model_key,
                api_keys=API_KEYS,
                strategies=[0, 1, 2],
                output_dir="results"
            )