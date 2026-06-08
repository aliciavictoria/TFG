"""
experiment_runner.py  v4
========================
TFG: "Estrategias de prompting para mejorar las capacidades metalingüísticas
de LLMs en la resolución de problemas de olimpiadas lingüísticas"

Metodologías:
  M0  Verificación      ¿conoce el modelo el idioma? (control)
  M1  Baseline          todas las frases de entrenamiento de golpe
  M2  Incremental       subconjuntos crecientes con diccionario acumulado
  M3  Step-by-step      léxico → fonología → morfosintaxis → sintaxis
  M4  Inspiración human razonamiento humano de OTRO puzzle como guía
  M5  Segundo revisor   M1 + autocorrección de contradicciones internas

Métricas: BLEU · chrF++ · TER · Exact Match
  BLEU    ↑ mejor  (0–100)   precisión de n-gramas
  chrF++  ↑ mejor  (0–100)   carácter + palabra F-score (métrica principal)
  TER     ↓ mejor  (0–∞)     tasa de edición por tokens
  EM      ↑ mejor  (0–100%)  exact match ignorando mayúsculas/puntuación

Nota sobre M2: se reporta el mejor paso según chrF++ en la tabla comparativa.
La evolución completa queda en el JSON para análisis de estabilidad.

Nota sobre M4: NO ejecutar en el mismo puzzle cuyo razonamiento se usa como
inspiración (evitar data leakage). Pasar human_resolution="" para ese puzzle.
"""

import json, os, sys, time, itertools, datetime, re
from pathlib import Path
import requests

LOG_FULL_RESPONSES = False

MODELS = {
    "llama-3.1-8b-groq": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "tier": "weak",
    },
    "llama-3.3-70b-groq": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "tier": "medium",
    },
    "gpt-oss-120b-openrouter": {
        "provider": "openrouter",
        "model_id": "openai/gpt-oss-120b:free",
        "tier": "strong",
    },
    "gpt-oss-120b-cerebras": {
        "provider": "cerebras",
        "model_id": "gpt-oss-120b",
        "tier": "strong",
    },
    "llama-3.3-70b-sambanova": {
        "provider": "sambanova",
        "model_id": "Meta-Llama-3.3-70B-Instruct",
        "tier": "medium",
    },
}

PROVIDER_URLS = {
    "groq":       "https://api.groq.com/openai/v1/chat/completions",
    "cerebras":   "https://api.cerebras.ai/v1/chat/completions",
    "sambanova":  "https://api.sambanova.ai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


def call_model(prompt: str, model_key: str, api_keys: dict,
               timeout: int = 150) -> str:
    model    = MODELS[model_key]
    provider = model["provider"]
    key      = api_keys.get(provider)
    if not key:
        raise ValueError(f"Falta API key para '{provider}'.")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://tfg-linguistic-olympiad"
        headers["X-Title"]      = "TFG Olimpiadas Linguisticas"

    # Cerebras usa "max_completion_tokens" en lugar de "max_tokens"
    tokens_key = "max_completion_tokens" if provider == "cerebras" else "max_tokens"
    payload = {
        "model":       model["model_id"],
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        tokens_key:    4096,
    }
    for attempt in range(3):
        resp = requests.post(PROVIDER_URLS[provider], headers=headers,
                             json=payload, timeout=timeout)
        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            print(f"    [429] Rate limit, esperando {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Error de {provider}: {data['error']}")
        msg  = data["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning") or ""
        return text.strip()
    raise RuntimeError("Rate limit persistente tras 3 intentos")


# ─── Extractor de traducciones ────────────────────────────────────────────────

_ARROW_RE = re.compile(
    r'[\u2192\->]+\s*\*{0,2}'
    r'(?:[\u201c\u2018"\']{1})?'
    r'([\w\u00c0-\u024f\u0250-\u02ff\u0300-\u036f\u1e00-\u1eff!|]'
    r'[^\u201d\u2019"\'\n\*]{2,110})'
    r'(?:[\u201d\u2019"\']{1})?\*{0,2}'
)
_NUMBERED_RE = re.compile(
    r'^\s*\d+[.)]\s+'
    r'([\w\u00c0-\u024f\u0250-\u02ff\u0300-\u036f\u1e00-\u1eff!|]'
    r'[^\n]{2,110})\s*$',
    re.MULTILINE | re.UNICODE,
)
_SKIP_RE       = re.compile(r'^(\*\*|STEP|DICTIONARY|GRAMMAR|ANALYSIS|NOTE|#|\[|Add |Update |Delete |Correct |Check |Remove )', re.IGNORECASE)
_DICT_ENTRY_RE = re.compile(
    r'^[\w\u00c0-\u024f\u0250-\u02ff\u1e00-\u1eff]'
    r'[\w\u00c0-\u024f\u0250-\u02ff\u1e00-\u1eff\-]*'
    r'(?:\s+[\w\u00c0-\u024f\u0250-\u02ff\u1e00-\u1eff]'
    r'[\w\u00c0-\u024f\u0250-\u02ff\u1e00-\u1eff\-]*)*'
    r'\s*[=:]',
    re.IGNORECASE | re.UNICODE
)
_INSTRUCTION_RE = re.compile(
    r'\b(update|add|delete|remove|correct|change|revise|fix|insert)\b.{0,60}\b(entry|rule|dictionary|suffix|morpheme|word)\b',
    re.IGNORECASE,
)
_META_LINGUISTIC_RE = re.compile(
    r'\b(word order|SOV|SVO|VSO|OVS|morpheme|phonolog|syntax|grammar rule|question particle'
    r'|subject marker|object marker|aspect marker|tense marker|allomorph|affix|suffix|prefix'
    r'|conjugat|declension|inflect)\b',
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = text.strip().strip('*').strip('"').strip('\u201c').strip('\u201d').strip()
    text = re.sub(r'\s*[\(\[].{0,100}[\)\]]\s*$', '', text).strip()
    return text


def _is_instruction(text: str) -> bool:
    """Detecta si una línea es una instrucción de revisión, no una traducción."""
    return bool(_INSTRUCTION_RE.search(text))


def _is_meta_analysis(text: str) -> bool:
    """Detecta si la línea es un análisis lingüístico meta (SOV, word order...) en vez de traducción."""
    return bool(_META_LINGUISTIC_RE.search(text))


def extract_translations(response: str, num_expected: int) -> list:
    """
    Extrae traducciones de la respuesta libre del modelo.
    Busca primero la sección TRANSLATIONS, luego aplica patrones de flecha
    y líneas numeradas. Filtra activamente instrucciones de revisión.
    """
    header = re.search(r'TRANSLATIONS\b[^\n]*\n', response, re.IGNORECASE)
    section = response[header.end():] if header else response

    stop = re.search(
        r'\n\s*(?:ANALYSIS|DICTIONARY|GRAMMAR\s+RULE|NOTE|EXPLANATION|CRITIQUE)\s*[:\*\n]',
        section, re.IGNORECASE,
    )
    if stop:
        section = section[:stop.start()]

    arrow_hits = [_clean(m.group(1)) for m in _ARROW_RE.finditer(section)]
    arrow_hits = [t for t in arrow_hits
                  if 4 < len(t) < 120 and not _is_instruction(t) and not _is_meta_analysis(t)]
    if len(arrow_hits) >= num_expected:
        return arrow_hits[:num_expected]

    num_hits = []
    for m in _NUMBERED_RE.finditer(section):
        t = _clean(m.group(1))
        if not t:
            continue
        if _SKIP_RE.match(t) or _DICT_ENTRY_RE.match(t) or _is_instruction(t) or _is_meta_analysis(t):
            continue
        if 4 < len(t) < 120:
            num_hits.append(t)

    combined = arrow_hits + [h for h in num_hits if h not in arrow_hits]
    return combined[:num_expected]


def extract_dict_and_rules(response: str):
    section, dict_l, rule_l = None, [], []
    for line in response.split("\n"):
        u = line.upper().strip()
        if "DICTIONARY" in u and ":" in u:
            section = "dict"; continue
        if "GRAMMAR" in u and "RULE" in u:
            section = "rules"; continue
        if "TRANSLATION" in u:
            section = "trans"; continue
        if section == "dict" and line.strip():
            dict_l.append(line.strip())
        elif section == "rules" and line.strip():
            rule_l.append(line.strip())
    return "\n".join(dict_l), "\n".join(rule_l)


# ─── Métricas ─────────────────────────────────────────────────────────────────

def exact_match(hypothesis: str, reference: str) -> bool:
    norm = lambda s: re.sub(r'[^\w\s]', '', s.lower()).strip()
    return norm(hypothesis) == norm(reference)


def compute_metrics(hypotheses: list, references: list) -> dict:
    try:
        import sacrebleu as sb
    except ImportError:
        return {"error": "sacrebleu no disponible"}

    if not hypotheses or not references:
        return {
            "corpus_bleu": 0.0, "corpus_chrfpp": 0.0,
            "corpus_ter": 100.0, "exact_match_corpus": 0.0,
            "sentence_bleu": [], "sentence_chrfpp": [],
            "sentence_ter": [], "sentence_em": [],
            "hypotheses": [], "references": [], "num_evaluated": 0,
        }

    # Normalizar referencias: si algún elemento es lista (múltiples respuestas válidas),
    # tomar siempre la primera alternativa como referencia gold
    references = [r[0] if isinstance(r, list) else str(r) for r in references]

    n    = min(len(hypotheses), len(references))
    hyps = hypotheses[:n]
    refs = references[:n]

    c_bleu = sb.corpus_bleu(hyps, [refs])
    c_chrf = sb.corpus_chrf(hyps, [refs], word_order=2)
    c_ter  = sb.corpus_ter(hyps, [refs])
    s_em   = [exact_match(h, r) for h, r in zip(hyps, refs)]

    return {
        "corpus_bleu":        round(c_bleu.score, 2),
        "corpus_chrfpp":      round(c_chrf.score, 2),
        "corpus_ter":         round(c_ter.score, 2),
        "exact_match_corpus": round(sum(s_em) / n * 100, 2),
        "sentence_bleu":      [round(sb.sentence_bleu(h,[r]).score,2) for h,r in zip(hyps,refs)],
        "sentence_chrfpp":    [round(sb.sentence_chrf(h,[r],word_order=2).score,2) for h,r in zip(hyps,refs)],
        "sentence_ter":       [round(sb.sentence_ter(h,[r]).score,2) for h,r in zip(hyps,refs)],
        "sentence_em":        s_em,
        "hypotheses":         hyps,
        "references":         refs,
        "num_evaluated":      n,
    }


# ─── Utilidades ───────────────────────────────────────────────────────────────

def _step_record(base: dict, prompt: str, response: str) -> dict:
    if LOG_FULL_RESPONSES:
        base["prompt"]   = prompt
        base["response"] = response
    return base


def _best_step(steps: list) -> dict:
    return max(steps, key=lambda s: s["metrics"].get("corpus_chrfpp", 0), default={})


def _truncate_context(text: str, max_chars: int = 3000) -> str:
    """Trunca el diccionario o reglas acumuladas para evitar error 413."""
    if not text or len(text) <= max_chars:
        return text
    lines = text.split("\n")
    result, total = [], 0
    for line in lines:
        if total + len(line) > max_chars:
            result.append(f"... [{len(lines) - len(result)} entradas adicionales omitidas por longitud]")
            break
        result.append(line)
        total += len(line) + 1
    return "\n".join(result)


def _print_step(metrics: dict):
    print(f"    BLEU={metrics['corpus_bleu']:5.2f} "
          f"chrF={metrics['corpus_chrfpp']:5.2f} "
          f"TER={metrics['corpus_ter']:7.2f} "
          f"EM={metrics['exact_match_corpus']:5.1f}%")


METALINGUISTIC_INSTRUCTION = """You are an expert linguistic analyst performing inductive reasoning on an unknown language.
You have never seen this language before. You must reason exclusively from the provided examples using your meta-linguistic knowledge of how human languages work in general (word order typology, morphological patterns, agreement systems, affixation, etc.).

Core principles:
- Do NOT use any prior knowledge of this specific language or its family
- Treat every hypothesis as provisional and subject to revision
- Always show your reasoning explicitly before reaching any conclusion
- A wrong rule confidently stated is worse than uncertainty acknowledged"""


def translation_instruction(direction: str, lang_name: str) -> str:
    if direction == "en2lang":
        return f"translate into {lang_name} (the unknown language)"
    return "translate into English"


# ─── M0 — Verificación ────────────────────────────────────────────────────────

def run_m0(puzzle: dict, model_key: str, api_keys: dict) -> dict:
    direction = puzzle.get("translation_direction", "lang2en")
    sentence  = puzzle["test_pairs"][0]["source"]
    reference = puzzle["test_pairs"][0]["target"]

    if direction == "en2lang":
        prompt = (f"Translate the following sentence into {puzzle['language']}. "
                  f"Give your best attempt even if you are not sure.\n\n"
                  f"Sentence: {sentence}\n\nTranslation:")
    else:
        prompt = (f"Translate the following sentence into English. "
                  f"Give your best attempt even if you are not sure.\n\n"
                  f"Sentence: {sentence}\n\nTranslation:")

    print(f"  [M0] '{sentence}'")
    t0       = time.time()
    response = call_model(prompt, model_key, api_keys)
    latency  = round(time.time() - t0, 2)
    metrics  = compute_metrics([response.strip()], [reference])

    print(f"    → '{response[:80]}'")
    _print_step(metrics)

    known = metrics["corpus_chrfpp"] > 20
    if known:
        print("    ⚠ chrF > 20 — el modelo puede conocer el idioma. Revisar.")

    rec = {
        "strategy":        "verification",
        "strategy_id":     0,
        "sentence_tested": sentence,
        "reference":       reference,
        "hypothesis":      response.strip(),
        "latency_s":       latency,
        "metrics":         metrics,
        "known_language":  known,
    }
    if LOG_FULL_RESPONSES:
        rec["prompt"] = prompt
    return rec


# ─── M1 — Baseline ────────────────────────────────────────────────────────────

def run_m1(puzzle: dict, model_key: str, api_keys: dict) -> dict:
    direction   = puzzle.get("translation_direction", "lang2en")
    train       = puzzle["train_pairs"]
    test_items  = puzzle["test_pairs"]
    refs        = [p["target"] for p in test_items]
    trans_instr = translation_instruction(direction, puzzle["language"])

    examples = "\n".join(f"  {p['source']} → {p['target']}" for p in train)
    tests    = "\n".join(f"  {i+1}. {item['source']}" for i, item in enumerate(test_items))

    prompt = f"""{METALINGUISTIC_INSTRUCTION}

--- TRAINING EXAMPLES ---
{examples}

--- YOUR TASK ---
Based on ALL examples above, write your inferred DICTIONARY and GRAMMAR RULES,
then {trans_instr}:
{tests}

Format your response EXACTLY as:
DICTIONARY:
[word mappings, one per line]

GRAMMAR RULES:
[rules, one per line]

TRANSLATIONS:
1. [translation only]
2. [translation only]
"""
    print(f"  [M1] {len(train)} frases de golpe")
    t0       = time.time()
    response = call_model(prompt, model_key, api_keys)
    latency  = round(time.time() - t0, 2)

    hyps    = extract_translations(response, len(test_items))
    metrics = compute_metrics(hyps, refs)
    _print_step(metrics)

    step = _step_record(
        {"step": 1, "num_train_pairs": len(train), "latency_s": latency, "metrics": metrics},
        prompt, response,
    )
    return {
        "strategy":     "baseline",
        "strategy_id":  1,
        "steps":        [step],
        "best_metrics": metrics,
    }


# ─── M2 — Incremental ─────────────────────────────────────────────────────────

def run_m2(puzzle: dict, model_key: str, api_keys: dict) -> dict:
    direction   = puzzle.get("translation_direction", "lang2en")
    train       = puzzle["train_pairs"]
    n           = len(train)
    test_items  = puzzle["test_pairs"]
    refs        = [p["target"] for p in test_items]
    trans_instr = translation_instruction(direction, puzzle["language"])

    combo_sizes   = list(range(2, min(n - 1, 15) + 2))
    steps         = []
    current_dict  = ""
    current_rules = ""

    for idx, k in enumerate(combo_sizes):
        combo       = list(itertools.combinations(range(n), k))[0]
        combo_pairs = [train[i] for i in combo]
        total       = len(combo_sizes)

        _d = _truncate_context(current_dict)  if current_dict  else '(empty)'
        _r = _truncate_context(current_rules) if current_rules else '(empty)'
        prev = f"""
--- YOUR CURRENT HYPOTHESIS (provisional — review critically) ---
DICTIONARY:
{_d}

GRAMMAR RULES:
{_r}

IMPORTANT: New examples may CONFIRM, REFINE or CONTRADICT entries above.
Delete proven-wrong entries; correct imprecise ones.
""" if (current_dict or current_rules) else "\n(No prior knowledge — first set of examples.)\n"

        examples = "\n".join(f"  {p['source']} -> {p['target']}" for p in combo_pairs)
        tests    = "\n".join(f"  {i+1}. {item['source']}" for i, item in enumerate(test_items))

        prompt = f"""{METALINGUISTIC_INSTRUCTION}

STEP {idx+1}/{total} — New subset of examples:
{prev}
--- NEW EXAMPLES ---
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

TRANSLATIONS:
1. [translation only]
2. [translation only]
"""
        print(f"  [M2] Paso {idx+1}/{total}: {k} frases")
        t0       = time.time()
        response = call_model(prompt, model_key, api_keys)
        latency  = round(time.time() - t0, 2)

        nd, nr = extract_dict_and_rules(response)
        if nd: current_dict  = nd
        if nr: current_rules = nr

        hyps    = extract_translations(response, len(test_items))
        metrics = compute_metrics(hyps, refs)
        _print_step(metrics)

        step = _step_record({
            "step":            idx + 1,
            "combo_size":      k,
            "combo_indices":   list(combo),
            "latency_s":       latency,
            "accumulated_dict":  current_dict,
            "accumulated_rules": current_rules,
            "metrics":         metrics,
        }, prompt, response)
        steps.append(step)
        time.sleep(10)

    best = _best_step(steps)
    return {
        "strategy":     "incremental_combinations",
        "strategy_id":  2,
        "combo_sizes":  combo_sizes,
        "steps":        steps,
        "best_metrics": best.get("metrics", {}),
        "best_step":    best.get("step"),
    }


# ─── M3 — Step-by-step ────────────────────────────────────────────────────────

def run_m3(puzzle: dict, model_key: str, api_keys: dict) -> dict:
    direction   = puzzle.get("translation_direction", "lang2en")
    train       = puzzle["train_pairs"]
    n           = len(train)
    test_items  = puzzle["test_pairs"]
    refs        = [p["target"] for p in test_items]
    trans_instr = translation_instruction(direction, puzzle["language"])

    lex_end  = max(1, int(n * 0.35))
    phon_end = max(lex_end + 1, int(n * 0.50))
    morp_end = max(phon_end + 1, int(n * 0.80))
    stages = {
        "lexical":      train[:lex_end],
        "phonology":    train[lex_end:phon_end] or train[:1],
        "morphosyntax": train[phon_end:morp_end] or train[:1],
        "syntax":       train[morp_end:] or train[-1:],
    }
    stage_focus = {
        "lexical":      "Focus on vocabulary and basic word order.",
        "phonology":    "Focus on phonological patterns: vowel changes, consonant alternations, affixes, allomorphs.",
        "morphosyntax": "Focus on agreement, tense, number, person, gender markings.",
        "syntax":       "Combine all rules: negation, questions, complex clauses.",
    }
    stage_order   = ["lexical", "phonology", "morphosyntax", "syntax"]
    steps         = []
    current_dict  = ""
    current_rules = ""

    for idx, stage in enumerate(stage_order):
        stage_pairs = stages[stage]
        examples    = "\n".join(f"  {p['source']} → {p['target']}" for p in stage_pairs)
        tests       = "\n".join(f"  {i+1}. {item['source']}" for i, item in enumerate(test_items))
        _d = _truncate_context(current_dict)  if current_dict  else '(empty)'
        _r = _truncate_context(current_rules) if current_rules else '(empty)'
        prev = f"""
--- ACCUMULATED KNOWLEDGE ---
DICTIONARY:
{_d}

GRAMMAR RULES:
{_r}
""" if (current_dict or current_rules) else ""

        prompt = f"""{METALINGUISTIC_INSTRUCTION}

STEP {idx+1}/4 — Stage: {stage.upper()}
{stage_focus[stage]}
{prev}
--- EXAMPLES FOR THIS STAGE ---
{examples}

--- YOUR TASK ---
1. Refine DICTIONARY with new vocabulary
2. Add/update GRAMMAR RULES for this stage
3. {trans_instr.capitalize()} using ALL accumulated knowledge

Format your response EXACTLY as:
DICTIONARY:
[word mappings]

GRAMMAR RULES:
[rules]

TRANSLATIONS:
1. [translation only]
2. [translation only]
"""
        print(f"  [M3] Etapa {idx+1}/4: {stage} ({len(stage_pairs)} frases)")
        t0       = time.time()
        response = call_model(prompt, model_key, api_keys)
        latency  = round(time.time() - t0, 2)

        nd, nr = extract_dict_and_rules(response)
        if nd: current_dict  = nd
        if nr: current_rules = nr

        hyps    = extract_translations(response, len(test_items))
        metrics = compute_metrics(hyps, refs)
        _print_step(metrics)

        step = _step_record({
            "step":            idx + 1,
            "stage":           stage,
            "num_stage_pairs": len(stage_pairs),
            "latency_s":       latency,
            "accumulated_dict":  current_dict,
            "accumulated_rules": current_rules,
            "metrics":         metrics,
        }, prompt, response)
        steps.append(step)
        time.sleep(10)

    best = _best_step(steps)
    return {
        "strategy":     "step_by_step",
        "strategy_id":  3,
        "stage_order":  stage_order,
        "steps":        steps,
        "best_metrics": best.get("metrics", {}),
        "best_stage":   best.get("stage"),
    }


# ─── M4 — Inspiración humana ──────────────────────────────────────────────────

HUMAN_RESOLUTION_HAKHUN = """The following is a real human's reasoning while translating an unknown language puzzle.
Use it as inspiration for your OWN reasoning process on a DIFFERENT language below.
Do NOT copy vocabulary or rules — they belong to a completely different language.

--- HUMAN REASONING EXAMPLE (Hakhun language, Sino-Tibetan) ---

Sentence 1: "nɤ ʒip ku ne"
- 'ne' appears at the end of every sentence → likely marks interrogative mood (questions)
- 'ŋa' consistently refers to 1st person singular (I/me); 'nɤ' to 2nd person singular (you)
- So the structure so far: "Did/Do you(sg) … ?"
- 'ʒip' appears only once → likely the verb 'sleep'
- 'ku' vs 'tuʔ' in sentence 2: 'tuʔ' seems to mark past tense (did), so 'ku' marks present (do)
- CONCLUSION: Do you(sg) sleep?

Sentence 2: "ati kəmə nirum lapkʰi tʰi ne"
- 'ati' appears as subject/object in several sentences → he/him depending on role
- 'kəmə' appears in many sentences without a clear lexical meaning → object marker or aspect particle
- 'nirum' = we/us (1st person plural)
- 'lapkʰi' = see (the verb, appears consistently with visual perception)
- 'tʰi' → past tense marker (Did)
- CONCLUSION: Did he see us?

Sentence 3: "tarum kəmə nuʔrum cʰam ran ne"
- 'tarum' = they (3rd person plural)
- 'nuʔrum' = you (2nd person plural) — note ʔ is not past tense here, it's part of the pronoun
- 'cʰam' = know (verb)
- 'ran' ≠ 'kan'; 'kan' appears in sentence 7 (present). 'ran' must be a different present marker
- CONCLUSION: Do they know you(pl)?

Sentence 4: "nirum kəmə tarum lan ki ne"
- nirum = we, tarum = them, lan = beat (verb, appears only once)
- 'ki' = present tense marker
- CONCLUSION: Do we beat them?

Sentence 5: "nirum kəmə nɤ cʰam tiʔ ne"
- nirum = we, nɤ = you(sg), cʰam = know
- 'tiʔ' = past tense (confirmed: contains the ʔ sound associated with past)
- CONCLUSION: Did we know you(sg)?

Sentence 6: "nirum ka tiʔ ne"
- nirum = we, tiʔ = past tense, no object marker kəmə (intransitive verb)
- 'ka' = go (verb, appears in sentence 1 as 'kɤ' — allomorph)
- CONCLUSION: Did we go?

Key patterns identified:
- Tense suffixes: tuʔ/tɤʔ/tʰɤ/tʰu/tʰi/tiʔ = past (Did); ku/ki/rɤ/ri/ran/kan = present (Do)
- Pronouns: ŋa=I, nɤ=you(sg), ati=he, nirum=we, nuʔrum=you(pl), tarum=they
- 'kəmə' = transitive object marker
- 'lapkʰi' = see, 'cʰam' = know, 'lan' = beat, 'ka/kɤ' = go, 'ʒip' = sleep
- 'ne' = sentence-final question marker
--- END OF HUMAN EXAMPLE ---"""


def run_m4(puzzle: dict, model_key: str, api_keys: dict,
           human_resolution: str = "") -> dict:
    """
    M4 — Inspiración humana.
    Pasa el razonamiento humano de UN puzzle diferente como guía de proceso.
    NO ejecutar en el mismo puzzle cuyo razonamiento se usa (data leakage).
    Si human_resolution está vacío, devuelve error descriptivo.
    """
    if not human_resolution:
        return {
            "strategy":    "human_inspiration",
            "strategy_id": 4,
            "error":       "M4 omitido: human_resolution vacío (evitar data leakage en este puzzle)",
        }

    direction   = puzzle.get("translation_direction", "lang2en")
    train       = puzzle["train_pairs"]
    test_items  = puzzle["test_pairs"]
    refs        = [p["target"] for p in test_items]
    trans_instr = translation_instruction(direction, puzzle["language"])

    examples = "\n".join(f"  {p['source']} → {p['target']}" for p in train)
    tests    = "\n".join(f"  {i+1}. {item['source']}" for i, item in enumerate(test_items))

    prompt = f"""{METALINGUISTIC_INSTRUCTION}

{human_resolution}

Now apply the same style of reasoning to a NEW and COMPLETELY DIFFERENT linguistic puzzle.
The language below is unrelated to the example above — do not transfer any vocabulary or rules.

--- TRAINING EXAMPLES (new language) ---
{examples}

--- YOUR TASK ---
Following the same step-by-step reasoning shown in the human example,
write your DICTIONARY and GRAMMAR RULES, then {trans_instr}:
{tests}

Format your response EXACTLY as:
DICTIONARY:
[word mappings, one per line]

GRAMMAR RULES:
[rules, one per line]

TRANSLATIONS:
1. [translation only]
2. [translation only]
"""
    print(f"  [M4] {len(train)} frases — con guía de razonamiento humano")
    t0       = time.time()
    response = call_model(prompt, model_key, api_keys)
    latency  = round(time.time() - t0, 2)

    hyps    = extract_translations(response, len(test_items))
    metrics = compute_metrics(hyps, refs)
    _print_step(metrics)

    step = _step_record(
        {"step": 1, "num_train_pairs": len(train), "latency_s": latency, "metrics": metrics},
        prompt, response,
    )
    return {
        "strategy":     "human_inspiration",
        "strategy_id":  4,
        "steps":        [step],
        "best_metrics": metrics,
    }


# ─── M5 — Segundo revisor ─────────────────────────────────────────────────────

def run_m5(puzzle: dict, model_key: str, api_keys: dict) -> dict:
    """
    M5 — Segundo revisor (autocorrección iterativa).
    Turno 1: igual que M1. Turno 2: el modelo revisa su propia salida.
    Inspirado en Self-Refine (Madaan et al., 2023).
    """
    direction   = puzzle.get("translation_direction", "lang2en")
    train       = puzzle["train_pairs"]
    test_items  = puzzle["test_pairs"]
    refs        = [p["target"] for p in test_items]
    trans_instr = translation_instruction(direction, puzzle["language"])

    examples = "\n".join(f"  {p['source']} → {p['target']}" for p in train)
    tests    = "\n".join(f"  {i+1}. {item['source']}" for i, item in enumerate(test_items))

    prompt_t1 = f"""{METALINGUISTIC_INSTRUCTION}

--- TRAINING EXAMPLES ---
{examples}

--- YOUR TASK ---
Based on ALL examples above, write your inferred DICTIONARY and GRAMMAR RULES,
then {trans_instr}:
{tests}

Format your response EXACTLY as:
DICTIONARY:
[word mappings, one per line]

GRAMMAR RULES:
[rules, one per line]

TRANSLATIONS:
1. [translation only]
2. [translation only]
"""
    print("  [M5] Turno 1: traducción inicial")
    t0        = time.time()
    response1 = call_model(prompt_t1, model_key, api_keys)
    latency1  = round(time.time() - t0, 2)

    hyps1    = extract_translations(response1, len(test_items))
    metrics1 = compute_metrics(hyps1, refs)
    _print_step(metrics1)

    dict1, rules1 = extract_dict_and_rules(response1)
    dict1_display  = _truncate_context(dict1,  2000) if dict1  else '(empty)'
    rules1_display = _truncate_context(rules1, 1500) if rules1 else '(empty)'
    translations1_block = "\n".join(
        f"  {i+1}. {h}" for i, h in enumerate(hyps1)
    ) if hyps1 else "  (no translations extracted)"

    # Pausa más larga entre turnos para evitar rate limit en modelos 70B
    time.sleep(20)

    prompt_t2 = f"""{METALINGUISTIC_INSTRUCTION}

You have already analyzed the following language and produced a first draft.
Now act as a careful reviewer of your own work.

--- TRAINING EXAMPLES (same as before) ---
{examples}

--- YOUR FIRST-DRAFT ANALYSIS ---
DICTIONARY:
{dict1_display}

GRAMMAR RULES:
{rules1_display}

FIRST-DRAFT TRANSLATIONS:
{translations1_block}

--- YOUR TASK ---
1. CRITIQUE: For each translation above, check whether it is consistent with your
   dictionary and grammar rules. List specific contradictions or uncertainties.
   Be precise: quote the rule or entry that is violated or missing.

2. REVISED DICTIONARY: Update entries if needed.

3. REVISED GRAMMAR RULES: Update rules if needed.

4. FINAL TRANSLATIONS: Produce corrected translations based on your critique.
   If a translation was already correct, keep it unchanged.

IMPORTANT: The TRANSLATIONS section must contain ONLY the final translated sentences,
one per line, with no explanations, notes, or comments.

Format your response EXACTLY as:
CRITIQUE:
[numbered list of issues, or "No contradictions found."]

DICTIONARY:
[entries]

GRAMMAR RULES:
[rules]

TRANSLATIONS:
1. [final translation only]
2. [final translation only]
"""
    print("  [M5] Turno 2: autocorrección")
    t0        = time.time()
    response2 = call_model(prompt_t2, model_key, api_keys)
    latency2  = round(time.time() - t0, 2)

    hyps2    = extract_translations(response2, len(test_items))
    metrics2 = compute_metrics(hyps2, refs)
    _print_step(metrics2)

    delta = {
        "bleu_delta":  round(metrics2["corpus_bleu"]        - metrics1["corpus_bleu"],        2),
        "chrf_delta":  round(metrics2["corpus_chrfpp"]      - metrics1["corpus_chrfpp"],      2),
        "ter_delta":   round(metrics2["corpus_ter"]         - metrics1["corpus_ter"],         2),
        "em_delta":    round(metrics2["exact_match_corpus"] - metrics1["exact_match_corpus"], 2),
    }
    print(f"    Δ chrF={delta['chrf_delta']:+.2f}  "
          f"Δ TER={delta['ter_delta']:+.2f}  "
          f"Δ EM={delta['em_delta']:+.1f}%")

    best = metrics2 if metrics2["corpus_chrfpp"] >= metrics1["corpus_chrfpp"] else metrics1

    step1 = _step_record(
        {"step": 1, "role": "initial_draft",   "latency_s": latency1, "metrics": metrics1},
        prompt_t1, response1,
    )
    step2 = _step_record(
        {"step": 2, "role": "self_correction", "latency_s": latency2, "metrics": metrics2},
        prompt_t2, response2,
    )
    return {
        "strategy":     "self_correction",
        "strategy_id":  5,
        "steps":        [step1, step2],
        "best_metrics": best,
        "delta":        delta,
    }


# ─── Runner principal ──────────────────────────────────────────────────────────

STRATEGY_NAMES = {
    0: "verification",
    1: "baseline",
    2: "incremental_combinations",
    3: "step_by_step",
    4: "human_inspiration",
    5: "self_correction",
}

RUNNERS = {
    0: lambda p, m, k, **kw: run_m0(p, m, k),
    1: lambda p, m, k, **kw: run_m1(p, m, k),
    2: lambda p, m, k, **kw: run_m2(p, m, k),
    3: lambda p, m, k, **kw: run_m3(p, m, k),
    4: lambda p, m, k, **kw: run_m4(p, m, k, kw.get("human_resolution", "")),
    5: lambda p, m, k, **kw: run_m5(p, m, k),
}


def run_experiment(puzzle_path, model_key, api_keys,
                   strategies=(0, 1, 2, 3, 4, 5),
                   output_dir="results",
                   human_resolution=""):
    with open(puzzle_path) as f:
        puzzle = json.load(f)

    timestamp = datetime.datetime.now().isoformat()
    direction = puzzle.get("translation_direction", "lang2en")

    print(f"\n{'='*60}")
    print(f"  Idioma:    {puzzle['language']}  ({puzzle.get('language_family','')})")
    print(f"  Dirección: {direction}")
    print(f"  Modelo:    {model_key}  ({MODELS[model_key]['tier']})")
    print(f"  Tiempo:    {timestamp[:19]}")
    print(f"{'='*60}")

    log = {
        "metadata": {
            "timestamp":             timestamp,
            "puzzle_id":             puzzle["id"],
            "language":              puzzle["language"],
            "language_family":       puzzle.get("language_family", ""),
            "translation_direction": direction,
            "model_key":             model_key,
            "model_tier":            MODELS[model_key]["tier"],
            "num_train_pairs":       len(puzzle["train_pairs"]),
            "num_test_pairs":        len(puzzle["test_pairs"]),
            "strategies_run":        list(strategies),
            "log_full_responses":    LOG_FULL_RESPONSES,
        },
        "puzzle": {
            "id":                    puzzle["id"],
            "language":              puzzle["language"],
            "language_family":       puzzle.get("language_family", ""),
            "translation_direction": direction,
            "train_pairs":           puzzle["train_pairs"],
            "test_pairs":            puzzle["test_pairs"],
        },
        "results": {},
    }

    for s_id in strategies:
        name = STRATEGY_NAMES[s_id]
        print(f"\n→ M{s_id}: {name}")
        try:
            result = RUNNERS[s_id](
                puzzle, model_key, api_keys,
                human_resolution=human_resolution,
            )
            log["results"][name] = result
        except Exception as e:
            print(f"  ✗ Error: {e}")
            log["results"][name] = {"error": str(e)}

    log["summary"] = _build_summary(log["results"])

    Path(output_dir).mkdir(exist_ok=True)
    fname = f"{output_dir}/{puzzle['id']}_{model_key}_{timestamp[:10]}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Guardado: {fname}")
    _print_summary(log["summary"])
    return log


def _build_summary(results: dict) -> dict:
    summary = {}
    for name, res in results.items():
        if "error" in res:
            summary[name] = {"error": res["error"]}
            continue
        if name == "verification":
            m = res.get("metrics", {})
            summary[name] = {
                "bleu":           m.get("corpus_bleu", 0),
                "chrf":           m.get("corpus_chrfpp", 0),
                "ter":            m.get("corpus_ter", 100),
                "em":             m.get("exact_match_corpus", 0),
                "known_language": res.get("known_language", False),
                "hypothesis":     res.get("hypothesis", "")[:120],
                "reference":      res.get("reference", ""),
            }
        else:
            best = res.get("best_metrics", {})
            entry = {
                "best_bleu":       best.get("corpus_bleu", 0),
                "best_chrf":       best.get("corpus_chrfpp", 0),
                "best_ter":        best.get("corpus_ter", 100),
                "best_em":         best.get("exact_match_corpus", 0),
                "num_steps":       len(res.get("steps", [])),
                "best_hypotheses": best.get("hypotheses", []),
                "best_references": best.get("references", []),
            }
            if name == "incremental_combinations":
                entry["best_step"]  = res.get("best_step")
            if name == "step_by_step":
                entry["best_stage"] = res.get("best_stage")
            if name == "self_correction":
                entry["delta"] = res.get("delta", {})
            summary[name] = entry
    return summary


def _print_summary(summary: dict):
    labels = {
        "verification":             "M0",
        "baseline":                 "M1",
        "incremental_combinations": "M2",
        "step_by_step":             "M3",
        "human_inspiration":        "M4",
        "self_correction":          "M5",
    }
    print("\nRESUMEN:")
    print(f"  {'Metodología':<32} {'BLEU':>6} {'chrF':>6} {'TER':>7} {'EM%':>6}")
    print(f"  {'-'*32} {'-'*6} {'-'*6} {'-'*7} {'-'*6}")
    for name, s in summary.items():
        prefix = labels.get(name, "??")
        label  = f"{prefix} {name}"
        if "error" in s:
            print(f"  {label:<32} ERROR: {s['error'][:40]}")
        elif name == "verification":
            known = " ⚠CONOCIDO" if s["known_language"] else " ✓desconocido"
            print(f"  {'M0 verification':<32} {s['bleu']:6.2f} {s['chrf']:6.2f} "
                  f"{s['ter']:7.2f} {s['em']:6.1f}{known}")
        else:
            print(f"  {label:<32} {s['best_bleu']:6.2f} {s['best_chrf']:6.2f} "
                  f"{s['best_ter']:7.2f} {s['best_em']:6.1f}")
            if name == "self_correction" and "delta" in s:
                d = s["delta"]
                print(f"  {'   Δ vs draft':<32} {d['bleu_delta']:+6.2f} "
                      f"{d['chrf_delta']:+6.2f} {d['ter_delta']:+7.2f} "
                      f"{d['em_delta']:+6.1f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    API_KEYS = {
        "groq":       os.getenv("GROQ_API_KEY", ""),
        "cerebras":   os.getenv("CEREBRAS_API_KEY", ""),
        "sambanova":  os.getenv("SAMBANOVA_API_KEY", ""),
        "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
    }

    available = {k: v for k, v in MODELS.items() if API_KEYS.get(v["provider"])}
    if not available:
        print("\nERROR: configura GROQ_API_KEY y/o OPENROUTER_API_KEY")
        sys.exit(1)

    print("\nModelos disponibles:")
    for k, v in available.items():
        print(f"  ✓ {k:<35} ({v['tier']})")

    # ── CONFIGURACIÓN ACTIVA ──────────────────────────────────────────────────
    # Descomentar el bloque del modelo que se quiera lanzar.
    # Lanzar UN modelo a la vez para gestionar bien los rate limits.

    # ── OPCIÓN A: Llama-70B (lanzar ahora — cuota independiente del 8B) ──────
    PUZZLES = [
        # ── Hakhun (M4 desactivado en lang2en por ser el puzzle fuente) ──
        "puzzles/linguini_012018020100.json",  # Hakhun →EN  [Sino-Tibetana]
        "puzzles/linguini_012018020200.json",  # Hakhun EN→  [Sino-Tibetana]
        "puzzles/linguini_012022030100.json",  # N|uuki →EN  [Tuu]
        "puzzles/linguini_012019010100.json",  # Yonggom →EN  [Trans-New Guinea]
        "puzzles/linguini_012015040100.json",  # Wambaya →EN  [Mirndi]
        "puzzles/linguini_012023020100.json",  # Apurinã →EN  [Arawakan]
        "puzzles/linguini_012023020300.json",  # Apurinã EN→  [Arawakan]
        "puzzles/linguini_012019010200.json",  # Yonggom EN→  [Trans-New Guinea]
        "puzzles/linguini_012015040200.json",  # Wambaya EN→  [Mirndi]
        "puzzles/linguini_012022030200.json",  # N|uuki EN→  [Tuu]
        "puzzles/linguini_012023030100.json",  # Coastal Marind →EN  [Trans-New Guinea]
        "puzzles/linguini_012012010200.json",  # Dyirbal →EN  [Pama-Nyungan]
        "puzzles/linguini_012016030100.json",  # Kunuz Nubian →EN  [Nilo-Saharan]
        "puzzles/linguini_012008050100.json",  # Inuktitut →EN  [Eskimo-Aleut]
        "puzzles/linguini_012005010100.json",  # Tzotzil →EN  [Mayan]
        "puzzles/linguini_012006010100.json",  # Lakota →EN  [Siouan]
    ]
    MODELS_RUN = [
        # "llama-3.1-8b-groq",          # ← EN USO
        # "llama-3.3-70b-groq",         # ← SE LE ACABA LA CUOTA RÁPIDO
        # "llama-3.3-70b-sambanova",    # ← SE LE ACABA LA CUOTA RÁPIDO
        # "gpt-oss-120b-openrouter",    # ← EN USO (CASI COMPLETADO)
        # "gpt-oss-120b-cerebras",      # ← NO ES NECESARIO
    ]

    # ── OPCIÓN B: Relanzar 8B — solo lo que faltó (ejecutar mañana) ──────────
    # PUZZLES_8B_RELAUNCH = [
    #     # Puzzles completamente fallidos (todo rate limit)
    #     "puzzles/linguini_012006010100.json",  # Lakota
    #     "puzzles/linguini_012008050100.json",  # Inuktitut
    #     "puzzles/linguini_012012010200.json",  # Dyirbal
    #     "puzzles/linguini_012022030200.json",  # N|uuki en2lang
    #     "puzzles/linguini_012023030100.json",  # Coastal Marind
    #     # Tzeltal y Kunuz Nubian (bug refs + rate limit)
    #     "puzzles/linguini_012005010100.json",  # Tzeltal
    #     "puzzles/linguini_012016030100.json",  # Kunuz Nubian
    #     # Parciales — solo las estrategias que fallaron
    #     # (para estos, descomentar y ajustar strategies=(...) manualmente)
    #     # "puzzles/linguini_012018020200.json",  # Hakhun en2lang → solo M5
    #     # "puzzles/linguini_012019010100.json",  # Yonggom lang2en → solo M5
    #     # "puzzles/linguini_012019010200.json",  # Yonggom en2lang → solo M5
    #     # "puzzles/linguini_012022030100.json",  # Ngemba → solo M2
    #     # "puzzles/linguini_012023020100.json",  # Apurinã lang2en → solo M3
    #     # "puzzles/linguini_012023020300.json",  # Apurinã en2lang → solo M2
    # ]
    # MODELS_RUN = ["llama-3.1-8b-groq"]
    # ─────────────────────────────────────────────────────────────────────────

    for puzzle_path in PUZZLES:
        if not Path(puzzle_path).exists():
            print(f"  [SKIP] No encontrado: {puzzle_path}")
            continue

        puzzle_id = Path(puzzle_path).stem
        is_hakhun = "012018020100" in puzzle_id

        for model_key in [m for m in MODELS_RUN if m in available]:
            run_experiment(
                puzzle_path      = puzzle_path,
                model_key        = model_key,
                api_keys         = API_KEYS,
                strategies       = (0, 1, 2, 3, 4, 5),
                output_dir       = "results",
                human_resolution = "" if is_hakhun else HUMAN_RESOLUTION_HAKHUN,
            )