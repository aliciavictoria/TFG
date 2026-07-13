"""
run_m2_combinaciones.py — M2-combinaciones (trabajo futuro)
============================================================
Estrategia original acordada con el tutor:
  - Explora TODAS las combinaciones C(n,k) para k=2, 3, ..., n
  - Primero todas las parejas, luego todos los triples, etc.
  - En cada iteración: recibe las k frases de ESA combinación
    + el diccionario y reglas acumulados de iteraciones anteriores
  - El modelo puede confirmar, corregir o eliminar hipótesis previas
  - Se reporta el mejor chrF++ de todas las iteraciones

Diferencia con M2-prefijos (implementado):
  M2-prefijos:      (f1,f2) → (f1,f2,f3) → (f1,f2,f3,f4) → ...
  M2-combinaciones: (f1,f2) → (f1,f3) → (f1,f4) → (f2,f3) → (f2,f4) → ...
                    → (f1,f2,f3) → (f1,f2,f4) → ... → (f1,...,fn)

USO:
  export SAMBANOVA_API_KEY="..."
  python3 run_m2_combinaciones.py

PARÁMETROS (editar abajo):
  PUZZLE_PATH : puzzle a ejecutar
  MODEL_KEY   : modelo a usar
  MAX_K       : tamaño máximo de combinación (2=solo parejas, 3=hasta triples...)
                con n=10: k=2→45 iters, k=3→165 iters, k=4→375 iters
"""

import json, os, sys, time, itertools, datetime, re
from pathlib import Path
import requests

# ─── Parámetros ───────────────────────────────────────────────────────────────

PUZZLE_PATH = "puzzles/linguini_012018020100.json"  # Hakhun lang2en
MODEL_KEY   = "gpt-oss-120b-openrouter"
MAX_K       = 3   # frases

# ─── Modelos y proveedores ────────────────────────────────────────────────────

MODELS = {
    "llama-3.3-70b-sambanova": {
        "provider": "sambanova",
        "model_id": "Meta-Llama-3.3-70B-Instruct",
        "tier": "medium",
    },
    "gpt-oss-120b-openrouter": {
        "provider": "openrouter",
        "model_id": "openai/gpt-oss-120b:free",
        "tier": "strong",
    },
    "llama-3.1-8b-groq": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "tier": "weak",
    },
}

PROVIDER_URLS = {
    "sambanova":  "https://api.sambanova.ai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "groq":       "https://api.groq.com/openai/v1/chat/completions",
}

# ─── Extractor (idéntico al de experiment_runner_final.py) ────────────────────

METALINGUISTIC_INSTRUCTION = """You are an expert linguistic analyst performing inductive reasoning on an unknown language.
You have never seen this language before. You must reason exclusively from the provided examples using your meta-linguistic knowledge of how human languages work in general (word order typology, morphological patterns, agreement systems, affixation, etc.).

Core principles:
- Do NOT use any prior knowledge of this specific language or its family
- Treat every hypothesis as provisional and subject to revision
- Always show your reasoning explicitly before reaching any conclusion
- A wrong rule confidently stated is worse than uncertainty acknowledged"""

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
_SKIP_RE        = re.compile(r'^(\*\*|STEP|DICTIONARY|GRAMMAR|ANALYSIS|NOTE|#|\[|Add |Update |Delete |Correct |Check |Remove )', re.IGNORECASE)
_DICT_ENTRY_RE  = re.compile(r'^[\w\u00c0-\u024f\u0250-\u02ff\u1e00-\u1eff][\w\u00c0-\u024f\u0250-\u02ff\u1e00-\u1eff\-]*(?:\s+[\w\u00c0-\u024f\u0250-\u02ff\u1e00-\u1eff][\w\u00c0-\u024f\u0250-\u02ff\u1e00-\u1eff\-]*)*\s*[=:]', re.IGNORECASE | re.UNICODE)
_INSTRUCTION_RE = re.compile(r'\b(update|add|delete|remove|correct|change|revise|fix|insert)\b.{0,60}\b(entry|rule|dictionary|suffix|morpheme|word)\b', re.IGNORECASE)
_META_RE        = re.compile(r'\b(word order|SOV|SVO|VSO|OVS|morpheme|phonolog|syntax|grammar rule|question particle|subject marker|object marker|aspect marker|tense marker|allomorph|affix|suffix|prefix|conjugat|declension|inflect)\b', re.IGNORECASE)

def _clean(t):
    t = t.strip().strip('*').strip('"').strip('\u201c').strip('\u201d').strip()
    return re.sub(r'\s*[\(\[].{0,100}[\)\]]\s*$', '', t).strip()

def _is_instruction(t): return bool(_INSTRUCTION_RE.search(t))
def _is_meta(t):        return bool(_META_RE.search(t))

def extract_translations(response, n):
    header  = re.search(r'TRANSLATIONS\b[^\n]*\n', response, re.IGNORECASE)
    section = response[header.end():] if header else response
    stop    = re.search(r'\n\s*(?:ANALYSIS|DICTIONARY|GRAMMAR\s+RULE|NOTE|EXPLANATION|CRITIQUE)\s*[:\*\n]', section, re.IGNORECASE)
    if stop: section = section[:stop.start()]
    arrows  = [_clean(m.group(1)) for m in _ARROW_RE.finditer(section)]
    arrows  = [t for t in arrows if 4 < len(t) < 120 and not _is_instruction(t) and not _is_meta(t)]
    if len(arrows) >= n: return arrows[:n]
    nums = []
    for m in _NUMBERED_RE.finditer(section):
        t = _clean(m.group(1))
        if t and not _SKIP_RE.match(t) and not _DICT_ENTRY_RE.match(t) and not _is_instruction(t) and not _is_meta(t) and 4 < len(t) < 120:
            nums.append(t)
    return (arrows + [h for h in nums if h not in arrows])[:n]

def extract_dict_and_rules(response):
    section, dl, rl = None, [], []
    for line in response.split('\n'):
        u = line.upper().strip()
        if 'DICTIONARY' in u and ':' in u: section = 'dict'; continue
        if 'GRAMMAR' in u and 'RULE' in u: section = 'rules'; continue
        if 'TRANSLATION' in u: section = 'trans'; continue
        if section == 'dict'  and line.strip(): dl.append(line.strip())
        elif section == 'rules' and line.strip(): rl.append(line.strip())
    return '\n'.join(dl), '\n'.join(rl)

def _truncate(text, max_chars=3000):
    if not text or len(text) <= max_chars: return text
    lines, result, total = text.split('\n'), [], 0
    for line in lines:
        if total + len(line) > max_chars:
            result.append(f'... [{len(lines)-len(result)} entradas omitidas por longitud]')
            break
        result.append(line); total += len(line) + 1
    return '\n'.join(result)

def call_model(prompt, model_key, api_keys, timeout=150):
    model    = MODELS[model_key]
    provider = model['provider']
    key      = api_keys.get(provider, '')
    if not key: raise ValueError(f"Falta API key para '{provider}'")
    headers  = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    if provider == 'openrouter':
        headers['HTTP-Referer'] = 'https://tfg-linguistic-olympiad'
        headers['X-Title']      = 'TFG M2-combinaciones'
    tokens_key = 'max_completion_tokens' if provider == 'cerebras' else 'max_tokens'
    payload = {'model': model['model_id'], 'messages': [{'role': 'user', 'content': prompt}],
               'temperature': 0.0, tokens_key: 2048}
    for attempt in range(3):
        resp = requests.post(PROVIDER_URLS[provider], headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            print(f'    [429] Rate limit, esperando {wait}s...')
            time.sleep(wait); continue
        resp.raise_for_status()
        data = resp.json()
        if 'error' in data: raise RuntimeError(f"Error de {provider}: {data['error']}")
        msg  = data['choices'][0]['message']
        return (msg.get('content') or msg.get('reasoning') or '').strip()
    raise RuntimeError('Rate limit persistente tras 3 intentos')

def compute_metrics(hyps, refs):
    try: import sacrebleu as sb
    except ImportError: return {'error': 'sacrebleu no disponible'}
    if not hyps or not refs:
        return {'corpus_bleu': 0.0, 'corpus_chrfpp': 0.0, 'corpus_ter': 100.0,
                'exact_match_corpus': 0.0, 'hypotheses': [], 'references': []}
    refs = [r[0] if isinstance(r, list) else str(r) for r in refs]
    n    = min(len(hyps), len(refs))
    h, r = hyps[:n], refs[:n]
    return {
        'corpus_bleu':        round(sb.corpus_bleu(h, [r]).score, 2),
        'corpus_chrfpp':      round(sb.corpus_chrf(h, [r], word_order=2).score, 2),
        'corpus_ter':         round(sb.corpus_ter(h, [r]).score, 2),
        'exact_match_corpus': round(sum(re.sub(r'[^\w\s]','',a.lower()).strip() ==
                                        re.sub(r'[^\w\s]','',b.lower()).strip()
                                        for a,b in zip(h,r)) / n * 100, 2),
        'hypotheses': h, 'references': r,
    }

# ─── M2-combinaciones ─────────────────────────────────────────────────────────

def run_m2_combinaciones(puzzle_path, model_key, api_keys, max_k=2, output_dir='results_m2_combinaciones'):
    with open(puzzle_path) as f:
        puzzle = json.load(f)

    train      = puzzle['train_pairs']
    test_items = puzzle['test_pairs']
    refs       = [p['target'] for p in test_items]
    n          = len(train)
    direction  = puzzle.get('translation_direction', 'lang2en')
    lang       = puzzle['language']
    max_k      = min(max_k, n)

    total_iters = sum(len(list(itertools.combinations(range(n), k)))
                      for k in range(2, max_k + 1))

    print(f"\n{'='*60}")
    print(f"  M2-COMBINACIONES — {lang} ({direction})")
    print(f"  Modelo: {model_key}  ({MODELS[model_key]['tier']})")
    print(f"  n={n} frases · k=2..{max_k} · {total_iters} iteraciones")
    print(f"{'='*60}")

    trans_instr = (f"translate into {lang} (the unknown language)"
                   if direction == 'en2lang' else "translate into English")
    tests = '\n'.join(f"  {i+1}. {item['source']}" for i, item in enumerate(test_items))

    steps         = []
    current_dict  = ''
    current_rules = ''
    best_chrf     = -1
    best_step     = None
    iter_count    = 0

    for k in range(2, max_k + 1):
        combos = list(itertools.combinations(range(n), k))
        print(f'\n  ── k={k}: {len(combos)} combinaciones ──')

        for combo in combos:
            iter_count += 1
            combo_pairs = [train[i] for i in combo]
            frase_ids   = [i + 1 for i in combo]
            examples    = '\n'.join(f"  {p['source']} -> {p['target']}" for p in combo_pairs)

            # Hipótesis acumulada (igual que M2-prefijos)
            _d = _truncate(current_dict)  if current_dict  else '(empty)'
            _r = _truncate(current_rules) if current_rules else '(empty)'
            prev = f"""
--- YOUR CURRENT HYPOTHESIS (provisional — review critically) ---
DICTIONARY:
{_d}

GRAMMAR RULES:
{_r}

IMPORTANT: New examples may CONFIRM, REFINE or CONTRADICT entries above.
Delete proven-wrong entries; correct imprecise ones.
""" if (current_dict or current_rules) else '\n(No prior knowledge — first iteration.)\n'

            prompt = f"""{METALINGUISTIC_INSTRUCTION}

ITERATION {iter_count}/{total_iters} — Combination of frases {frase_ids} (k={k}):
{prev}
--- NEW EXAMPLES (this combination) ---
{examples}

--- YOUR TASK (in this order) ---

STEP A · ANALYSIS
Identify morphemes in these examples. Check for contradictions with your
current hypothesis. Think aloud. Do NOT translate the test sentences yet.

STEP B · REVISED DICTIONARY
One entry per line: morpheme = meaning
Mark deleted entries: [DELETED: entry — reason]

STEP C · REVISED GRAMMAR RULES
Numbered list. Mark deleted rules: [DELETED: rule — reason]

STEP D · TRANSLATIONS
{trans_instr.capitalize()} — write ONLY the translation on each line:
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
            print(f'    [{iter_count:3d}/{total_iters}] frases={frase_ids}', end=' ', flush=True)
            t0 = time.time()
            try:
                response = call_model(prompt, model_key, api_keys)
                latency  = round(time.time() - t0, 2)

                # Actualizar dict y reglas acumulados
                nd, nr = extract_dict_and_rules(response)
                if nd: current_dict  = nd
                if nr: current_rules = nr

                hyps    = extract_translations(response, len(test_items))
                metrics = compute_metrics(hyps, refs)
                chrf    = metrics['corpus_chrfpp']
                print(f'chrF={chrf:.1f}  ({len(hyps)}/{len(test_items)} hyps)')

                step = {
                    'iter':        iter_count,
                    'k':           k,
                    'combo':       list(combo),
                    'frase_ids':   frase_ids,
                    'latency_s':   latency,
                    'accumulated_dict':  current_dict,
                    'accumulated_rules': current_rules,
                    'metrics':     metrics,
                }
                steps.append(step)

                if chrf > best_chrf:
                    best_chrf = chrf
                    best_step = step
                    print(f'      ★ nuevo mejor: chrF={chrf:.1f}  hyps={hyps}')

            except Exception as e:
                print(f'ERR: {e}')
                steps.append({'iter': iter_count, 'k': k, 'combo': list(combo),
                               'frase_ids': frase_ids, 'error': str(e)})

            time.sleep(10)  # respetar rate limits

    # Guardar
    timestamp = datetime.datetime.now().isoformat()
    log = {
        'metadata': {
            'timestamp':   timestamp,
            'puzzle_id':   puzzle['id'],
            'language':    lang,
            'direction':   direction,
            'model_key':   model_key,
            'n_train':     n,
            'max_k':       max_k,
            'total_iters': total_iters,
            'strategy':    'm2_combinaciones',
        },
        'puzzle': {
            'train_pairs': train,
            'test_pairs':  test_items,
        },
        'steps':     steps,
        'best_step': best_step,
        'best_chrf': best_chrf,
    }

    Path(output_dir).mkdir(exist_ok=True)
    pid   = puzzle['id'].replace('linguini_', '')
    fname = f"{output_dir}/m2_combinaciones_{pid}_{model_key}_{timestamp[:10]}.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f'\n✓ Guardado: {fname}')
    print(f'\nRESUMEN M2-COMBINACIONES:')
    print(f'  Iteraciones OK: {len([s for s in steps if "error" not in s])}/{total_iters}')
    print(f'  Mejor chrF++:   {best_chrf:.1f}')
    if best_step:
        print(f'  Mejor combo:    k={best_step["k"]}  frases={best_step["frase_ids"]}')
        print(f'  Hipótesis:      {best_step["metrics"].get("hypotheses", [])}')
    return log


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    API_KEYS = {
        'sambanova':  os.getenv('SAMBANOVA_API_KEY', ''),
        'openrouter': os.getenv('OPENROUTER_API_KEY', ''),
        'groq':       os.getenv('GROQ_API_KEY', ''),
    }

    available = {k: v for k, v in MODELS.items() if API_KEYS.get(v['provider'])}
    print('\nModelos disponibles:')
    for k, v in available.items():
        print(f'  ✓ {k:<35} ({v["tier"]})')

    if MODEL_KEY not in available:
        print(f'\nERROR: {MODEL_KEY} no disponible. Configura la API key.')
        sys.exit(1)

    if not Path(PUZZLE_PATH).exists():
        print(f'\nERROR: no encontrado {PUZZLE_PATH}')
        sys.exit(1)

    run_m2_combinaciones(
        puzzle_path = PUZZLE_PATH,
        model_key   = MODEL_KEY,
        api_keys    = API_KEYS,
        max_k       = MAX_K,
        output_dir  = 'results_m2_combinaciones',
    )