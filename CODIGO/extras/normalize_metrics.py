"""
normalize_sgpl.py
=================
Recalcula las métricas de todos los JSONs de resultados eliminando los
marcadores de número gramatical (sg) y (pl) de las referencias gold.

Ejemplos de normalización:
  'Did you(sg) sleep?'              → 'Did you sleep?'
  'Do they know you(pl)?'           → 'Do they know you?'
  'You_{sg} drank the water.'       → 'You drank the water.'
  'your_{sg} brother'               → 'your brother'
  'We really chased you_{pl} away.' → 'We really chased you away.'

USO:
  python3 normalize_sgpl.py --input results/ --output results_normalized/

Los ficheros originales NO se modifican. Los normalizados se guardan en
el directorio de salida con el mismo nombre.
"""

import json, re, argparse, sacrebleu
from pathlib import Path
from copy import deepcopy


def normalize_sgpl(text: str) -> str:
    """Elimina (sg), (pl), _{sg}, _{pl} y variantes de un texto."""
    text = re.sub(r'\s*\(s?g\.?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\(pl\.?\)',  '', text, flags=re.IGNORECASE)
    text = re.sub(r'_\{?s?g\.?\}?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'_\{?pl\.?\}?',  '', text, flags=re.IGNORECASE)
    return re.sub(r'  +', ' ', text).strip()


def recompute_metrics(hyps: list, refs: list) -> dict:
    """Recalcula BLEU, chrF++, TER y Exact Match."""
    if not hyps or not refs:
        return {'corpus_bleu': 0.0, 'corpus_chrfpp': 0.0,
                'corpus_ter': 100.0, 'exact_match_corpus': 0.0}
    n = min(len(hyps), len(refs))
    h, r = hyps[:n], refs[:n]
    em = sum(
        re.sub(r'[^\w\s]', '', a.lower()).strip() ==
        re.sub(r'[^\w\s]', '', b.lower()).strip()
        for a, b in zip(h, r)
    ) / n * 100
    return {
        'corpus_bleu':        round(sacrebleu.corpus_bleu(h, [r]).score, 2),
        'corpus_chrfpp':      round(sacrebleu.corpus_chrf(h, [r], word_order=2).score, 2),
        'corpus_ter':         round(sacrebleu.corpus_ter(h, [r]).score, 2),
        'exact_match_corpus': round(em, 2),
    }


def process_file(fpath: Path) -> tuple:
    """
    Procesa un JSON de resultados y devuelve (d_normalizado, n_cambios).
    """
    with open(fpath, encoding='utf-8') as f:
        d = json.load(f)

    d_new = deepcopy(d)
    changes = 0

    # ── 1. Normalizar referencias gold en puzzle.test_pairs ────────────────
    for tp in d_new.get('puzzle', {}).get('test_pairs', []):
        if isinstance(tp.get('target'), list):
            tp['target'] = [normalize_sgpl(t) for t in tp['target']]
        elif isinstance(tp.get('target'), str):
            orig = tp['target']
            tp['target'] = normalize_sgpl(orig)
            if tp['target'] != orig:
                changes += 1

    # ── 2. Recalcular métricas en cada step de cada estrategia ────────────
    for strat, res in d_new.get('results', {}).items():
        if 'error' in res:
            continue

        # M0 verificación
        if strat == 'verification':
            ref_orig = res.get('reference', '')
            ref_norm = normalize_sgpl(ref_orig)
            if ref_norm != ref_orig:
                res['reference'] = ref_norm
                hyp = res.get('hypothesis', '')
                new_m = recompute_metrics([hyp], [ref_norm])
                res.setdefault('metrics', {}).update(new_m)
                res['metrics']['references'] = [ref_norm]
                changes += 1
            continue

        # M1..M5: iterar sobre steps
        for step in res.get('steps', []):
            m = step.get('metrics', {})
            refs_raw = m.get('references', [])
            hyps     = m.get('hypotheses', [])
            if not refs_raw:
                continue

            refs_norm = [
                normalize_sgpl(r[0] if isinstance(r, list) else r)
                for r in refs_raw
            ]
            refs_orig = [
                (r[0] if isinstance(r, list) else r) for r in refs_raw
            ]

            if refs_norm != refs_orig:
                new_m = recompute_metrics(hyps, refs_norm)
                m.update(new_m)
                m['references'] = refs_norm
                # Actualizar también sentence_chrfpp, sentence_bleu, sentence_ter
                if hyps and refs_norm:
                    n = min(len(hyps), len(refs_norm))
                    m['sentence_chrfpp'] = [
                        round(sacrebleu.sentence_chrf(h, [r], word_order=2).score, 2)
                        for h, r in zip(hyps[:n], refs_norm[:n])
                    ]
                    m['sentence_bleu'] = [
                        round(sacrebleu.sentence_bleu(h, [r]).score, 2)
                        for h, r in zip(hyps[:n], refs_norm[:n])
                    ]
                    m['sentence_em'] = [
                        re.sub(r'[^\w\s]','',a.lower()).strip() ==
                        re.sub(r'[^\w\s]','',b.lower()).strip()
                        for a, b in zip(hyps[:n], refs_norm[:n])
                    ]
                changes += 1

        # Recalcular best_metrics
        steps = res.get('steps', [])
        valid = [s for s in steps
                 if 'error' not in s and s.get('metrics', {}).get('hypotheses')]
        if valid:
            best = max(valid,
                       key=lambda s: s['metrics'].get('corpus_chrfpp', 0))
            res['best_metrics'] = best['metrics']

    # ── 3. Recalcular summary ─────────────────────────────────────────────
    for strat, s in d_new.get('summary', {}).items():
        if 'error' in s:
            continue
        res = d_new['results'].get(strat, {})
        if strat == 'verification':
            m = res.get('metrics', {})
            s.update({
                'bleu': m.get('corpus_bleu', 0),
                'chrf': m.get('corpus_chrfpp', 0),
                'ter':  m.get('corpus_ter', 100),
                'em':   m.get('exact_match_corpus', 0),
            })
        else:
            bm = res.get('best_metrics', {})
            if bm:
                s.update({
                    'best_bleu': bm.get('corpus_bleu', 0),
                    'best_chrf': bm.get('corpus_chrfpp', 0),
                    'best_ter':  bm.get('corpus_ter', 100),
                    'best_em':   bm.get('exact_match_corpus', 0),
                })
            if strat == 'self_correction':
                steps = res.get('steps', [])
                if len(steps) >= 2:
                    t1 = steps[0].get('metrics', {}).get('corpus_chrfpp', 0)
                    t2 = steps[1].get('metrics', {}).get('corpus_chrfpp', 0)
                    s.setdefault('delta', {})['chrf_delta'] = round(t2 - t1, 2)

    d_new['metadata']['sgpl_normalized'] = True
    return d_new, changes


def main():
    parser = argparse.ArgumentParser(description='Normaliza (sg)/(pl) en JSONs de resultados.')
    parser.add_argument('--input',  default='results',            help='Directorio con los JSONs originales')
    parser.add_argument('--output', default='results_normalized', help='Directorio de salida')
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob('*.json'))
    if not files:
        print(f"No se encontraron JSONs en '{input_dir}'")
        return

    print(f"Normalizando {len(files)} ficheros de '{input_dir}' → '{output_dir}'\n")

    total_files_changed = 0
    total_steps_changed = 0

    for fpath in files:
        try:
            d_new, changes = process_file(fpath)
            out = output_dir / fpath.name
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(d_new, f, ensure_ascii=False, indent=2)

            lang    = d_new['metadata'].get('language', '?')
            direc   = d_new['metadata'].get('translation_direction', '?')
            model   = d_new['metadata'].get('model_key', '?')

            if changes:
                total_files_changed += 1
                total_steps_changed += changes
                print(f"  ✓ {lang:<18} {direc:<8} {model:<30} → {changes} pasos recalculados")
            else:
                print(f"  — {lang:<18} {direc:<8} {model:<30} → sin cambios (no hay sg/pl)")

        except Exception as e:
            print(f"  ✗ ERROR en {fpath.name}: {e}")

    print(f"\n{'─'*65}")
    print(f"Ficheros con cambios: {total_files_changed}/{len(files)}")
    print(f"Pasos recalculados:   {total_steps_changed}")
    print(f"Guardado en: {output_dir}/")
    print("\nNOTA: los ficheros originales en 'results/' no han sido modificados.")


if __name__ == '__main__':
    main()