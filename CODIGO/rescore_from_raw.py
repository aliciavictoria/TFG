"""
rescore_from_raw.py
====================
Segundo paso del pipeline, DESACOPLADO de la API.

Lee los logs crudos guardados por experiment_runner_v5.py en raw_logs/
(un .jsonl por puzzle+modelo, con el prompt y la respuesta completa de
CADA llamada, sin parsear) y genera, sin volver a llamar a ningún modelo:

  1. results/<puzzle>_<model>_<fecha>.json   — mismo formato que antes,
     pero recalculado con el extractor nuevo (indexado por número).
  2. Un resumen en consola de qué quedó completo y qué no.

Como esto NO toca la API, se puede ejecutar tantas veces como haga falta
mientras se afina extract_translations() en experiment_runner_v5.py —
coste cero de cuota.

Uso:
    python3 rescore_from_raw.py raw_logs/linguini_012006010100_gpt-oss-120b-openrouter.jsonl \
        --puzzle puzzles/linguini_012006010100.json --out results_v2
"""
import json, sys, argparse, datetime
from pathlib import Path

# Reutiliza el extractor y las métricas del runner parcheado, no los reimplementa
from experiment_runner_row import extract_translations, compute_metrics, extract_dict_and_rules

STRATEGY_NAMES = {
    "verification":             "verification",
    "baseline":                 "baseline",
    "incremental_combinations": "incremental_combinations",
    "step_by_step":             "step_by_step",
    "human_inspiration":        "human_inspiration",
    "self_correction":          "self_correction",
}


def load_raw(jsonl_path: str) -> list:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def rescore_puzzle(raw_records: list, puzzle: dict) -> dict:
    test_items = puzzle["test_pairs"]
    refs_full  = [p["target"] for p in test_items]
    n_test     = len(test_items)

    by_strategy = {}
    for rec in raw_records:
        by_strategy.setdefault(rec["strategy"], []).append(rec)

    results = {}

    # Qué paso se considera "principal" en cada metodología multi-paso,
    # según la Tabla 5.4 de la memoria ("Selección optimista del mejor
    # paso" -> decisión: el ÚLTIMO paso es el resultado principal; el
    # mejor paso se conserva solo como información diagnóstica).
    PRIMARY_STEP_RULE = {
        "incremental_combinations": "last",   # M2: último paso = todos los ejemplos
        "step_by_step":             "last",   # M3: última etapa = sintaxis (todo el conocimiento acumulado)
        "self_correction":          "last",   # M5: turno 2 = autocorrección (no "mejor de los 2 turnos")
    }

    for strategy, records in by_strategy.items():
        if strategy == "verification":
            rec = records[0]
            ref = test_items[0]["target"]
            hyp = rec["response"].strip()
            metrics = compute_metrics([hyp], [ref])
            results[strategy] = {
                "strategy": "verification", "strategy_id": 0,
                "reference": ref, "hypothesis": hyp,
                "metrics": metrics,
                "known_language": metrics.get("corpus_chrfpp", 0) > 20,
            }
            continue

        steps = []
        for rec in records:
            hyps = extract_translations(rec["response"], n_test)
            metrics = compute_metrics(hyps, refs_full)
            steps.append({
                "step": rec["step"],
                "metrics": metrics,
                "complete": metrics.get("complete", False),
            })

        best = max(steps, key=lambda s: s["metrics"].get("corpus_chrfpp", 0))
        last = steps[-1]
        rule = PRIMARY_STEP_RULE.get(strategy, "last")
        primary = last if rule == "last" else best

        results[strategy] = {
            "strategy":            strategy,
            "steps":               steps,
            "primary_metrics":     primary["metrics"],     # <- usar este para la memoria
            "primary_step":        primary["step"],
            "primary_rule":        rule,
            "best_metrics":        best["metrics"],         # <- solo diagnóstico (Tabla 5.4)
            "best_step":           best["step"],
            "all_steps_complete":  all(s["complete"] for s in steps),
        }

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_jsonl", help="raw_logs/<puzzle>_<model>.jsonl")
    ap.add_argument("--puzzle", required=True, help="ruta al JSON original del puzzle")
    ap.add_argument("--out", default="results_v2", help="carpeta de salida")
    args = ap.parse_args()

    with open(args.puzzle, encoding="utf-8") as f:
        puzzle = json.load(f)

    raw_records = load_raw(args.raw_jsonl)
    stem = Path(args.raw_jsonl).stem  # "<puzzle_id>_<model_key>"
    prefix = puzzle["id"] + "_"
    model_key = stem[len(prefix):] if stem.startswith(prefix) else stem

    results = rescore_puzzle(raw_records, puzzle)

    log = {
        "metadata": {
            "rescored_at":           datetime.datetime.now().isoformat(),
            "puzzle_id":             puzzle["id"],
            "language":              puzzle.get("language"),
            "translation_direction": puzzle.get("translation_direction"),
            "model_key":             model_key,
            "num_test_pairs":        len(puzzle["test_pairs"]),
            "source_raw_log":        args.raw_jsonl,
        },
        "results": results,
    }

    Path(args.out).mkdir(exist_ok=True)
    fname = f"{args.out}/{puzzle['id']}_{model_key}_rescored.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Guardado: {fname}\n")
    print(f"{'Estrategia':28s} {'chrF++ (principal)':>20s}  {'chrF++ (mejor, diagn.)':>24s}  {'completo'}")
    for strat, res in results.items():
        if strat == "verification":
            m = res["metrics"]
            print(f"{'verification':28s} {m.get('corpus_chrfpp',0):20.2f}  {'—':>24s}  "
                  f"{'conocido' if res['known_language'] else 'desconocido'}")
        else:
            mp = res["primary_metrics"]
            mb = res["best_metrics"]
            tag = "✓ completo" if res["all_steps_complete"] else "⚠ algún paso incompleto"
            print(f"{strat:28s} {mp.get('corpus_chrfpp',0):20.2f}  {mb.get('corpus_chrfpp',0):24.2f}  {tag}")


if __name__ == "__main__":
    main()