"""
generate_report.py
==================
Lee los logs JSON de los experimentos y genera:
  - Tabla comparativa de BLEU por modelo y estrategia (Markdown)
  - Evolución de BLEU por pasos en estrategias incrementales
  - Ejemplos de prompts/respuestas para la memoria del TFG

Uso:
    python generate_report.py                    # procesa todos los JSON en results/
    python generate_report.py results/mi_log.json
"""

import json
import sys
import glob
from pathlib import Path
from datetime import datetime


def load_logs(paths: list[str]) -> list[dict]:
    logs = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                logs.append(json.load(f))
        except Exception as e:
            print(f"[WARN] No se pudo leer {p}: {e}")
    return logs


def summary_table(logs: list[dict]) -> str:
    """Genera tabla Markdown comparativa de resultados."""
    lines = [
        "## Tabla comparativa de resultados (BLEU)\n",
        "| Puzzle | Idioma | Modelo | Tier | Baseline | Incremental (final) | Incremental (best) | Step-by-Step (final) | Step-by-Step (best) |",
        "|--------|--------|--------|------|----------|--------------------|--------------------|---------------------|---------------------|"
    ]

    for log in logs:
        meta = log.get("metadata", {})
        res = log.get("results", {})
        puzzle = meta.get("puzzle_id", "?")
        lang = meta.get("language", "?")
        model = meta.get("model_key", "?")
        tier = meta.get("model_info", {}).get("tier", "?")

        def safe_bleu(strategy_name, key="final_bleu"):
            r = res.get(strategy_name, {})
            if "error" in r:
                return "ERROR"
            v = r.get(key, r.get("final_bleu", "N/A"))
            return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)

        row = (
            f"| {puzzle} | {lang} | {model} | {tier} "
            f"| {safe_bleu('baseline')} "
            f"| {safe_bleu('incremental_combinations')} "
            f"| {safe_bleu('incremental_combinations', 'best_bleu')} "
            f"| {safe_bleu('step_by_step')} "
            f"| {safe_bleu('step_by_step', 'best_bleu')} |"
        )
        lines.append(row)

    return "\n".join(lines)


def bleu_progression(log: dict) -> str:
    """Muestra evolución de BLEU por pasos para estrategias incrementales."""
    meta = log.get("metadata", {})
    res = log.get("results", {})
    lang = meta.get("language", "?")
    model = meta.get("model_key", "?")

    lines = [f"\n### Evolución BLEU — {lang} | {model}\n"]

    for strat_name in ["incremental_combinations", "step_by_step"]:
        strat = res.get(strat_name, {})
        if "error" in strat or not strat:
            continue
        steps = strat.get("steps", [])
        lines.append(f"**{strat_name}:**\n")
        lines.append("| Paso | Descripción | BLEU |")
        lines.append("|------|-------------|------|")
        for s in steps:
            desc = s.get("stage", f"C(n,{s.get('combo_size','?')})")
            bleu = s.get("bleu", {}).get("corpus_bleu", "N/A")
            lines.append(f"| {s['step']} | {desc} | {bleu} |")
        lines.append("")

    return "\n".join(lines)


def example_prompt_response(log: dict, strategy: str = "baseline", step: int = 0) -> str:
    """Extrae un ejemplo de prompt+respuesta para la memoria."""
    res = log.get("results", {}).get(strategy, {})
    if "error" in res or not res:
        return f"_(No hay datos para {strategy})_"

    steps = res.get("steps", [])
    if not steps or step >= len(steps):
        return "_(Sin pasos disponibles)_"

    s = steps[step]
    meta = log.get("metadata", {})
    lang = meta.get("language", "?")
    model = meta.get("model_key", "?")

    lines = [
        f"\n### Ejemplo de interacción — {strategy} | {lang} | {model} (paso {step+1})\n",
        "**PROMPT enviado al modelo:**\n",
        "```",
        s.get("prompt", "(vacío)")[:2000],  # Truncar para la memoria
        "```\n",
        "**RESPUESTA del modelo:**\n",
        "```",
        s.get("response", "(vacío)")[:2000],
        "```\n",
        f"**BLEU en este paso:** {s.get('bleu', {}).get('corpus_bleu', 'N/A')}\n",
        f"**Latencia:** {s.get('latency_s', '?')}s\n"
    ]
    return "\n".join(lines)


def generate_full_report(logs: list[dict], output_path: str = "reports/report.md"):
    """Genera el informe Markdown completo."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = [
        f"# Informe de Experimentos — Olimpiadas Lingüísticas con LLMs\n",
        f"_Generado automáticamente: {now}_\n",
        "---\n",
        summary_table(logs),
        "\n---\n",
        "## Evolución de BLEU por pasos\n"
    ]

    for log in logs:
        sections.append(bleu_progression(log))

    sections.append("\n---\n")
    sections.append("## Ejemplos de interacciones (para la memoria)\n")

    for log in logs:
        for strategy in ["baseline", "incremental_combinations", "step_by_step"]:
            sections.append(example_prompt_response(log, strategy, step=0))

    sections.append("\n---\n")
    sections.append("## Metadatos de los experimentos\n")
    for log in logs:
        meta = log.get("metadata", {})
        sections.append(
            f"- **{meta.get('puzzle_id')}** | {meta.get('language')} | "
            f"{meta.get('model_key')} | "
            f"{meta.get('num_train_pairs')} frases entrenamiento | "
            f"{meta.get('num_test_pairs')} frases test | "
            f"{meta.get('timestamp', '')[:19]}\n"
        )

    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))

    print(f"✓ Informe generado: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(glob.glob("results/*.json"))

    if not paths:
        print("No se encontraron logs en results/. Ejecuta primero experiment_runner.py")
        sys.exit(1)

    logs = load_logs(paths)
    print(f"Cargados {len(logs)} logs.")
    generate_full_report(logs, output_path="reports/report.md")
