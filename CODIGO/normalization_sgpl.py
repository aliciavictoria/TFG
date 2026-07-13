"""
compare_normalizacion.py
========================
Compara los valores de chrF++ ANTES y DESPUÉS de normalizar (sg)/(pl).
Genera un JSON con:
  - todos los valores celda a celda
  - delta (nuevo - viejo)
  - qué celdas cambian y cuánto
  - nuevas medias y σ por metodología y modelo

Uso:
    python3 compare_normalizacion.py

Requiere:
    results_gpt_final/   results_gpt_norm/
    results_8b_final/    results_8b_norm/
    results_70b_final/   results_70b_sambanova/   results_70b_norm/
"""

import json, math
from pathlib import Path

LABELS = {
    "012018020100": "Hakhun",
    "012022030100": "Ngemba",
    "012019010100": "Yonggom",
    "012015040100": "Warlpiri",
    "012023020100": "Apurinã",
    "012023030100": "Coastal Marind",
    "012012010200": "Dyirbal",
    "012016030100": "Kunuz Nubian",
    "012008050100": "Inuktitut",
    "012005010100": "Tzeltal",
    "012006010100": "Lakota",
}
ORDER = [
    "Hakhun", "Ngemba", "Yonggom", "Warlpiri", "Apurinã",
    "Coastal Marind", "Dyirbal", "Kunuz Nubian", "Inuktitut",
    "Tzeltal", "Lakota",
]
STRATS = [
    ("baseline",                "M1"),
    ("incremental_combinations","M2"),
    ("step_by_step",            "M3"),
    ("human_inspiration",       "M4"),
    ("self_correction",         "M5"),
]

def load(folder):
    d = {}
    for fp in Path(folder).glob("*_rescored.json"):
        r = json.load(open(fp))
        pid = r["metadata"]["puzzle_id"].replace("linguini_", "")
        lang = LABELS.get(pid, pid)
        d[lang] = r
    return d

def pv(data, lang, strat):
    """chrF++ del paso principal. None si ausente o excluido."""
    if lang not in data:
        return None
    if lang == "Hakhun" and strat == "human_inspiration":
        return None
    r = data[lang]["results"].get(strat)
    if not r or "error" in r:
        return None
    m = r.get("primary_metrics", {})
    v = m.get("corpus_chrfpp", 0)
    return round(v, 2) if v > 0 else None

def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    mu = sum(vals) / len(vals)
    sig = math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))
    return round(mu, 2), round(sig, 2), len(vals)

# ── Cargar resultados ─────────────────────────────────────────────────────────
old = {
    "GPT": load("results_gpt_final"),
    "8B":  load("results_8b_final"),
    "70B": {**load("results_70b_final"), **load("results_70b_sambanova")},
}
new = {
    "GPT": load("results_gpt_norm"),
    "8B":  load("results_8b_norm"),
    "70B": load("results_70b_norm"),
}
models = ["8B", "70B", "GPT"]

# ── Construir tabla completa ──────────────────────────────────────────────────
table = {}          # table[lang][strat][model] = {old, new, delta}
changes = []        # celdas que cambian (|delta| >= 0.1)

for lang in ORDER:
    table[lang] = {}
    for strat, label in STRATS:
        table[lang][label] = {}
        for model in models:
            o = pv(old[model], lang, strat)
            n = pv(new[model], lang, strat)
            delta = round(n - o, 2) if (o is not None and n is not None) else None
            table[lang][label][model] = {
                "antes":  o,
                "despues": n,
                "delta":  delta,
            }
            if delta is not None and abs(delta) >= 0.1:
                changes.append({
                    "lengua":      lang,
                    "metodologia": label,
                    "modelo":      model,
                    "antes":       o,
                    "despues":     n,
                    "delta":       delta,
                })

# Ordenar cambios de mayor a menor Δ absoluto
changes.sort(key=lambda x: -abs(x["delta"]))

# ── Medias por metodología ────────────────────────────────────────────────────
means = {}
for strat, label in STRATS:
    means[label] = {}
    for model in models:
        excl = ["Hakhun"] if strat == "human_inspiration" else []
        old_vals = [pv(old[model], l, strat) for l in ORDER if l not in excl]
        new_vals = [pv(new[model], l, strat) for l in ORDER if l not in excl]
        mu_o, sig_o, n_o = stats(old_vals)
        mu_n, sig_n, n_n = stats(new_vals)
        means[label][model] = {
            "antes":  {"mu": mu_o, "sigma": sig_o, "n": n_o},
            "despues": {"mu": mu_n, "sigma": sig_n, "n": n_n},
            "delta_mu": round(mu_n - mu_o, 2) if (mu_o and mu_n) else None,
        }

# ── Guardar ──────────────────────────────────────────────────────────────────
output = {
    "resumen": {
        "total_celdas_evaluadas": sum(
            1 for lang in ORDER
            for _, label in STRATS
            for model in models
            if table[lang][label][model]["antes"] is not None
        ),
        "celdas_que_cambian_>=0.1": len(changes),
        "celdas_que_cambian_>=1.0": sum(1 for c in changes if abs(c["delta"]) >= 1.0),
        "celdas_que_cambian_>=5.0": sum(1 for c in changes if abs(c["delta"]) >= 5.0),
    },
    "cambios_ordenados_por_delta": changes,
    "tabla_completa": table,
    "nuevas_medias": means,
}

fname = "comparacion_normalizacion.json"
with open(fname, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# ── Imprimir resumen ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"NORMALIZACIÓN (sg)/(pl) — RESUMEN")
print(f"{'='*60}")
print(f"Celdas evaluadas:        {output['resumen']['total_celdas_evaluadas']}")
print(f"Cambian ≥ 0.1 puntos:   {output['resumen']['celdas_que_cambian_>=0.1']}")
print(f"Cambian ≥ 1.0 puntos:   {output['resumen']['celdas_que_cambian_>=1.0']}")
print(f"Cambian ≥ 5.0 puntos:   {output['resumen']['celdas_que_cambian_>=5.0']}")

print(f"\n--- CAMBIOS ≥ 1.0 punto (de mayor a menor Δ) ---")
for c in changes:
    if abs(c["delta"]) >= 1.0:
        signo = "↑" if c["delta"] > 0 else "↓"
        print(f"  {signo} {c['lengua']:16s} {c['metodologia']} {c['modelo']:3s}: "
              f"{c['antes']:.1f} → {c['despues']:.1f}  (Δ={c['delta']:+.1f})")

print(f"\n--- NUEVAS MEDIAS (antes → después) ---")
for label in ["M1","M2","M3","M4","M5"]:
    for model in models:
        m = means[label][model]
        o = m["antes"]; n = m["despues"]
        if o["mu"] and n["mu"]:
            print(f"  {label} {model}: μ {o['mu']:.1f}±{o['sigma']:.1f} → "
                  f"{n['mu']:.1f}±{n['sigma']:.1f}  (Δ={m['delta_mu']:+.1f})")

print(f"\n✓ JSON completo guardado en: {fname}")