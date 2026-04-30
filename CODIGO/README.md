# TFG — Olimpiadas Lingüísticas con LLMs

**Objetivo:** Comparar estrategias de prompting para que modelos de lenguaje
resuelvan puzzles de olimpiadas lingüísticas en idiomas nunca vistos durante
el entrenamiento, usando únicamente razonamiento metalingüístico.

---

## Estructura del proyecto

```
ling_olympiad/
├── experiment_runner.py     # Motor principal: ejecuta las 3 estrategias
├── generate_report.py       # Genera informes Markdown desde los logs JSON
├── setup_and_test.py        # Verifica API key y dependencias
├── puzzles/
│   └── ayutla_mixe.json     # Puzzle de ejemplo (Ayutla Mixe, México)
├── results/                 # Logs JSON de cada experimento (auto-generados)
├── reports/                 # Informes Markdown (auto-generados)
└── logs/                    # Logs de ejecución
```

---

## Paso 1 — Obtener la API key de OpenRouter (2 minutos, gratis)

OpenRouter es un proxy que da acceso a TODOS los modelos con UNA SOLA key.

1. Ve a **https://openrouter.ai**
2. Haz clic en **Sign in** → regístrate con Google (sin tarjeta de crédito)
3. Menú izquierdo → **Keys** → **Create key** → ponle un nombre → copiar
4. La key empieza por `sk-or-v1-...`

Límites gratuitos: 50 peticiones/día (suficiente para los primeros experimentos).
Con $10 de crédito sube a 1000/día — opcional.

---

## Paso 2 — Configurar la key en tu terminal

```bash
# macOS / Linux
export OPENROUTER_API_KEY="sk-or-v1-TU_KEY_AQUI"

# Windows PowerShell
$env:OPENROUTER_API_KEY="sk-or-v1-TU_KEY_AQUI"
```

---

## Paso 3 — Instalar dependencias y verificar

```bash
pip install sacrebleu requests
python setup_and_test.py
```

Deberías ver:
```
✓ Llama 3.2 3B  (débil) → 'OK'
✓ Llama 3.3 70B (medio) → 'OK'
✓ Llama 3.1 405B(fuerte)→ 'OK'
✓ Todo listo. Ejecuta: python experiment_runner.py
```

---

## Paso 4 — Ejecutar experimentos

```bash
python experiment_runner.py
```

Ejecuta las **3 estrategias** × **3 modelos** sobre el puzzle de Ayutla Mixe.
Cada experimento genera un JSON en `results/` con prompts, respuestas y BLEU.

---

## Paso 5 — Generar informe para la memoria

```bash
python generate_report.py
```

Genera `reports/report.md` con tabla comparativa y ejemplos de interacción.

---

## Paso 6 — Añadir puzzles de Linguini

1. Descarga el dataset en https://github.com/facebookresearch/linguini
   - Contraseña del ZIP: `linguisticreasoning`
2. Convierte un puzzle al formato JSON (ver `puzzles/ayutla_mixe.json`)
3. Cambia `PUZZLE_PATH` en `experiment_runner.py` y vuelve a ejecutar

---

## Las 3 estrategias comparadas

| ID | Nombre | Descripción |
|----|--------|-------------|
| 0  | **Baseline** | Todas las frases de entrenamiento de golpe |
| 1  | **Incremental** | C(n,2)→C(n,3)→...→C(n,n): diccionario+reglas acumulativo (propuesta tutor) |
| 2  | **Step-by-step** | Léxico→fonología→morfo-sintaxis→sintaxis (Zhu et al., 2025) |

## Modelos (todos gratuitos vía OpenRouter)

| Tier | Modelo | ID en OpenRouter |
|------|--------|-----------------|
| Débil | Llama 3.2 3B | `meta-llama/llama-3.2-3b-instruct:free` |
| Medio | Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct:free` |
| Fuerte | Llama 3.1 405B | `meta-llama/llama-3.1-405b-instruct:free` |

## Referencias clave (Cap. 2 de la memoria)

- **Linguini**: Sánchez-Alastruey et al. (2024). Meta AI. https://github.com/facebookresearch/linguini
- **LingOly**: Bean et al. (2024). NeurIPS 2024. https://arxiv.org/abs/2406.06196
- **Zhu et al.**: Zhu, Liang, Xu & Xu (2025). LoResLM @ ACL 2025. https://aclanthology.org/2025.loreslm-1.31
- **modeLing**: Chi et al. (2024). SIGTYP 2024.
