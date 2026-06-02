"""
setup_and_test.py — Verifica API keys de Groq y/o Cerebras.

Registro gratuito (sin tarjeta):
  Groq:     https://console.groq.com     → API Keys → Create
  Cerebras: https://cloud.cerebras.ai    → API Keys → Create

Configurar antes de ejecutar:
  export GROQ_API_KEY='gsk_...'
  export CEREBRAS_API_KEY='csk_...'
"""

import os, sys, subprocess, requests

def install_deps():
    for dep in ["sacrebleu", "requests"]:
        r = subprocess.run([sys.executable, "-m", "pip", "install", dep, "-q",
                            "--break-system-packages"], capture_output=True)
        if r.returncode != 0:
            subprocess.run([sys.executable, "-m", "pip", "install", dep, "-q"], capture_output=True)
    print("✓ Dependencias listas\n")

def test_api(provider, url, key, model_id, label):
    try:
        r = requests.post(url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model_id,
                  "messages": [{"role": "user", "content": "Reply with only the word OK."}],
                  "max_tokens": 10, "temperature": 0},
            timeout=30)
        r.raise_for_status()
        d = r.json()
        if "error" in d:
            print(f"  ✗ {label}: {d['error']}")
            return False
        ans = d["choices"][0]["message"]["content"].strip()
        print(f"  ✓ {label} → '{ans}'")
        return True
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ {label}: HTTP {e.response.status_code} — {e.response.text[:120]}")
        return False
    except Exception as e:
        print(f"  ✗ {label}: {e}")
        return False

if __name__ == "__main__":
    print("=" * 55)
    print("SETUP — TFG Olimpiadas Lingüísticas con LLMs")
    print("=" * 55 + "\n")

    install_deps()

    groq_key     = os.getenv("GROQ_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    cerebras_key = os.getenv("CEREBRAS_API_KEY", "")
    any_ok = False

    print("Estado de las API keys:")
    if groq_key:
        print(f"  ✓ GROQ_API_KEY encontrada:     {groq_key[:14]}...")
    else:
        print("  ✗ GROQ_API_KEY no configurada")

    if openrouter_key:
        print(f"  ✓ OPENROUTER_API_KEY encontrada: {openrouter_key[:14]}...")
    else:
        print("  ✗ OPENROUTER_API_KEY no configurada")

    if cerebras_key:
        print(f"  ✓ CEREBRAS_API_KEY encontrada: {cerebras_key[:14]}...")
    else:
        print("  ✗ CEREBRAS_API_KEY no configurada (opcional)")

    if not groq_key and not google_key and not cerebras_key:
        print("""
Necesitas al menos una key. Registro gratuito (sin tarjeta):

  GROQ (14.400 req/día):
    1. https://console.groq.com → Sign in → API Keys → Create API Key
    2. export GROQ_API_KEY='gsk_...'

  GOOGLE AI STUDIO (1.500 req/día):
    1. https://aistudio.google.com → Get API Key → Create API Key
    2. export GOOGLE_API_KEY='AIza...'

Luego vuelve a ejecutar: python3 setup_and_test.py
""")
        sys.exit(1)

    print("\nProbando conectividad:")
    if groq_key:
        ok1 = test_api("groq",
            "https://api.groq.com/openai/v1/chat/completions",
            groq_key, "llama-3.1-8b-instant", "Groq / Llama 3.1 8B  (débil)")
        ok2 = test_api("groq",
            "https://api.groq.com/openai/v1/chat/completions",
            groq_key, "llama-3.3-70b-versatile", "Groq / Llama 3.3 70B (medio)")
        any_ok = any_ok or ok1 or ok2

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_key:
        ok3 = test_api("openrouter",
            "https://openrouter.ai/api/v1/chat/completions",
            openrouter_key, "google/gemma-4-26b-a4b-it:free",
            "OpenRouter / Gemma 4 26B MoE  (fuerte/Google)")
        ok4 = test_api("openrouter",
            "https://openrouter.ai/api/v1/chat/completions",
            openrouter_key, "openai/gpt-oss-120b:free",
            "OpenRouter / GPT OSS 120B     (fuerte/OpenAI)")
        any_ok = any_ok or ok3 or ok4

    if cerebras_key:
        ok4 = test_api("cerebras",
            "https://api.cerebras.ai/v1/chat/completions",
            cerebras_key, "llama-3.3-70b", "Cerebras / Llama 3.3 70B")
        any_ok = any_ok or ok4

    if any_ok:
        print("\n✓ Todo listo. Ejecuta: python3 experiment_runner.py")
    else:
        print("\n✗ Ningún modelo respondió correctamente.")
        print("  Comprueba tus keys en console.groq.com o cloud.cerebras.ai")