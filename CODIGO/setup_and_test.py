"""
setup_and_test.py — Verifica API keys de Groq, Cerebras y OpenRouter.

Configurar antes de ejecutar:
  export GROQ_API_KEY='gsk_...'
  export OPENROUTER_API_KEY='sk-or-...'
  export CEREBRAS_API_KEY='csk_...'  (opcional)
"""

import os, sys, subprocess, requests

def install_deps():
    for dep in ["sacrebleu", "requests"]:
        r = subprocess.run([sys.executable, "-m", "pip", "install", dep, "-q",
                            "--break-system-packages"], capture_output=True)
        if r.returncode != 0:
            subprocess.run([sys.executable, "-m", "pip", "install", dep, "-q"], capture_output=True)
    print("✓ Dependencias listas\n")

def test_api(url, key, model_id, label, extra_headers={}):
    try:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        headers.update(extra_headers)
        r = requests.post(url,
            headers=headers,
            json={"model": model_id,
                  "messages": [{"role": "user", "content": "Reply with only the word OK. Do not write anything else."}],
                  "max_tokens": 500, "temperature": 0.1},
            timeout=60)
        r.raise_for_status()
        d = r.json()
        if "error" in d:
            print(f"  ✗ {label}: {d['error']}")
            return False
        msg = d["choices"][0]["message"]
        ans = msg.get("content") or msg.get("reasoning") or ""
        ans = ans.strip()
        if ans:
            print(f"  ✓ {label} → '{ans[:50]}'")
        else:
            print(f"  ✓ {label} → (conectado pero sin respuesta visible)")
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

    groq_key       = os.getenv("GROQ_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    cerebras_key   = os.getenv("CEREBRAS_API_KEY", "")

    print("Estado de las API keys:")
    print(f"  {'✓' if groq_key else '✗'} GROQ_API_KEY:       {groq_key[:14]+'...' if groq_key else 'no configurada'}")
    print(f"  {'✓' if openrouter_key else '✗'} OPENROUTER_API_KEY: {openrouter_key[:14]+'...' if openrouter_key else 'no configurada'}")
    print(f"  {'✓' if cerebras_key else '✗'} CEREBRAS_API_KEY:   {cerebras_key[:14]+'...' if cerebras_key else 'no configurada (opcional)'}")

    if not groq_key:
        print("\nNecesitas al menos GROQ_API_KEY para continuar.")
        sys.exit(1)

    print("\nProbando conectividad:")
    any_ok = False
    groq_url = "https://api.groq.com/openai/v1/chat/completions"

    ok1 = test_api(groq_url, groq_key, "llama-3.1-8b-instant",
                   "Groq       / Llama 3.1 8B   (weak)")
    ok2 = test_api(groq_url, groq_key, "llama-3.3-70b-versatile",
                   "Groq       / Llama 3.3 70B  (medium)")
    any_ok = ok1 or ok2

    if openrouter_key:
        or_url = "https://openrouter.ai/api/v1/chat/completions"
        or_headers = {"HTTP-Referer": "https://tfg-linguistic-olympiad", "X-Title": "TFG"}
        ok3 = test_api(or_url, openrouter_key, "openai/gpt-oss-120b:free",
                       "OpenRouter / GPT OSS 120B   (strong)", or_headers)
        any_ok = any_ok or ok3
    else:
        print("  — OpenRouter / GPT OSS 120B   (strong) → sin key")

    if cerebras_key:
        cerebras_url = "https://api.cerebras.ai/v1/chat/completions"
        ok4 = test_api(cerebras_url, cerebras_key, "gpt-oss-120b",
                       "Cerebras   / GPT OSS 120B   (backup)", {})
        any_ok = any_ok or ok4

    if any_ok:
        print("\n✓ Todo listo. Ejecuta: python3 experiment_runner.py")
    else:
        print("\n✗ Ningún modelo respondió correctamente.")