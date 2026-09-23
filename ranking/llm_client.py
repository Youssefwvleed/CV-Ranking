import os

from dotenv import load_dotenv
from groq import Groq
import ollama


load_dotenv()


# ==========================================
# GROQ
# ==========================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

groq_client = Groq(
    api_key=GROQ_API_KEY
)

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)
def build_reasoning_kwargs():
    """
    reasoning_effort / include_reasoning only apply to GPT-OSS
    (reasoning) models. Keep them scoped here so switching
    GROQ_MODEL to a non-reasoning model (like llama-3.3-70b)
    doesn't send parameters it doesn't understand.
    """

    if "gpt-oss" in MODEL_NAME:
        return {
            "reasoning_effort": "low",
            "include_reasoning": False
        }

    return {}


# ==========================================
# OLLAMA
# ==========================================

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.1"
)

# Context window size sent to Ollama. Must be set explicitly —
# otherwise Ollama silently truncates large prompts using its
# default (small) context, which causes empty/inconsistent output.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

ollama_client = ollama.Client(
    host=OLLAMA_HOST
)


# ==========================================
# UNIFIED CHAT
# ==========================================

def chat_completion(
    messages: list[dict],
    temperature: float = 0
):
    """
    Try Groq first.

    If Groq fails for any reason, use the local
    Ollama model as a fallback.

    Both models receive the exact same messages.
    """

    # --------------------------------------
    # 1. GROQ
    # --------------------------------------

    try:

        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_object"
            },
            **build_reasoning_kwargs()
        )

        return response.choices[0].message.content, "groq"

    except Exception as e:

        print(f"Groq LLM failed: {e}")
        print("Falling back to local Ollama...")


    # --------------------------------------
    # 2. OLLAMA
    # --------------------------------------

        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            format="json",
            options={
                "temperature": temperature,
                "num_ctx": OLLAMA_NUM_CTX
            }
        )

        content = response["message"]["content"]

        return content, "ollama_fallback"

    except Exception as e:

        raise RuntimeError(
            f"Both Groq and Ollama failed. "
            f"Ollama error: {e}"
        )