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
            }
        )

        return response.choices[0].message.content, "groq"

    except Exception as e:

        print(f"Groq LLM failed: {e}")
        print("Falling back to local Ollama...")


    # --------------------------------------
    # 2. OLLAMA
    # --------------------------------------

    try:

        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            format="json",
            options={
                "temperature": temperature
            }
        )

        content = response["message"]["content"]

        return content, "ollama_fallback"

    except Exception as e:

        raise RuntimeError(
            f"Both Groq and Ollama failed. "
            f"Ollama error: {e}"
        )