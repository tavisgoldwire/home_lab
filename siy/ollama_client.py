"""
ollama_client.py — Talks to Ollama's /api/chat endpoint.
══════════════════════════════════════════════════════════════════════════════

THIS MODULE DOES ONE THING: send messages to Ollama and get responses back.
It's deliberately thin — no business logic, no memory, no session tracking.
Those live in other modules. This is just the "mouth and ears" to the LLM.

WHY /api/chat INSTEAD OF /api/generate:
  The old Windows Siy used /api/generate, which takes a single prompt string.
  That meant we had to manually stitch together the conversation history into
  one big blob of text. It worked, but poorly — the model couldn't reliably
  tell who said what, and multi-turn conversations were fragile.

  /api/chat takes a LIST of messages, each with a role:
    - "system"    → instructions (who Siy is, what it knows, rules)
    - "user"      → what Tavis said
    - "assistant" → what Siy said previously
    - "tool"      → results from tool calls (file contents, HA state, etc.)

  The model sees structured turn-by-turn conversation. This is how ChatGPT,
  Claude, and every modern chat model works under the hood.

HOW TOOL CALLING WORKS:
  When we send tool definitions along with messages, the model can choose
  to call a tool instead of (or before) giving a text response. The flow:

    1. We send: messages + tool definitions
    2. Model responds with: {"tool_calls": [{"function": {"name": "...", "arguments": {...}}}]}
    3. We execute the tool, get a result string
    4. We append the result as a "tool" message
    5. We send everything back to the model
    6. Model either calls another tool or gives a text response

  This loop is handled by app.py's run_chat_with_tools(). This module
  just does step 1 and returns whatever the model says.

NETWORK NOTE:
  Ollama runs on this same machine (OptiPlex #2, 192.168.8.212) as a
  systemd service. The URL is http://127.0.0.1:11434 (localhost).
  If you ever move Ollama to your gaming PC for GPU acceleration,
  change OLLAMA_BASE in config.py — this file doesn't need to change.
"""

import requests
import logging

from config import OLLAMA_CHAT_URL, MODEL_NAME, OLLAMA_TIMEOUT, OLLAMA_BASE

# ── Logger ──────────────────────────────────────────────────────────
# Uses the "siy.ollama" namespace so log lines are easy to filter.
# Example output:  14:32:05 [siy.ollama] INFO: Ollama OK. Model 'qwen3:8b' available.
logger = logging.getLogger("siy.ollama")


# ═══════════════════════════════════════════════════════════════════════
# HEALTH CHECK — called once on startup by app.py's lifespan handler
# ═══════════════════════════════════════════════════════════════════════

def check_ollama() -> bool:
    """
    Verify that Ollama is running and the configured model is available.

    Called during server startup so you get a clear, immediate error
    instead of a mystery crash on the first chat request 10 minutes later.

    WHAT IT CHECKS:
      1. Can we reach Ollama at all? (ConnectionError = Ollama not running)
      2. Is our model pulled? (checks /api/tags for the model name)

    RETURNS:
      True  → Ollama is up and the model is ready
      False → something is wrong (logged with details)

    WHY THE MODEL NAME MATCHING IS FUZZY:
      Ollama's /api/tags returns model names like "qwen3:8b" but sometimes
      includes the ":latest" suffix. We also check if the base name
      (before the colon) matches any available model. This handles cases
      like MODEL_NAME="qwen3:8b" matching "qwen3:8b-q4_0" in the list.
    """
    try:
        # ── Step 1: Can we reach Ollama? ────────────────────────────
        # /api/tags returns a JSON list of all pulled models.
        # We use a short timeout (5 sec) because this is just a ping.
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()

        # ── Step 2: Is our model available? ─────────────────────────
        # Extract just the model names from the response.
        # Response format: {"models": [{"name": "qwen3:8b", ...}, ...]}
        models = [m["name"] for m in r.json().get("models", [])]

        # Direct match: "qwen3:8b" in ["qwen3:8b", "phi3:mini", ...]
        if MODEL_NAME in models:
            logger.info(f"Ollama OK. Model '{MODEL_NAME}' available.")
            return True

        # Fuzzy match: "qwen3:8b" might appear as "qwen3:8b:latest"
        if f"{MODEL_NAME}:latest" in models:
            logger.info(f"Ollama OK. Model '{MODEL_NAME}' available (as :latest).")
            return True

        # Base-name match: "qwen3" in "qwen3:8b-q4_0"
        # This catches variant models you might have pulled.
        base_name = MODEL_NAME.split(":")[0]
        matching = [m for m in models if base_name in m]
        if matching:
            logger.info(
                f"Ollama OK. Exact model '{MODEL_NAME}' not found, "
                f"but similar models available: {matching}"
            )
            return True

        # Nothing matched at all
        logger.warning(
            f"Model '{MODEL_NAME}' not found in Ollama. "
            f"Available models: {models}. "
            f"Run: ollama pull {MODEL_NAME}"
        )
        return False

    except requests.ConnectionError:
        # Ollama service isn't running or isn't listening on the expected port
        logger.error(
            f"Cannot reach Ollama at {OLLAMA_BASE}. "
            f"Is the Ollama service running? Check with: systemctl status ollama"
        )
        return False

    except Exception as e:
        # Catch-all for unexpected errors (DNS issues, malformed JSON, etc.)
        logger.error(f"Ollama health check failed unexpectedly: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# CHAT — the main function that sends messages to the LLM
# ═══════════════════════════════════════════════════════════════════════

def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """
    Send a conversation to Ollama's /api/chat and return the response.

    This is the core function that every chat request flows through.
    It's called by app.py's run_chat_with_tools() in a loop.

    ARGS:
      messages: Ordered list of conversation messages. Each is a dict:
                {"role": "system"|"user"|"assistant"|"tool", "content": "..."}

                Example for a 3-turn conversation:
                [
                  {"role": "system",    "content": "You are Siy..."},
                  {"role": "user",      "content": "turn on the bedroom light"},
                  {"role": "assistant", "content": "Done, bedroom light is on."},
                  {"role": "user",      "content": "thanks, now dim it to 50%"},
                ]

      tools:    Optional list of tool definitions for function calling.
                When provided, the model can choose to call a tool instead
                of giving a text response. Tool definitions follow Ollama's
                format (similar to OpenAI's function calling spec).

                Example tool definition:
                {
                  "type": "function",
                  "function": {
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                      "type": "object",
                      "properties": {
                        "path": {"type": "string", "description": "Directory path"}
                      },
                      "required": ["path"]
                    }
                  }
                }

    RETURNS:
      The full Ollama response dict. Two possible shapes:

      TEXT RESPONSE (model answered directly):
        {
          "message": {
            "role": "assistant",
            "content": "The bedroom light is now at 50%."
          },
          "eval_count": 42,           ← tokens generated
          "prompt_eval_count": 350    ← tokens in prompt
        }

      TOOL CALL (model wants to use a tool):
        {
          "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
              {
                "function": {
                  "name": "ha_call_service",
                  "arguments": {"entity_id": "light.bedroom", "brightness": 128}
                }
              }
            ]
          }
        }

    RAISES:
      requests.ConnectionError  → Ollama is down
      requests.Timeout          → model took too long (> OLLAMA_TIMEOUT)
      requests.HTTPError        → Ollama returned an error status code
    """

    # ── Build the request payload ───────────────────────────────────
    # "stream": False means we wait for the complete response.
    # Streaming (Phase 4 feature) would let us show tokens as they
    # generate, but for now a complete response is simpler to handle
    # and debug.
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    }

    # Only include the tools key if we actually have tool definitions.
    # Ollama handles an empty list fine, but being explicit is cleaner
    # and avoids sending unnecessary data over the wire.
    if tools:
        payload["tools"] = tools

    logger.debug(f"Sending {len(messages)} messages to Ollama ({MODEL_NAME})")

    # ── Send to Ollama ──────────────────────────────────────────────
    # OLLAMA_TIMEOUT is set to 180 seconds in config.py.
    # On the i5-6500T (CPU-only), qwen3:8b can take 15-40 seconds per
    # response. Complex prompts with lots of context might push toward
    # the higher end. The 180s timeout gives headroom for tool loops
    # where the model might generate multiple responses in sequence.
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=OLLAMA_TIMEOUT)

    # raise_for_status() throws an HTTPError if Ollama returned 4xx/5xx.
    # Common causes:
    #   400 → malformed request (bad message format)
    #   404 → model not found (need to `ollama pull`)
    #   500 → Ollama internal error (usually OOM on large contexts)
    r.raise_for_status()

    result = r.json()

    # ── Log token usage (if available) ──────────────────────────────
    # Ollama includes token counts in the response. This is useful for:
    #   - Debugging slow responses (high prompt_eval_count = big context)
    #   - Monitoring context window usage (qwen3:8b max is 32K tokens)
    #   - Spotting runaway memory injection (episodic recall too aggressive)
    #
    # These fields aren't always present (e.g., on tool-call responses),
    # so we use .get() with fallback.
    prompt_tokens = result.get("prompt_eval_count", "?")
    gen_tokens = result.get("eval_count", "?")
    if prompt_tokens != "?" or gen_tokens != "?":
        logger.debug(f"Tokens — prompt: {prompt_tokens}, generated: {gen_tokens}")

    return result
