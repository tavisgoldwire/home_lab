"""
app.py — Siy Brain Server (SIY-01 on OptiPlex #2)
══════════════════════════════════════════════════════════════════════════════

THIS IS THE MAIN SERVER. Everything flows through here.

WHAT IT DOES:
  1. Receives messages from Home Assistant via HTTP POST to /chat
  2. Manages multi-turn conversations (sessions)
  3. Injects core memory + episodic memory into every request
  4. Sends conversation to Ollama with tool definitions
  5. If the model wants to use a tool, executes it and loops back
  6. Returns the final text response
  7. Archives expired sessions to episodic memory

ENDPOINTS:
  POST /chat          → main chat endpoint (HA sends messages here)
  GET  /health        → health check (Uptime Kuma watches this)
  GET  /memory/core   → view current core memory
  PUT  /memory/core   → update core memory (for future UI)
  GET  /sessions      → list active sessions (for debugging)

HOW TO RUN (two options):

  OPTION A — Manual (for development/testing):
    cd ~/siy
    source .venv/bin/activate
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

  OPTION B — systemd service (for production):
    sudo systemctl start siy
    sudo systemctl status siy
    journalctl -u siy -f    ← live log tail

  The systemd unit file is documented in SETUP.md.

NETWORK CONTEXT:
  This server listens on 0.0.0.0:8000.
  Home Assistant (on this same box at :8123) sends POST requests here.
  Uptime Kuma (on RPi 3) pings GET /health to confirm Siy is alive.
  Ollama (on this same box at :11434) does the actual LLM inference.

  Request flow:
    HA → POST http://127.0.0.1:8000/chat → app.py → Ollama → response → HA

WHAT CHANGED FROM WINDOWS:
  - Port 8123 → 8000 (HA occupies 8123 on this machine)
  - All paths are Linux (/home/siy_brain/siy/...)
  - SERVER_HOST and SERVER_PORT come from config.py
  - The __main__ block uses config constants (not hardcoded values)
  - Added /no_think stripping for qwen3's thinking tags
══════════════════════════════════════════════════════════════════════════════
"""

import json
import re
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Our modules ─────────────────────────────────────────────────────
from config import MODEL_NAME, SERVER_HOST, SERVER_PORT, MAX_TOOL_ROUNDS
from ollama_client import chat as ollama_chat, check_ollama
from session_manager import SessionManager
from memory import (
    EpisodicMemory,
    load_core_memory,
    save_core_memory,
    format_core_for_prompt,
    summarize_session_for_storage,
)
from tools import (
    ALL_TOOL_DEFINITIONS, TOOL_FUNCTIONS,
    HOME_TOOL_DEFINITIONS, HOME_TOOL_FUNCTIONS,
    FILE_TOOL_DEFINITIONS, FILE_TOOL_FUNCTIONS,
)
import router

# ── HA state pre-fetch — used by home agent to skip tool calling ─────

def fetch_ha_state_snapshot() -> str | None:
    """
    Fetch current state of all lights, switches, and sensors from HA.
    Returns a compact formatted string injected into the home agent's system
    prompt so the model can answer status questions without calling any tools.

    WHY THIS EXISTS:
      The home agent previously needed 2-3 LLM tool-calling rounds to answer
      "what's the status of my lights?" (list_entities → get_state → respond).
      On qwen3 CPU-only, each round is 15-40 sec → total timeout.
      Pre-fetching state server-side costs ~10ms and eliminates all those rounds.

    GRACEFUL DEGRADATION:
      Returns None if HA is unreachable. build_system_prompt() silently skips
      the state section when None — the home agent still works, just slower
      (falls back to tool-calling behavior).
    """
    from config import HA_URL, HA_TOKEN
    if not HA_TOKEN:
        return None
    try:
        import requests as _requests
        r = _requests.get(
            f"{HA_URL}/api/states",
            headers={"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"},
            timeout=10,
        )
        if not r.ok:
            logger.warning(f"HA state pre-fetch failed: HTTP {r.status_code}")
            return None

        states = r.json()
        useful_domains = {"light", "switch"}
        filtered = [s for s in states if s.get("entity_id", "").split(".")[0] in useful_domains]

        if not filtered:
            return None

        lines = []
        for s in sorted(filtered, key=lambda x: x.get("entity_id", "")):
            eid = s.get("entity_id", "unknown")
            state = s.get("state", "?")
            attrs = s.get("attributes", {})
            name = attrs.get("friendly_name", eid)

            extras = []
            if "brightness" in attrs and attrs["brightness"] is not None:
                extras.append(f"{round(attrs['brightness'] / 255 * 100)}% brightness")
            if "color_temp" in attrs and attrs["color_temp"] is not None:
                extras.append(f"{attrs['color_temp']} mireds")
            if "current_power_w" in attrs:
                extras.append(f"{attrs['current_power_w']}W")

            suffix = f", {', '.join(extras)}" if extras else ""
            lines.append(f"  {name} ({eid}): {state}{suffix}")

        logger.info(f"Home state snapshot: {len(lines)} devices pre-fetched")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"HA state pre-fetch error: {type(e).__name__}: {e}")
        return None


# ── Ollama request lock ──────────────────────────────────────────────
# Ollama processes one request at a time on CPU. Without this lock,
# concurrent requests both hit Ollama simultaneously — one succeeds,
# the other times out after 180 seconds. The lock serializes requests
# so the second one waits instead of failing.
_ollama_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════
#
# All log lines include timestamp, module name, and level.
# When running under systemd, these go to the journal automatically
# (journalctl -u siy -f). When running manually, they print to stdout.
#
# Log levels:
#   DEBUG   → token counts, memory lookups, tool results (verbose)
#   INFO    → startup, session creation, tool execution (normal)
#   WARNING → Ollama unavailable, missing sessions (something's off)
#   ERROR   → crashes, connection failures (needs attention)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("siy.app")


# ═══════════════════════════════════════════════════════════════════════
# GLOBAL STATE — initialized during startup
# ═══════════════════════════════════════════════════════════════════════
#
# These are module-level variables set during the lifespan startup.
# They persist for the lifetime of the server process.
#
# sessions:  in-memory dict of active conversations
# episodic:  ChromaDB-backed long-term memory (initialized on startup)

sessions = SessionManager()
episodic: EpisodicMemory | None = None


# ═══════════════════════════════════════════════════════════════════════
# LIFESPAN — startup and shutdown hooks
# ═══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on server startup and shutdown.

    STARTUP:
      1. Check Ollama is reachable (warns if not — doesn't crash)
      2. Initialize episodic memory (loads embedding model + ChromaDB)
      3. Load core memory (creates default file if missing)

    SHUTDOWN:
      1. Archive any active sessions to episodic memory
      2. Log goodbye

    WHY NOT CRASH ON MISSING OLLAMA?
      During development, you might start the Siy server before Ollama
      is ready (especially on boot, where systemd services start in
      parallel). The warning lets you know, and the first chat request
      will give a clear error. This is better than Siy refusing to
      start entirely.
    """
    global episodic

    logger.info("=" * 60)
    logger.info("  Siy Brain (SIY-01) starting up...")
    logger.info(f"  Host: {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info("=" * 60)

    # ── Check Ollama ────────────────────────────────────────────────
    if not check_ollama():
        logger.warning(
            "⚠ Ollama not available — Siy will error on chat requests. "
            "Check: systemctl status ollama"
        )

    # ── Initialize episodic memory ──────────────────────────────────
    # This loads the embedding model (~4 sec on CPU) and opens ChromaDB.
    episodic = EpisodicMemory()

    # ── Load core memory ────────────────────────────────────────────
    # Creates /home/siy_brain/siy/memory/core_memory.json with defaults
    # if it doesn't exist yet.
    core = load_core_memory()
    logger.info(f"Core memory loaded: {list(core.keys())}")

    # ── Load entity cache for fast path ─────────────────────────────
    # Fetches HA lights + switches so "turn off the lamp" resolves
    # instantly without an LLM call. Gracefully skips if HA is down.
    router.load_entity_cache()

    logger.info("Siy Brain ready.")
    logger.info("=" * 60)

    yield  # ← server runs here, handling requests

    # ── Shutdown ────────────────────────────────────────────────────
    logger.info("Shutting down — archiving active sessions...")
    _archive_all_sessions()
    logger.info("Goodbye.")


app = FastAPI(title="Siy Brain", version="0.2.0", lifespan=lifespan)


# ═══════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════
#
# Pydantic models define the shape of incoming requests and outgoing
# responses. FastAPI uses these for:
#   - Automatic request validation (wrong types → 422 error)
#   - Automatic OpenAPI docs (http://192.168.8.212:8000/docs)
#   - Type hints for IDE autocompletion

class ChatRequest(BaseModel):
    """
    What HA sends to us.

    Fields:
      text:       The user's message (required, must not be empty)
      session_id: Optional session identifier. If provided, continues
                  that session. If omitted, creates a new session.

                  HA should use a fixed ID like "ha_main" so all
                  voice/text commands share one conversation.
    """
    text: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    """
    What we send back to HA.

    Fields:
      reply:      Siy's text response
      session_id: Which session this belongs to (useful if HA
                  didn't provide one and we auto-generated it)
    """
    reply: str
    session_id: str


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_system_prompt(
    core_memory: str,
    episodic_memories: list[str],
    agent_context: str | None = None,
    home_state: str | None = None,
) -> str:
    """
    Build the system message that tells Siy who it is and what it knows.

    This is the first message in every Ollama request. It sets identity,
    injects memory, and establishes behavioral rules.

    The prompt is assembled from parts rather than a giant f-string
    because it's easier to read, modify, and debug.

    WHY /no_think:
      qwen3:8b has a "thinking" mode where it wraps internal reasoning
      in <think>...</think> tags. This is useful for complex tasks but
      adds latency and clutters responses for a personal assistant.
      The /no_think directive tells qwen3 to skip the thinking step
      and respond directly. Remove it if you want reasoning traces
      (useful for debugging complex tool chains).

    ARGS:
      core_memory:       formatted string from format_core_for_prompt()
      episodic_memories: list of past conversation summaries (top-K)
      agent_context:     optional agent-specific tool description line
      home_state:        optional pre-fetched HA device state (home agent only)
    """
    # Agent context overrides the generic tool description line.
    # home agent → focused HA prompt; file agent → focused file prompt;
    # general → the default "you have access to tools" line.
    tool_line = (
        agent_context
        if agent_context
        else "You have access to tools for browsing files and controlling home devices."
    )

    parts = [
        # ── Identity ────────────────────────────────────────────────
        "You are Siy, Tavis's personal AI assistant running locally on his home network.",
        "You are practical, direct, and concise. Go deeper only when asked.",
        tool_line,
        "Use tools when helpful — don't guess at file paths or contents.",
        "/no_think",  # Suppress qwen3's <think> tags for faster responses
        "",
        # ── Core memory (always present) ────────────────────────────
        "── Core Memory (curated facts — trust these) ──",
        core_memory,
    ]

    # ── Home device state (home agent only) ─────────────────────────
    # Pre-fetched from HA at request time so the model can answer status
    # questions ("are my lights on?") by reading this section directly
    # instead of calling ha_list_entities / ha_get_state tools.
    # Tools remain available as a fallback for edge cases.
    if home_state:
        parts.append("")
        parts.append("── Current Device State (live from Home Assistant) ──")
        parts.append(home_state)

    # ── Episodic memories (if any are relevant) ─────────────────────
    if episodic_memories:
        parts.append("")
        parts.append("── Recent Memory (past conversations — may be noisy) ──")
        for mem in episodic_memories:
            # Truncate long memories to save context window space.
            # 300 chars is enough to capture topic + key details.
            parts.append(f"• {mem[:300]}")

    # ── Rules ───────────────────────────────────────────────────────
    parts.extend([
        "",
        "── Rules ──",
        "• If unsure, ask for clarification.",
        "• Never invent facts or file paths.",
        "• Be concise — Tavis prefers short answers unless he asks for depth.",
        "• Do NOT include <think> tags or internal reasoning in your response.",
    ])

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# RESPONSE CLEANING — strip qwen3 thinking artifacts
# ═══════════════════════════════════════════════════════════════════════

# Regex to match qwen3's <think>...</think> blocks (including multiline)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

# ── Agent-specific system prompt additions ───────────────────────────
# These replace the generic tool description line in build_system_prompt.
# Each agent knows its role, its tools, and its response style.

HOME_AGENT_CONTEXT = (
    "You are Siy's home control agent. Your job is controlling Tavis's "
    "apartment devices via Home Assistant. "
    "The current state of all devices is listed above under 'Current Device State' — "
    "use it directly to answer status questions without calling any tools. "
    "To control a device, call ha_call_service using the entity_id from the state list. "
    "Only call ha_list_entities if a device is missing from the state list. "
    "Be concise — 'Floor lamp is off.' or 'Done. Bedside lamp is now at 40%.' are the right length."
)

FILE_AGENT_CONTEXT = (
    "You are Siy's file agent. Your job is helping Tavis browse, search, "
    "and read files on the Siy server. "
    "Tools available: list_files (show directory contents), "
    "search_files (find files by name pattern), read_file (read text files). "
    "Always verify a path exists before reading. "
    "Summarize large files rather than dumping raw content."
)


def clean_response(text: str) -> str:
    """
    Remove any <think>...</think> blocks from the model's response.

    WHY:
      Even with /no_think in the system prompt, qwen3 sometimes still
      emits thinking tags, especially on complex multi-step reasoning.
      Rather than sending those raw to HA (where they'd look like
      broken HTML), we strip them out.

    Also strips leading/trailing whitespace that can accumulate
    after tag removal.
    """
    cleaned = THINK_PATTERN.sub("", text)
    return cleaned.strip()


# ═══════════════════════════════════════════════════════════════════════
# TOOL CALL LOOP
# ═══════════════════════════════════════════════════════════════════════

def _execute_local(name: str, arguments: dict, tool_funcs: dict) -> str:
    """
    Agent-scoped tool dispatcher. Unlike the global execute_tool(), this
    only looks up tools in the provided tool_funcs dict — so a home agent
    cannot accidentally call file tools and vice versa.
    """
    func = tool_funcs.get(name)
    if not func:
        logger.warning(f"Tool '{name}' not available in this agent's scope")
        return (
            f"Tool '{name}' is not available for this request type. "
            f"Available tools: {list(tool_funcs.keys())}"
        )
    try:
        logger.info(f"Executing tool: {name}({arguments})")
        return func(**arguments)
    except TypeError as e:
        logger.error(f"Tool '{name}' got wrong arguments: {e}")
        return f"Error calling {name}: wrong arguments — {e}"
    except Exception as e:
        logger.error(f"Tool '{name}' failed: {e}", exc_info=True)
        return f"Error running {name}: {type(e).__name__}: {e}"


def run_chat_with_tools(
    messages: list[dict],
    tool_defs: list[dict] | None = None,
    tool_funcs: dict | None = None,
) -> str:
    """
    Send messages to Ollama. If the model calls tools, execute them
    and feed results back. Repeat until the model gives a text response
    or we hit the safety limit.

    THE LOOP:
      ┌────────────────────────────────────────────────────────────┐
      │  1. Send messages + tool definitions to Ollama             │
      │  2. Response has tool_calls?                               │
      │     YES → execute each tool, append results, go to 1      │
      │     NO  → return the text response, we're done            │
      │  3. Hit MAX_TOOL_ROUNDS? → return whatever we have        │
      └────────────────────────────────────────────────────────────┘

    WHY A LOOP?
      Some tasks require multiple tool calls. Example:
        User: "Find CSV files in my documents and read the first one"
        Round 1: LLM calls search_files → finds 3 CSVs
        Round 2: LLM calls read_file on the first CSV → gets content
        Round 3: LLM generates a text summary of the file

    MAX_TOOL_ROUNDS (5) prevents infinite loops if the model gets
    confused and keeps calling tools without producing a text response.

    ARGS:
      messages: full conversation (system + history + new message)

    RETURNS:
      The model's final text response (cleaned of <think> tags)
    """
    # Default to full tool set if not specified (backward compatible)
    if tool_defs is None:
        tool_defs = ALL_TOOL_DEFINITIONS
    if tool_funcs is None:
        tool_funcs = TOOL_FUNCTIONS

    msg = {}  # Will hold the last response from Ollama

    for round_num in range(MAX_TOOL_ROUNDS):
        # ── Send to Ollama ──────────────────────────────────────────
        response = ollama_chat(messages, tools=tool_defs)
        msg = response.get("message", {})

        tool_calls = msg.get("tool_calls")

        if not tool_calls:
            # No tool calls — model gave a text response. We're done.
            return clean_response(msg.get("content", ""))

        # ── Model wants to use tools ────────────────────────────────
        logger.info(f"Tool round {round_num + 1}: {len(tool_calls)} call(s)")

        # Add the assistant's tool-call message to history.
        # This is important: the model needs to see its own tool
        # request in the conversation to make sense of the tool result.
        messages.append(msg)

        for call in tool_calls:
            func_info = call.get("function", {})
            tool_name = func_info.get("name", "unknown")
            tool_args = func_info.get("arguments", {})

            # Execute via agent-scoped dispatcher (respects tool subset)
            result = _execute_local(tool_name, tool_args, tool_funcs)

            # Add tool result to messages. The model reads this on the
            # next round and either calls another tool or responds.
            messages.append({
                "role": "tool",
                "content": result,
            })

    # ── Safety: hit max rounds ──────────────────────────────────────
    logger.warning(f"Hit max tool rounds ({MAX_TOOL_ROUNDS})")
    fallback = msg.get("content", "")
    if fallback:
        return clean_response(fallback)
    return "I got stuck in a tool loop. Could you rephrase your request?"


# ═══════════════════════════════════════════════════════════════════════
# MAIN CHAT ENDPOINT
# ═══════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Main endpoint — HA sends messages here.

    FLOW:
      1. Validate input
      2. Get or create session
      3. Add user message to history
      4. Fast path check — simple HA commands bypass the LLM entirely
      5. Classify intent → select agent (home / file / general)
      6. Load core + episodic memory
      7. Build agent-specific system prompt
      8. Acquire Ollama lock (one request at a time)
      9. Run LLM with agent's tool subset
      10. Save response, archive expired sessions, return

    FAST PATH (step 4):
      "turn off the floor lamp" → resolves entity, calls HA, returns in ~1 sec.
      No Ollama call, no lock needed.

    OLLAMA LOCK (step 8):
      Serializes LLM calls. If two requests arrive simultaneously, the
      second waits instead of timing out. The fast path bypasses this lock.
    """
    # ── 1. Validate ─────────────────────────────────────────────────
    user_text = req.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message")

    # ── 2. Session ──────────────────────────────────────────────────
    session = sessions.get_or_create(req.session_id)

    # ── 3. Add user message to history ──────────────────────────────
    sessions.add_message(session.session_id, "user", user_text)

    # ── 4. Fast path — instant HA commands, no LLM ──────────────────
    fast_reply = router.handle_fast(user_text)
    if fast_reply:
        logger.info(f"Fast path handled: '{user_text[:50]}'")
        sessions.add_message(session.session_id, "assistant", fast_reply)
        _archive_expired_sessions()
        return ChatResponse(reply=fast_reply, session_id=session.session_id)

    # ── 5. Classify intent → select agent ───────────────────────────
    intent = router.classify(user_text)
    home_state: str | None = None
    if intent == "home":
        tool_defs = HOME_TOOL_DEFINITIONS
        tool_funcs = HOME_TOOL_FUNCTIONS
        agent_context = HOME_AGENT_CONTEXT
        # Pre-fetch HA state so model can answer status queries without tool calls.
        # Returns None if HA is unreachable — gracefully degrades to tool calling.
        home_state = fetch_ha_state_snapshot()
    elif intent == "file":
        tool_defs = FILE_TOOL_DEFINITIONS
        tool_funcs = FILE_TOOL_FUNCTIONS
        agent_context = FILE_AGENT_CONTEXT
    else:
        tool_defs = ALL_TOOL_DEFINITIONS
        tool_funcs = TOOL_FUNCTIONS
        agent_context = None

    # ── 6. Memory ───────────────────────────────────────────────────
    core = load_core_memory()
    core_text = format_core_for_prompt(core)
    episodic_memories = episodic.recall(user_text) if episodic else []

    # ── 7. System prompt (agent-specific) ───────────────────────────
    system_prompt = build_system_prompt(core_text, episodic_memories, agent_context, home_state)

    # ── 8. Assemble messages ────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(sessions.get_messages(session.session_id))

    # ── 9. Call Ollama — serialized via lock ─────────────────────────
    try:
        with _ollama_lock:
            reply = run_chat_with_tools(messages, tool_defs, tool_funcs)
    except Exception as e:
        logger.error(f"Ollama error: {e}", exc_info=True)
        reply = (
            f"I'm having trouble connecting to my brain "
            f"({type(e).__name__}). Is Ollama running? "
            f"Check: systemctl status ollama"
        )

    # ── 10. Save + return ────────────────────────────────────────────
    sessions.add_message(session.session_id, "assistant", reply)
    _archive_expired_sessions()
    return ChatResponse(reply=reply, session_id=session.session_id)


# ═══════════════════════════════════════════════════════════════════════
# SESSION ARCHIVAL — move expired sessions to episodic memory
# ═══════════════════════════════════════════════════════════════════════

def _archive_expired_sessions():
    """
    Check for expired sessions, summarize them, store in episodic memory,
    then remove from active sessions.

    Called after every chat request. This is lightweight — it only does
    work when a session has actually expired (30 min of inactivity).
    """
    if not episodic:
        return

    for session in sessions.get_expired_sessions():
        if session.messages:
            summary = summarize_session_for_storage(session.messages)
            episodic.store(summary, metadata={"session_id": session.session_id})
            logger.info(
                f"Archived session {session.session_id} "
                f"({len(session.messages)} messages)"
            )
        sessions.remove_session(session.session_id)


def _archive_all_sessions():
    """
    Archive ALL active sessions (called on shutdown).

    When the server is stopping (systemctl stop siy, or Ctrl+C),
    we want to save any in-progress conversations to episodic memory
    so they're not lost entirely.
    """
    if not episodic:
        return

    for sid, session in list(sessions._sessions.items()):
        if session.messages:
            summary = summarize_session_for_storage(session.messages)
            episodic.store(summary, metadata={"session_id": sid})
            logger.info(f"Shutdown archive: session {sid}")


# ═══════════════════════════════════════════════════════════════════════
# UTILITY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """
    Quick health check.

    Uptime Kuma pings this endpoint every 60 seconds.
    HA can also use it to check if Siy is alive before sending messages.

    Returns model name, session count, and episodic memory count
    so you can spot issues at a glance (0 episodic_memories on a
    server that's been running for weeks = something broke).
    """
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "active_sessions": sessions.active_count,
        "episodic_memories": episodic.count() if episodic else 0,
    }


@app.get("/memory/core")
def get_core_memory():
    """
    View current core memory (for debugging or a future UI).

    Try it: curl http://192.168.8.212:8000/memory/core | python3 -m json.tool
    """
    return load_core_memory()


@app.put("/memory/core")
def update_core_memory(data: dict):
    """
    Update core memory. Accepts a full or partial dict.

    MERGE BEHAVIOR:
      Top-level keys get replaced or merged:
        - If both old and new values are dicts → shallow merge
        - Otherwise → new value replaces old

    Example:
      curl -X PUT http://192.168.8.212:8000/memory/core \
        -H "Content-Type: application/json" \
        -d '{"preferences": {"style": "more casual"}}'

      This updates ONLY the "style" key inside "preferences",
      keeping everything else in core memory intact.
    """
    current = load_core_memory()

    # Shallow merge — top-level keys get replaced, nested dicts get merged
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(current.get(key), dict):
            current[key].update(value)
        else:
            current[key] = value

    save_core_memory(current)
    return {"status": "updated", "core_memory": current}


@app.get("/sessions")
def list_sessions():
    """
    List active sessions (for debugging).

    Shows session ID, message count, creation time, and last activity.
    Useful for checking if HA's session is alive and how many messages
    are in the working memory window.

    Try it: curl http://192.168.8.212:8000/sessions | python3 -m json.tool
    """
    return {
        "active": sessions.active_count,
        "sessions": [
            {
                "id": s.session_id,
                "messages": len(s.messages),
                "created": s.created_at,
                "last_active": s.last_active,
            }
            for s in sessions._sessions.values()
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT — run directly with `python app.py`
# ═══════════════════════════════════════════════════════════════════════
#
# You can also run with:
#   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
#
# The __main__ block is a convenience so `python app.py` also works.
# It uses the HOST and PORT from config.py so there's one source of truth.

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,  # Auto-restart on code changes (dev only)
    )
