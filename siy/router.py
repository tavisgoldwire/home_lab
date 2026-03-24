"""
router.py — Intent classifier, fast path handler, and entity cache.
══════════════════════════════════════════════════════════════════════════════

THIS MODULE SITS IN FRONT OF THE LLM.

It does two things before any Ollama call happens:

  1. FAST PATH — recognizes simple HA commands and executes them directly,
     bypassing the LLM entirely. Result: ~1 sec instead of 15-40 sec.
     Examples: "turn off the floor lamp", "dim the bedside lamp to 40%"

  2. INTENT CLASSIFICATION — for requests that need the LLM, routes them
     to the right agent (home / file / general) so each agent only sees
     the tools it needs. Fewer tools = less model confusion + fewer tokens.

FLOW:
  chat_endpoint(text)
    → router.handle_fast(text)
        ├─ MATCH  → instant HA call, <1 sec, no LLM
        └─ NO MATCH
            → router.classify(text)
                ├─ "home"    → HA tools + home system prompt
                ├─ "file"    → file tools + file system prompt
                └─ "general" → all tools + general system prompt

ENTITY CACHE:
  On startup, we fetch all lights and switches from HA and build a
  name→entity_id map. This lets the fast path resolve "floor lamp" →
  "light.floor_lamp" without asking the LLM.

  The cache is loaded once at startup (via load_entity_cache() in
  app.py lifespan). If HA is down at startup, the fast path silently
  falls through to the LLM on every request until the cache loads.

  To refresh after adding new devices to HA:
    sudo systemctl restart siy
  (or hit POST /router/reload if you add that endpoint later)
══════════════════════════════════════════════════════════════════════════════
"""

import re
import string
import logging
import threading

import requests

from config import HA_URL, HA_TOKEN
from tools.ha_tools import tool_ha_call_service

logger = logging.getLogger("siy.router")


# ═══════════════════════════════════════════════════════════════════════
# ENTITY CACHE — maps friendly names to HA entity IDs
# ═══════════════════════════════════════════════════════════════════════
#
# Example after loading:
#   {
#     "bedside lamp":  "light.bedside_lamp",
#     "floor lamp":    "light.floor_lamp",
#     "hs300 plug 1":  "switch.hs300_plug_1",
#     ...
#   }
#
# Thread-safe: _cache_lock guards reads/writes to _entity_map.

_entity_map: dict[str, str] = {}
_cache_loaded = False
_cache_lock = threading.Lock()


def load_entity_cache() -> bool:
    """
    Fetch lights + switches from HA REST API and build a fuzzy name map.

    Called once at startup from app.py lifespan. Returns True on success.

    WHY NOT LOAD ON FIRST REQUEST?
      Loading on startup means the first real user request is fast.
      If we lazy-load, the first request after a restart would take
      an extra ~1 second for the HTTP call to HA.

    WHAT IF HA IS DOWN AT STARTUP?
      The function catches all exceptions and returns False.
      handle_fast() checks _cache_loaded before trying the fast path,
      so it silently falls through to the LLM if the cache is empty.
    """
    global _entity_map, _cache_loaded

    if not HA_TOKEN:
        logger.warning("Router: SIY_HA_TOKEN not set — fast path disabled")
        return False

    try:
        r = requests.get(
            f"{HA_URL}/api/states",
            headers={"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"},
            timeout=10,
        )
        if not r.ok:
            logger.warning(f"Router: HA returned {r.status_code} — fast path disabled")
            return False

        states = r.json()
        new_map: dict[str, str] = {}

        for s in states:
            entity_id: str = s.get("entity_id", "")
            domain = entity_id.split(".")[0]

            # Only cache controllable entities (lights + switches)
            if domain not in ("light", "switch"):
                continue

            friendly_name: str = s.get("attributes", {}).get("friendly_name", "")
            if friendly_name:
                new_map[friendly_name.lower()] = entity_id

        with _cache_lock:
            _entity_map = new_map
            _cache_loaded = True

        logger.info(f"Router: entity cache loaded — {len(new_map)} devices")
        for name, eid in sorted(new_map.items()):
            logger.debug(f"  '{name}' → {eid}")

        return True

    except requests.ConnectionError:
        logger.warning(f"Router: cannot reach HA at {HA_URL} — fast path disabled")
        return False
    except Exception as e:
        logger.warning(f"Router: cache load failed ({type(e).__name__}: {e}) — fast path disabled")
        return False


def _resolve(device_name: str) -> str | None:
    """
    Fuzzy-match a device name string to its HA entity_id.

    MATCHING ORDER:
      1. Exact match (case-insensitive): "floor lamp" → "light.floor_lamp"
      2. Substring match: "floor" → "light.floor_lamp" (if unambiguous)

    RETURNS:
      entity_id string if matched, None if no match found.

    WHY FUZZY?
      Users say "the lamp", "floor lamp", "my floor lamp" — all should
      resolve to the same entity. The friendly name is the anchor.
    """
    name = device_name.lower().strip()

    with _cache_lock:
        # Exact match first
        if name in _entity_map:
            return _entity_map[name]

        # Substring match — return first hit
        # For a small home (2-10 devices), ambiguity is rare
        for k, v in _entity_map.items():
            if name in k or k in name:
                logger.debug(f"Router: fuzzy match '{name}' → '{k}' ({v})")
                return v

    return None


# ═══════════════════════════════════════════════════════════════════════
# FAST PATH — execute simple HA commands without the LLM
# ═══════════════════════════════════════════════════════════════════════
#
# Patterns (checked in order):
#   turn on/off <device>      → ha_call_service(turn_on / turn_off)
#   dim/set <device> to N%    → ha_call_service(turn_on, brightness_pct=N)
#
# Returns a string response if handled, None to fall through to LLM.
#
# WHY REGEX AND NOT LLM?
#   For "turn off the floor lamp", the LLM does:
#     1. Parse intent
#     2. Decide to call ha_list_entities
#     3. Receive entity list
#     4. Decide to call ha_call_service
#     5. Receive confirmation
#     6. Write a response
#   = 15-40 seconds, 2 Ollama round-trips.
#
#   The regex does:
#     1. Match pattern
#     2. Resolve entity
#     3. Call HA directly
#   = ~0.3 seconds, 1 HTTP call to HA.

_TURN_PATTERN = re.compile(
    r"^turn\s+(on|off)\s+(?:the\s+)?(.+?)\.?\s*$",
    re.IGNORECASE,
)
_DIM_PATTERN = re.compile(
    r"^(?:dim|set)\s+(?:the\s+)?(.+?)\s+to\s+(\d+)\s*%\.?\s*$",
    re.IGNORECASE,
)


def handle_fast(text: str) -> str | None:
    """
    Try to handle a request instantly without the LLM.

    RETURNS:
      A response string if the command was handled (lamp was toggled/dimmed).
      None if the command doesn't match a fast pattern or entity can't be
      resolved — caller should fall through to LLM pipeline.

    SAFE FALLBACK:
      If HA is down, the tool_ha_call_service call returns an error string
      (not an exception). We return that error string so the user gets
      feedback rather than a silent failure.
    """
    if not _cache_loaded:
        # Cache not ready yet (HA was down at startup) — skip fast path
        return None

    t = text.strip()

    # ── turn on / turn off ──────────────────────────────────────────
    m = _TURN_PATTERN.match(t)
    if m:
        action, device = m.groups()
        entity_id = _resolve(device)
        if entity_id:
            domain = entity_id.split(".")[0]
            service = "turn_on" if action.lower() == "on" else "turn_off"
            logger.info(f"Fast path: {service} → {entity_id}")
            return tool_ha_call_service(domain, service, entity_id)

    # ── dim / set to N% ────────────────────────────────────────────
    m = _DIM_PATTERN.match(t)
    if m:
        device, pct_str = m.groups()
        entity_id = _resolve(device)
        if entity_id:
            pct = max(0, min(100, int(pct_str)))  # clamp to 0-100
            logger.info(f"Fast path: dim {entity_id} to {pct}%")
            return tool_ha_call_service(
                "light", "turn_on", entity_id, {"brightness_pct": pct}
            )

    # No fast path matched
    return None


# ═══════════════════════════════════════════════════════════════════════
# INTENT CLASSIFICATION — route to the right LLM agent
# ═══════════════════════════════════════════════════════════════════════
#
# Three agents:
#   home    → HA control (lights, switches, energy monitoring)
#   file    → file system (browse, search, read files on server)
#   general → everything else (identity, memory, research questions)
#
# Classification is keyword-based — fast, deterministic, no LLM needed.
# The keywords are broad enough to catch most real requests. Anything
# that doesn't match home or file goes to the general agent.
#
# WHY NOT USE THE LLM TO CLASSIFY?
#   That would double the Ollama calls for every request. Keyword
#   classification adds ~0ms and is good enough for a personal assistant
#   with a small, well-known domain.

_HOME_KEYWORDS = {
    "light", "lamp", "lights", "lamps",
    "switch", "switches", "plug", "plugs", "outlet", "outlets",
    "brightness", "bright", "dim", "dimmer",
    "scene", "scenes",
    "energy", "power", "watt", "watts",
    "device", "devices",
    "smart", "home",
}

_FILE_KEYWORDS = {
    "file", "files", "folder", "folders",
    "directory", "directories",
    "read", "reading",
    "find", "search", "searching",
    "document", "documents",
    "show", "contents", "content",
    "open", "view",
    "csv", "json", "txt", "log", "logs",
}


def classify(text: str) -> str:
    """
    Classify user intent for LLM agent routing.

    RETURNS:
      "home"    → route to home agent (HA tools + home system prompt)
      "file"    → route to file agent (file tools + file system prompt)
      "general" → route to general agent (all tools + general prompt)

    NOTE: This is only called when handle_fast() returns None.
    Simple on/off commands never reach this function.
    """
    words = {w.strip(string.punctuation) for w in text.lower().split()}

    if words & _HOME_KEYWORDS:
        logger.info(f"Router: intent=home (matched: {words & _HOME_KEYWORDS})")
        return "home"

    if words & _FILE_KEYWORDS:
        logger.info(f"Router: intent=file (matched: {words & _FILE_KEYWORDS})")
        return "file"

    logger.info("Router: intent=general")
    return "general"
