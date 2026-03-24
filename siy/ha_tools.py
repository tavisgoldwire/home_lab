"""
tools/ha_tools.py — Home Assistant control tools.
══════════════════════════════════════════════════════════════════════════════

THESE TOOLS GIVE SIY "HANDS" — the ability to physically interact with
your home instead of just answering questions.

With these tools, conversations like this work:

  "Turn off the bedroom lamp"
    → ha_call_service(domain="light", service="turn_off", entity_id="light.bedroom_lamp")
    → "Done. Bedroom Lamp is now off."

  "Dim the floor lamp to 40%"
    → ha_call_service(domain="light", service="turn_on",
                      entity_id="light.floor_lamp",
                      service_data={"brightness_pct": 40})
    → "Done. Floor Lamp is now on."

  "What lights do I have?"
    → ha_list_entities(domain="light")
    → "light.bedroom_lamp [on] (Bedroom Lamp) ..."

HOW THIS CONNECTS TO HOME ASSISTANT:
  Home Assistant runs on THIS machine at http://127.0.0.1:8123.
  The HA REST API requires a "Bearer token" in every request — that's
  the long-lived access token you store in the systemd service as
  SIY_HA_TOKEN. Without it, every call returns 401 Unauthorized.

  HA REST API basics:
    GET  /api/states              → all entity states
    GET  /api/states/<entity_id>  → one entity's state + attributes
    POST /api/services/<domain>/<service>  → do something (turn on, off, etc.)

  Example service call:
    POST /api/services/light/turn_on
    Body: {"entity_id": "light.bedroom_lamp", "brightness_pct": 50}

WHAT'S AN ENTITY?
  HA uses "entities" to represent every device. Each entity has:
    - entity_id:  machine name like "light.bedroom_lamp"
    - state:      current status like "on", "off", "unavailable"
    - attributes: extra info like brightness, color, friendly_name

  Your current entities (approximate — use ha_list_entities to confirm):
    light.bedroom_lamp   → TP-Link L535 bedside lamp (ip)
    light.floor_lamp     → TP-Link L535 floor lamp (ip)
    switch.hs300_*       → TP-Link HS300 power strip outlets (ip)

  The friendly names depend on what you named them in HA's UI.
  Use ha_list_entities() to see the exact entity IDs for your setup.

ADDING TO THIS FILE LATER:
  When you add new devices to HA (robot vacuum, climate sensors, etc.),
  you don't need to add new functions. ha_call_service() is a universal
  caller — any HA service, any entity. Just update the DEFINITIONS
  description if you want Siy to know about the new device type.

FILE LOCATION:
  /home/siy_brain/siy/tools/ha_tools.py

DEPENDS ON:
  config.py    → HA_URL, HA_TOKEN
  requests     → already in requirements.txt (used by ollama_client.py)
══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import requests

# ── Import HA connection settings from config ────────────────────────
# HA_URL = "http://127.0.0.1:8123"   (HA Docker on this same machine)
# HA_TOKEN = from environment variable SIY_HA_TOKEN
# Both are already defined in config.py — nothing to change there.
from config import HA_URL, HA_TOKEN

logger = logging.getLogger("siy.tools.ha")


# ═══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS — used by all tool functions below
# ═══════════════════════════════════════════════════════════════════════

def _ha_headers() -> dict:
    """
    Build the HTTP headers required by HA's REST API.

    Every HA REST request needs two headers:
      Authorization: Bearer <your_long_lived_token>
      Content-Type:  application/json

    The Bearer token is what proves to HA that Siy is allowed to
    read state and call services. Without it, HA returns 401 Unauthorized.

    This function is called internally — tool functions don't need
    to think about headers.
    """
    return {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }


def _check_token() -> str | None:
    """
    Verify the HA token is configured before making any API call.

    RETURNS:
      None    → token is present, proceed normally
      str     → error message to return to the LLM (token missing)

    WHY THIS EXISTS:
      HA_TOKEN comes from the SIY_HA_TOKEN environment variable (set
      in the systemd service). If someone runs Siy without setting the
      env var, HA_TOKEN will be an empty string. This gives a clear,
      friendly error instead of a cryptic 401 from HA.

    HOW TO FIX if you see this error:
      sudo systemctl edit siy
      # Add under [Service]:
      #   Environment=SIY_HA_TOKEN=your_token_here
      sudo systemctl restart siy
    """
    if not HA_TOKEN:
        return (
            "Home Assistant token not configured. "
            "Set SIY_HA_TOKEN in the systemd service: "
            "run 'sudo systemctl edit siy' and add: "
            "Environment=SIY_HA_TOKEN=your_token_here"
        )
    return None  # ← None means "all good, proceed"


def _format_state(state_data: dict) -> str:
    """
    Convert a raw HA state dict into a readable summary string.

    HA returns a lot of raw data. This extracts the useful bits:
      - friendly name (the name you gave the device in HA)
      - current state (on/off/unavailable/etc.)
      - brightness as a percentage (for lights)
      - color temperature (for tunable white lights)
      - RGB color (if set)

    ARGS:
      state_data: the JSON dict HA returns for a single entity

    RETURNS:
      A formatted multi-line string for the LLM to read.
      Example:
        Name: Bedroom Lamp
        Entity: light.bedroom_lamp
        State: on
        Brightness: 75%
        Color temp: 370 mireds
    """
    entity_id = state_data.get("entity_id", "unknown")
    state = state_data.get("state", "unknown")
    attr = state_data.get("attributes", {})

    lines = []

    # Friendly name (what you called it in the HA UI) first
    if attr.get("friendly_name"):
        lines.append(f"Name: {attr['friendly_name']}")

    lines.append(f"Entity: {entity_id}")
    lines.append(f"State: {state}")

    # Light-specific attributes (only present when light is on)
    if "brightness" in attr:
        # HA stores brightness as 0-255; convert to 0-100% for readability
        brightness_pct = round(attr["brightness"] / 255 * 100)
        lines.append(f"Brightness: {brightness_pct}%")

    if "color_temp" in attr and attr["color_temp"] is not None:
        lines.append(f"Color temp: {attr['color_temp']} mireds")

    if "rgb_color" in attr and attr["rgb_color"] is not None:
        r, g, b = attr["rgb_color"]
        lines.append(f"Color: RGB({r}, {g}, {b})")

    # For switches: include any power monitoring data if present
    if "current_power_w" in attr:
        lines.append(f"Power draw: {attr['current_power_w']}W")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# TOOL: GET STATE — check what a device is currently doing
# ═══════════════════════════════════════════════════════════════════════

def tool_ha_get_state(entity_id: str) -> str:
    """
    Get the current state and attributes of a single HA entity.

    The LLM calls this when the user asks things like:
      "Is the bedroom lamp on?"
      "What brightness is the floor lamp set to?"
      "What's the power strip doing?"

    ARGS:
      entity_id: the HA entity ID like "light.bedroom_lamp"
                 (use ha_list_entities to find exact IDs)

    RETURNS:
      Formatted string with name, state, brightness, etc.
      Or an error message if the entity doesn't exist.
    """
    # ── Token check ─────────────────────────────────────────────────
    err = _check_token()
    if err:
        return err

    # ── Make the API request ─────────────────────────────────────────
    # GET /api/states/<entity_id> returns the full state object
    # for that one entity. Timeout of 10s is generous — HA usually
    # responds in under 1 second for local requests.
    try:
        url = f"{HA_URL}/api/states/{entity_id}"
        logger.debug(f"Fetching state for: {entity_id}")
        r = requests.get(url, headers=_ha_headers(), timeout=10)
        r.raise_for_status()

        state_data = r.json()
        result = _format_state(state_data)
        logger.info(f"Got state for {entity_id}: {state_data.get('state')}")
        return result

    except requests.HTTPError as e:
        # 404 = entity doesn't exist in HA
        if e.response is not None and e.response.status_code == 404:
            return (
                f"Entity '{entity_id}' not found in Home Assistant. "
                f"Use ha_list_entities to see available entities."
            )
        logger.error(f"HA API error getting state for {entity_id}: {e}")
        return f"Home Assistant API error: {e}"

    except requests.ConnectionError:
        return (
            f"Cannot reach Home Assistant at {HA_URL}. "
            f"Is HA running? Check: docker ps | grep homeassistant"
        )

    except Exception as e:
        logger.error(f"Unexpected error getting state for {entity_id}: {e}", exc_info=True)
        return f"Error getting state for {entity_id}: {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════
# TOOL: LIST ENTITIES — discover what devices exist in HA
# ═══════════════════════════════════════════════════════════════════════

def tool_ha_list_entities(domain: str = "") -> str:
    """
    List Home Assistant entities and their current states.

    The LLM calls this when:
      - It needs to find the exact entity_id before controlling a device
        (e.g., user says "dim the lamp" but LLM doesn't know if it's
        "light.bedroom_lamp" or "light.bedside_lamp")
      - User asks "what devices do I have?" or "show me my lights"

    ARGS:
      domain: optional filter. Pass "light" for only lights,
              "switch" for only switches, "" (empty) for all useful entities.

    RETURNS:
      Formatted list: "  light.bedroom_lamp  [on]  (Bedroom Lamp)"

    DOMAINS YOU'LL USE MOST:
      "light"    → your L535 lamps
      "switch"   → your HS300 power strip outlets
      "sensor"   → temperature, power monitoring, etc.
      "scene"    → saved lighting scenes
      "automation" → your automations
      ""         → all of the above + input_boolean, climate, binary_sensor
    """
    # ── Token check ─────────────────────────────────────────────────
    err = _check_token()
    if err:
        return err

    try:
        # ── Fetch all states ────────────────────────────────────────
        # GET /api/states returns ALL entities in HA.
        # This can be a big list (100+ entities including HA internals).
        # We filter to just the useful domains.
        url = f"{HA_URL}/api/states"
        r = requests.get(url, headers=_ha_headers(), timeout=15)
        r.raise_for_status()
        all_states = r.json()

        # ── Filter by domain ────────────────────────────────────────
        if domain:
            # User asked for a specific domain (e.g., "light")
            filtered = [
                s for s in all_states
                if s["entity_id"].startswith(domain.lower() + ".")
            ]
        else:
            # No domain specified — show the useful domains, skip HA internals
            # (Internal entities like "update.home_assistant_core_update" aren't useful)
            useful_domains = {
                "light",
                "switch",
                "input_boolean",
                "climate",
                "sensor",
                "binary_sensor",
                "automation",
                "scene",
                "cover",        # blinds/garage doors
                "fan",
                "media_player",
            }
            filtered = [
                s for s in all_states
                if s["entity_id"].split(".")[0] in useful_domains
            ]

        # ── Handle empty results ─────────────────────────────────────
        if not filtered:
            if domain:
                return (
                    f"No entities found for domain '{domain}'. "
                    f"Check if '{domain}' is the correct domain name. "
                    f"Common domains: light, switch, sensor, automation, scene"
                )
            return "No entities found. Is Home Assistant running and connected?"

        # ── Format results ───────────────────────────────────────────
        # Sort by entity_id so lights are grouped together, switches together, etc.
        lines = [f"Home Assistant entities ({len(filtered)} found):"]

        for s in sorted(filtered, key=lambda x: x["entity_id"]):
            entity_id = s["entity_id"]
            state = s.get("state", "unknown")
            friendly_name = s.get("attributes", {}).get("friendly_name", "")

            # Format: "  light.bedroom_lamp  [on]  (Bedroom Lamp)"
            # Only show friendly name if it's different from the entity_id
            if friendly_name and friendly_name != entity_id:
                lines.append(f"  {entity_id:<40} [{state}]  ({friendly_name})")
            else:
                lines.append(f"  {entity_id:<40} [{state}]")

        logger.info(f"Listed {len(filtered)} entities (domain filter: '{domain or 'all'}')")
        return "\n".join(lines)

    except requests.ConnectionError:
        return (
            f"Cannot reach Home Assistant at {HA_URL}. "
            f"Is HA running? Check: docker ps | grep homeassistant"
        )

    except Exception as e:
        logger.error(f"Error listing HA entities: {e}", exc_info=True)
        return f"Error listing entities: {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════
# TOOL: CALL SERVICE — control any HA device
# ═══════════════════════════════════════════════════════════════════════

def tool_ha_call_service(
    domain: str,
    service: str,
    entity_id: str,
    service_data: dict | None = None,
) -> str:
    """
    Call a Home Assistant service to control an entity.

    This is the "do something" tool. Turn lights on/off, adjust brightness,
    toggle switches, activate scenes — it all goes through here.

    ARGS:
      domain:       "light", "switch", "homeassistant", "scene", etc.
      service:      "turn_on", "turn_off", "toggle", "activate", etc.
      entity_id:    "light.bedroom_lamp", "switch.hs300_plug_1", etc.
      service_data: optional dict of extra parameters:
                    {"brightness_pct": 50}      → set brightness to 50%
                    {"brightness_pct": 100}     → full brightness
                    {"rgb_color": [255, 0, 0]}  → red (only on color bulbs)
                    {"color_temp": 370}         → warm white in mireds

    COMMON USAGE PATTERNS:
      Turn off a light:
        domain="light", service="turn_off", entity_id="light.bedroom_lamp"

      Dim to 30%:
        domain="light", service="turn_on", entity_id="light.floor_lamp",
        service_data={"brightness_pct": 30}

      Full bright:
        domain="light", service="turn_on", entity_id="light.bedroom_lamp",
        service_data={"brightness_pct": 100}

      Toggle a switch:
        domain="switch", service="toggle", entity_id="switch.hs300_plug_1"

      Turn off EVERYTHING (all lights + switches at once):
        domain="homeassistant", service="turn_off", entity_id="all"
        NOTE: "all" targets all entities — use with care!

    RETURNS:
      Confirmation with the new state, e.g. "Done. Bedroom Lamp is now off."
      Or an error message if HA rejects the request.

    BRIGHTNESS NOTE:
      Use brightness_pct (0-100) rather than brightness (0-255).
      Percentages are natural — "dim to 40%" → brightness_pct: 40.
      HA accepts both formats; brightness_pct is easier for the LLM.
    """
    # ── Token check ─────────────────────────────────────────────────
    err = _check_token()
    if err:
        return err

    # ── Build the service call payload ───────────────────────────────
    # The payload ALWAYS includes the entity_id.
    # service_data (if provided) adds extra params like brightness.
    payload = {"entity_id": entity_id}

    if service_data:
        # Merge service_data into the payload.
        # This is how HA gets brightness, color, color_temp, etc.
        payload.update(service_data)

    logger.info(f"Calling HA service: {domain}.{service} on {entity_id} | data={service_data}")

    try:
        # ── POST to HA's service endpoint ───────────────────────────
        # URL format: POST /api/services/<domain>/<service>
        # Example:    POST /api/services/light/turn_on
        url = f"{HA_URL}/api/services/{domain}/{service}"
        r = requests.post(url, headers=_ha_headers(), json=payload, timeout=15)
        r.raise_for_status()

        # ── Verify by fetching the new state ───────────────────────
        # The service call itself doesn't return the new state — it just
        # returns a list of affected entities. We fetch the state separately
        # to confirm and report back in human language.
        #
        # Special case: "all" isn't a real entity_id, so skip the state fetch.
        if entity_id != "all":
            try:
                state_r = requests.get(
                    f"{HA_URL}/api/states/{entity_id}",
                    headers=_ha_headers(),
                    timeout=10,
                )
                if state_r.ok:
                    state_data = state_r.json()
                    new_state = state_data.get("state", "unknown")
                    name = state_data.get("attributes", {}).get("friendly_name", entity_id)
                    logger.info(f"Confirmed {entity_id} new state: {new_state}")
                    return f"Done. {name} is now {new_state}."
            except Exception:
                # State fetch failed, but service call succeeded — report success anyway
                pass

        # Fallback confirmation (when state fetch fails or entity_id="all")
        return (
            f"Done. Called {domain}.{service} on {entity_id}."
            + (f" (extra params: {service_data})" if service_data else "")
        )

    except requests.HTTPError as e:
        # HA returned an error status code.
        # Common causes:
        #   401 → bad/expired token (regenerate in HA profile)
        #   404 → domain/service doesn't exist
        #   400 → bad request (wrong entity_id format, etc.)
        status = e.response.status_code if e.response is not None else "?"
        if status == 401:
            return (
                "Home Assistant rejected the token (401 Unauthorized). "
                "The token may have been deleted or expired. "
                "Create a new one in HA → Profile → Long-Lived Access Tokens, "
                "then update it in the systemd service."
            )
        elif status == 404:
            return (
                f"Service '{domain}.{service}' not found in Home Assistant (404). "
                f"Check the domain and service names. "
                f"Common services: turn_on, turn_off, toggle"
            )
        logger.error(f"HA service call failed with HTTP {status}: {e}")
        return f"Home Assistant API error (HTTP {status}): {e}"

    except requests.ConnectionError:
        return (
            f"Cannot reach Home Assistant at {HA_URL}. "
            f"Is HA running? Check: docker ps | grep homeassistant"
        )

    except Exception as e:
        logger.error(f"Unexpected error calling HA service: {e}", exc_info=True)
        return f"Error calling service {domain}.{service}: {type(e).__name__}: {e}"


# ═══════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS — sent to Ollama so the model knows these tools exist
# ═══════════════════════════════════════════════════════════════════════
#
# These are the JSON schemas that tell qwen3:8b WHAT tools exist,
# WHAT they do, and WHAT parameters to pass. The "description" fields
# are the most important part — the LLM reads them to decide when to
# call each tool. Write them like you're explaining to a smart intern.
#
# The "required" list tells the LLM which params are mandatory.
# Optional params (like service_data, domain filter) aren't listed there.

DEFINITIONS: list[dict] = [

    # ── Tool 1: Get state of one entity ────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "ha_get_state",
            "description": (
                "Get the current state of a Home Assistant entity (light, switch, sensor, etc.). "
                "Use this when the user asks whether a device is on or off, what brightness "
                "a light is at, or wants to know the current status of something in the home. "
                "If you don't know the exact entity_id, call ha_list_entities first. "
                "Example entity IDs: 'light.bedroom_lamp', 'switch.hs300_plug_1'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": (
                            "The HA entity ID to check. Format: 'domain.name' "
                            "e.g. 'light.bedroom_lamp', 'switch.power_strip'"
                        ),
                    },
                },
                "required": ["entity_id"],
            },
        },
    },

    # ── Tool 2: List entities ──────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "ha_list_entities",
            "description": (
                "List Home Assistant entities and their current states. "
                "Use this FIRST when you need to find the correct entity_id before "
                "controlling a device, or when the user asks 'what lights do I have?', "
                "'show me all my devices', or 'what's available in the home'. "
                "Filter by domain to narrow results: 'light', 'switch', 'sensor', "
                "'automation', 'scene'. Leave domain empty to see all useful entities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": (
                            "Optional domain to filter by. "
                            "Use 'light' for lights only, 'switch' for switches, "
                            "'sensor' for sensors, 'scene' for saved scenes. "
                            "Leave empty for all useful entity types."
                        ),
                    },
                },
                "required": [],
            },
        },
    },

    # ── Tool 3: Call a service (control a device) ──────────────────
    {
        "type": "function",
        "function": {
            "name": "ha_call_service",
            "description": (
                "Control a Home Assistant device by calling a service. "
                "Use this to turn lights on/off, adjust brightness, toggle switches, "
                "or trigger any HA automation or scene. "
                "\n\nCommon usage:\n"
                "  Turn off a light:  domain='light', service='turn_off', entity_id='light.X'\n"
                "  Turn on a light:   domain='light', service='turn_on',  entity_id='light.X'\n"
                "  Set brightness:    service='turn_on', service_data={'brightness_pct': 50}\n"
                "  Toggle a switch:   domain='switch', service='toggle', entity_id='switch.X'\n"
                "  Activate a scene:  domain='scene', service='turn_on', entity_id='scene.X'\n"
                "\nFor brightness, always use brightness_pct (0-100), not raw brightness (0-255). "
                "If you don't know the entity_id, call ha_list_entities first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": (
                            "HA service domain: 'light', 'switch', 'scene', "
                            "'automation', 'homeassistant', 'fan', 'climate', etc."
                        ),
                    },
                    "service": {
                        "type": "string",
                        "description": (
                            "Service to call: 'turn_on', 'turn_off', 'toggle', "
                            "'activate' (for scenes). Most devices support turn_on/off/toggle."
                        ),
                    },
                    "entity_id": {
                        "type": "string",
                        "description": (
                            "Target entity ID e.g. 'light.bedroom_lamp', "
                            "'switch.hs300_plug_1'. Use 'all' to target all entities "
                            "in the domain (use carefully)."
                        ),
                    },
                    "service_data": {
                        "type": "object",
                        "description": (
                            "Optional extra parameters for the service call:\n"
                            "  {'brightness_pct': 50}      → 50% brightness\n"
                            "  {'brightness_pct': 100}     → full brightness\n"
                            "  {'color_temp': 370}         → warm white (higher = warmer)\n"
                            "  {'rgb_color': [255, 0, 0]}  → red color\n"
                            "  {'effect': 'colorloop'}     → color cycling effect"
                        ),
                    },
                },
                "required": ["domain", "service", "entity_id"],
            },
        },
    },
]
