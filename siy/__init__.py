"""
tools/__init__.py — Tool registry and dispatcher.
══════════════════════════════════════════════════════════════════════════════

THIS IS THE CENTRAL HUB for all of Siy's tools.

WHAT ARE TOOLS?
  Tools let Siy DO things beyond just generating text. When the LLM
  decides it needs to take an action (read a file, list a directory,
  toggle a light), it emits a "tool call" instead of a text response.
  app.py catches the tool call, runs the appropriate function here,
  and feeds the result back to the LLM.

  Example flow:
    User: "What files are in my documents folder?"
    LLM:  → tool_call: list_files(path="/home/siy_brain/documents")
    Tool: → "budget.xlsx, notes.md, thesis_draft.pdf"
    LLM:  "I found 3 files in your documents: budget.xlsx, notes.md,
           and thesis_draft.pdf."

HOW TO ADD A NEW TOOL:
  1. Create a new file in tools/ (e.g., tools/web_tools.py)
  2. Define your tool functions and their DEFINITIONS (see file_tools.py)
  3. Import them here and add to ALL_TOOL_DEFINITIONS and TOOL_FUNCTIONS
  4. That's it — the tool loop in app.py picks them up automatically

ARCHITECTURE:
  tools/
  ├── __init__.py       ← you are here (registry + dispatcher)
  ├── file_tools.py     ← file system browsing/reading
  └── ha_tools.py       ← Home Assistant control (lights, switches, etc.)

  ALL_TOOL_DEFINITIONS: list of tool schemas sent to Ollama so the
    model knows WHAT tools exist and how to call them.

  TOOL_FUNCTIONS: dict mapping tool names → Python functions so
    execute_tool() knows HOW to run each tool.
══════════════════════════════════════════════════════════════════════════════
"""

import logging

# ── Import file system tool module ──────────────────────────────────
# Each module exports:
#   - DEFINITIONS: list of tool schemas (for Ollama)
#   - Individual functions (for execution)
from tools.file_tools import (
    DEFINITIONS as FILE_TOOL_DEFS,
    tool_list_files,
    tool_search_files,
    tool_read_file,
)

# ── Import Home Assistant tool module ───────────────────────────────
# These tools give Siy the ability to control lights, switches, and
# any other HA-connected device. Requires SIY_HA_TOKEN env variable
# to be set in the systemd service (see SETUP.md step 9).
from tools.ha_tools import (
    DEFINITIONS as HA_TOOL_DEFS,
    tool_ha_get_state,
    tool_ha_list_entities,
    tool_ha_call_service,
)

logger = logging.getLogger("siy.tools")


# ═══════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS — sent to Ollama with every request
# ═══════════════════════════════════════════════════════════════════════
#
# This list tells the LLM what tools are available. Each entry follows
# Ollama's tool format (similar to OpenAI's function calling spec):
#
#   {
#     "type": "function",
#     "function": {
#       "name": "tool_name",
#       "description": "What this tool does (the LLM reads this!)",
#       "parameters": { ... JSON Schema ... }
#     }
#   }
#
# The LLM uses the name + description to decide WHEN to call a tool.
# Good descriptions = better tool selection. Bad descriptions = the
# model calls the wrong tool or ignores tools entirely.
#
# To add more tool modules in the future:
#   ALL_TOOL_DEFINITIONS = FILE_TOOL_DEFS + HA_TOOL_DEFS + NEW_TOOL_DEFS

ALL_TOOL_DEFINITIONS: list[dict] = [
    *FILE_TOOL_DEFS,   # list_files, search_files, read_file
    *HA_TOOL_DEFS,     # ha_get_state, ha_list_entities, ha_call_service
    # Future additions:
    # *WEB_TOOL_DEFS,   # web search, URL fetch
    # *SYS_TOOL_DEFS,   # systemctl status, disk usage, etc.
]


# ═══════════════════════════════════════════════════════════════════════
# TOOL FUNCTION REGISTRY — maps tool names to Python functions
# ═══════════════════════════════════════════════════════════════════════
#
# When the LLM emits a tool call like:
#   {"function": {"name": "ha_call_service", "arguments": {"domain": "light", ...}}}
#
# execute_tool() looks up "ha_call_service" in this dict and calls the
# corresponding function with the provided arguments.
#
# The function names here MUST match the "name" field in the tool
# definitions above. If they don't match, the tool call silently fails.

TOOL_FUNCTIONS: dict[str, callable] = {
    # ── File system tools ────────────────────────────────────────────
    "list_files":   tool_list_files,    # list directory contents
    "search_files": tool_search_files,  # glob search for files
    "read_file":    tool_read_file,     # read a text file

    # ── Home Assistant tools ─────────────────────────────────────────
    "ha_get_state":       tool_ha_get_state,       # check if a device is on/off
    "ha_list_entities":   tool_ha_list_entities,   # find available entity IDs
    "ha_call_service":    tool_ha_call_service,    # turn things on/off/dim/etc.

    # Future additions:
    # "web_search":         tool_web_search,
    # "get_system_status":  tool_get_system_status,
}


# ═══════════════════════════════════════════════════════════════════════
# PER-AGENT TOOL SUBSETS — used by app.py to focus each agent
# ═══════════════════════════════════════════════════════════════════════
#
# Each agent only sees the tools it needs. Benefits:
#   - Fewer tokens in every prompt (smaller tool schema = faster inference)
#   - Less model confusion (file agent can't accidentally call HA tools)
#   - Cleaner logs (tool calls are always relevant to the agent's domain)
#
# home agent  → HA control tools only
# file agent  → file system tools only
# general     → ALL_TOOL_DEFINITIONS + TOOL_FUNCTIONS (full set, above)

HOME_TOOL_DEFINITIONS: list[dict] = [*HA_TOOL_DEFS]
HOME_TOOL_FUNCTIONS: dict[str, callable] = {
    "ha_get_state":     tool_ha_get_state,
    "ha_list_entities": tool_ha_list_entities,
    "ha_call_service":  tool_ha_call_service,
}

FILE_TOOL_DEFINITIONS: list[dict] = [*FILE_TOOL_DEFS]
FILE_TOOL_FUNCTIONS: dict[str, callable] = {
    "list_files":   tool_list_files,
    "search_files": tool_search_files,
    "read_file":    tool_read_file,
}


# ═══════════════════════════════════════════════════════════════════════
# DISPATCHER — called by app.py's tool loop
# ═══════════════════════════════════════════════════════════════════════

def execute_tool(name: str, arguments: dict) -> str:
    """
    Look up a tool by name and execute it with the given arguments.

    ARGS:
      name:      the tool name from the LLM's tool_call
      arguments: dict of keyword arguments (parsed from JSON)

    RETURNS:
      A string result that gets sent back to the LLM as a "tool" message.
      The LLM reads this result and incorporates it into its response.

      On error, returns a human-readable error string (not an exception).
      The LLM can then tell the user what went wrong in natural language.

    WHY RETURN STRINGS:
      Ollama's /api/chat expects tool results as plain text in a message
      with role="tool". Complex return types (dicts, lists) get
      JSON-serialized into strings by the tool functions themselves.
    """
    func = TOOL_FUNCTIONS.get(name)

    if not func:
        # Tool name doesn't exist in our registry.
        # This can happen if the LLM hallucinates a tool name.
        logger.warning(f"Unknown tool called: '{name}' with args: {arguments}")
        return f"Error: tool '{name}' does not exist. Available tools: {list(TOOL_FUNCTIONS.keys())}"

    try:
        logger.info(f"Executing tool: {name}({arguments})")
        result = func(**arguments)
        # Log a preview of the result (truncated for readability)
        logger.debug(f"Tool result preview: {str(result)[:200]}...")
        return result
    except TypeError as e:
        # Wrong arguments — the LLM passed params that don't match
        # the function signature. This usually means the tool definition's
        # parameter schema doesn't match the function's actual args.
        logger.error(f"Tool '{name}' got wrong arguments: {e}")
        return f"Error calling {name}: wrong arguments — {e}"
    except Exception as e:
        # Catch-all for any other error (file not found, permission denied, etc.)
        logger.error(f"Tool '{name}' failed: {e}", exc_info=True)
        return f"Error running {name}: {type(e).__name__}: {e}"
