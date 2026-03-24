"""
config.py — All Siy Brain configuration in one place (Linux / OptiPlex #2)
══════════════════════════════════════════════════════════════════════════════

THIS FILE IS THE SINGLE SOURCE OF TRUTH for every tunable setting.
Every other module imports from here — nothing is hardcoded elsewhere.

WHY THIS EXISTS:
  The old Windows version scattered paths and constants across files.
  This centralizes everything so when you need to change a port, model,
  timeout, or path, you edit ONE file. No hunting.

WHAT CHANGED FROM WINDOWS:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Windows (old)                  →  Linux (this file)            │
  │  ─────────────────────────────────────────────────────────────  │
  │  C:\\SiyBrain                    →  /home/siy_brain/siy         │
  │  r"C:\\Users\\Tavis\\..."        →  /home/siy_brain/...         │
  │  Port 8123                      →  Port 8000                   │
  │  (HA was elsewhere)             →  (HA is on 8123 on this box) │
  └─────────────────────────────────────────────────────────────────┘

  Port 8000 is deliberate: Home Assistant Docker is already on 8123
  on this same machine (192.168.8.212). Uptime Kuma is watching
  192.168.8.212:8000 and will go green the moment this server starts.

NETWORK CONTEXT (for your reference):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Device          │  IP              │  Key Ports                │
  │  ─────────────────────────────────────────────────────────────  │
  │  GL.iNet Router  │  192.168.8.1     │  80 (admin)              │
  │  Managed Switch  │  192.168.8.2     │  —                       │
  │  HS300 Strip     │  192.168.8.180   │  —                       │
  │  Bedside Lamp    │  192.168.8.159   │  —                       │
  │  Floor Lamp      │  192.168.8.191   │  —                       │
  │  TrueNAS/NAS     │  192.168.8.211   │  80 (web), 30013 (JF)   │
  │  OptiPlex #2     │  192.168.8.212   │  8123 (HA), 8000 (Siy)  │
  │  RPi 3           │  192.168.8.???   │  3001 (Uptime Kuma)      │
  └─────────────────────────────────────────────────────────────────┘
"""

import os


# ═══════════════════════════════════════════════════════════════════════
# OLLAMA — the LLM engine that does the actual "thinking"
# ═══════════════════════════════════════════════════════════════════════
#
# Ollama runs on THIS machine (OptiPlex #2) as a system service.
# It exposes a REST API on port 11434 by default.
#
# Two key endpoints:
#   /api/chat     → multi-turn conversations (what Siy uses)
#   /api/generate → single-shot prompts (old approach, inferior)
#   /api/tags     → list available models (used for health checks)
#
# If you ever move Ollama to a different machine (e.g., your gaming PC
# for GPU acceleration), just change OLLAMA_BASE to that machine's IP.
# Everything else stays the same.

OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"

# ── Model ───────────────────────────────────────────────────────────
# qwen3:8b is your current model. On the i5-6500T (CPU-only, no GPU),
# expect ~15-40 second response times depending on prompt length.
#
# If that's too slow, your options (from fastest to best quality):
#   "qwen3:1.7b"      → ~3-8 sec, decent for simple tasks
#   "phi3:mini"        → ~5-12 sec, good balance (3.8B params)
#   "qwen3:8b-q4_0"   → ~10-25 sec, same model but quantized harder
#   "qwen3:8b"         → ~15-40 sec, current (best quality)
#
# To switch: change this value, then `ollama pull <model_name>` in SSH.
MODEL_NAME = "qwen3:1.7b"  # swap back to qwen3:8b after RAM upgrade (Thursday)

# ── Timeout ─────────────────────────────────────────────────────────
# How long (in seconds) to wait for Ollama to finish generating.
# On CPU-only with 8B params, some complex prompts + tool loops can
# take a while. 180 seconds gives plenty of headroom.
# If you switch to a smaller model, you can lower this to 60-90.
OLLAMA_TIMEOUT = 180  # seconds


# ═══════════════════════════════════════════════════════════════════════
# PATHS — where Siy stores its data on disk
# ═══════════════════════════════════════════════════════════════════════
#
# Everything lives under /home/siy_brain/siy/. The directory structure:
#
#   /home/siy_brain/siy/
#   ├── config.py            ← you are here
#   ├── app.py               ← FastAPI server
#   ├── ollama_client.py     ← talks to Ollama
#   ├── session_manager.py   ← conversation tracking
#   ├── memory.py            ← 3-tier memory system
#   ├── requirements.txt     ← Python dependencies
#   ├── tools/               ← tool modules (file browser, HA control, etc.)
#   │   ├── __init__.py
#   │   └── file_tools.py
#   ├── memory/              ← persistent data (survives restarts)
#   │   ├── core_memory.json ← curated facts (auto-created on first run)
#   │   └── chroma/          ← ChromaDB vector database for episodic memory
#   └── logs/                ← application logs

SIY_DIR = "/home/siy_brain/siy"

# ── Memory storage ──────────────────────────────────────────────────
# MEMORY_DIR holds all persistent memory data.
# CHROMA_DIR is specifically for ChromaDB's vector database files.
# CORE_MEMORY_PATH is the JSON file with curated facts about you.
MEMORY_DIR = os.path.join(SIY_DIR, "memory")
CHROMA_DIR = os.path.join(MEMORY_DIR, "chroma")
CORE_MEMORY_PATH = os.path.join(MEMORY_DIR, "core_memory.json")

# ── Logs ────────────────────────────────────────────────────────────
# Application logs go here. The FastAPI server also logs to stdout,
# but file logs are useful for debugging crashes after they happen.
LOGS_DIR = os.path.join(SIY_DIR, "logs")


# ═══════════════════════════════════════════════════════════════════════
# SERVER — FastAPI settings
# ═══════════════════════════════════════════════════════════════════════
#
# Port 8000: chosen because HA Docker already occupies 8123 on this box.
# Uptime Kuma is already watching 192.168.8.212:8000 — it'll go green
# the moment you start the server.
#
# HOST 0.0.0.0: listen on all network interfaces so HA (and anything
# else on KittyNET) can reach Siy. If you used 127.0.0.1, only
# processes on this same machine could connect.

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000


# ═══════════════════════════════════════════════════════════════════════
# SESSIONS — conversation tracking
# ═══════════════════════════════════════════════════════════════════════
#
# A "session" is one continuous conversation. When you chat with Siy
# through HA, each message includes a session_id. Messages within
# the same session share context — Siy remembers what you said earlier.
#
# MAX_SESSION_MESSAGES: how many messages to keep in the sliding window.
#   Too high = fills the model's context window (qwen3:8b has 32K tokens,
#   but ~8K is practical once you include system prompt + memory).
#   Too low = Siy forgets mid-conversation.
#   30 is a good balance for an 8B model.
#
# SESSION_TIMEOUT: seconds of inactivity before a session expires.
#   When a session expires, it gets summarized and stored in episodic
#   memory (so it's not truly lost — just compressed). 30 minutes
#   means if you walk away, Siy starts fresh when you come back,
#   but the old conversation is searchable in memory.

MAX_SESSION_MESSAGES = 30
SESSION_TIMEOUT = 1800  # 30 minutes


# ═══════════════════════════════════════════════════════════════════════
# MEMORY — the three-tier system
# ═══════════════════════════════════════════════════════════════════════
#
# Tier 1: WORKING MEMORY  → current session (handled by SessionManager)
# Tier 2: EPISODIC MEMORY → past conversation summaries (ChromaDB)
# Tier 3: CORE MEMORY     → curated facts (JSON file)
#
# EPISODIC_RECALL_K: how many past memories to pull into each request.
#   These get added to the system prompt as context. More memories =
#   better recall but costs tokens. 4 is a good starting point.
#   If Siy's responses get slow or confused, try lowering to 2-3.
#
# EMBEDDING_MODEL: the model that converts text → vectors for ChromaDB.
#   "all-MiniLM-L6-v2" is small (~80MB), fast, and good enough for
#   similarity search. It runs on CPU and loads in ~4 seconds.
#   This is NOT the LLM — it's a separate model just for search.

EPISODIC_RECALL_K = 4
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ═══════════════════════════════════════════════════════════════════════
# FILE TOOLS — directories Siy is allowed to browse
# ═══════════════════════════════════════════════════════════════════════
#
# SAFETY BOUNDARY: Siy can ONLY see files inside these directories.
# Any request to read/list files outside these roots gets rejected.
# This prevents the LLM from accidentally (or through prompt injection)
# reading sensitive system files like /etc/shadow or SSH keys.
#
# Add paths as you need them. Common additions:
#   - Research data directories
#   - Shared folders from TrueNAS (if you mount them via NFS/SMB)
#   - Downloads or staging directories
#
# On this Ubuntu box, siy_brain's home is /home/siy_brain.
# The Siy project itself is in /home/siy_brain/siy.

ALLOWED_FILE_ROOTS = [
    "/home/siy_brain/siy",           # Siy's own project files (for self-inspection)
    "/home/siy_brain/documents",     # General documents (create this dir as needed)
    "/home/siy_brain/downloads",     # Downloads staging area
    # ── Add more as your needs grow ──
    # "/mnt/nas/media",              # Example: NFS mount from TrueNAS
    # "/home/siy_brain/research",    # Example: research data directory
]

# ── File size safety ────────────────────────────────────────────────
# Max file size (bytes) that Siy will attempt to read into memory.
# Prevents accidentally loading a 2GB log file into the LLM context.
# 500KB is generous for text files — increase if you need to read
# larger datasets, but remember the LLM can only use ~8K tokens of
# context anyway. Reading a 10MB file would waste time.
MAX_FILE_READ_SIZE = 500_000  # ~500KB


# ═══════════════════════════════════════════════════════════════════════
# HOME ASSISTANT — connection details for HA tool integration
# ═══════════════════════════════════════════════════════════════════════
#
# Siy can control HA entities (lights, switches, etc.) via HA's REST API.
# Since HA Docker runs on THIS machine, we use localhost.
#
# To get your long-lived access token:
#   1. Open HA web UI → your profile (bottom-left)
#   2. Scroll to "Long-Lived Access Tokens"
#   3. Create one, name it "Siy", copy the token
#   4. Paste it below (or better: set it as an environment variable)
#
# SECURITY NOTE: In production, use an environment variable instead of
# hardcoding the token. For now, hardcoding is fine since this machine
# is on your local network behind the GL.iNet router.
#
# To use an env var instead:
#   export SIY_HA_TOKEN="your_token_here"  (in .bashrc or systemd unit)
#   Then this line reads it automatically.

HA_URL = "http://127.0.0.1:8123"
HA_TOKEN = os.environ.get("SIY_HA_TOKEN", "")
# ↑ Empty string means "not configured yet" — HA tools will gracefully
#   skip if this is blank. Fill it in when you're ready to wire up
#   light control and other automations.


# ═══════════════════════════════════════════════════════════════════════
# TOOL LOOP — safety limits for function calling
# ═══════════════════════════════════════════════════════════════════════
#
# When Siy calls a tool (e.g., "list files in ~/documents"), the response
# goes back to the LLM, which might call another tool, and so on.
# MAX_TOOL_ROUNDS prevents infinite loops if the model gets confused.
# 5 rounds handles even complex multi-step tasks (browse → filter → read).

MAX_TOOL_ROUNDS = 5


# ═══════════════════════════════════════════════════════════════════════
# DIRECTORY CREATION — ensure all data directories exist on startup
# ═══════════════════════════════════════════════════════════════════════
#
# This runs at import time (when any module does `from config import ...`).
# os.makedirs with exist_ok=True is safe to call repeatedly — it creates
# the directory if missing and does nothing if it already exists.
# This means you never get a "directory not found" crash on first run.

for _dir in [MEMORY_DIR, CHROMA_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)
