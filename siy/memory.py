"""
memory.py — Siy's three-tier memory system.
══════════════════════════════════════════════════════════════════════════════

THE THREE TIERS (think of them like human memory):

  1. WORKING MEMORY  → the current conversation
     - Handled by SessionManager (not in this file)
     - Like short-term memory: "what were we just talking about?"
     - Lives in RAM, lost on restart (that's fine)

  2. EPISODIC MEMORY  → summaries of past conversations
     - Stored as vectors in ChromaDB (on disk at /home/siy_brain/siy/memory/chroma/)
     - Like long-term episodic memory: "last week we talked about X"
     - Persists across restarts, survives reboots
     - Searchable by MEANING, not keywords (vector similarity)

  3. CORE MEMORY  → curated facts and preferences
     - Stored as JSON at /home/siy_brain/siy/memory/core_memory.json
     - Like semantic memory: "Tavis is a grad student at UF"
     - Manually editable, highest trust level
     - Injected into EVERY request — Siy always knows these facts

HOW THEY WORK TOGETHER IN A SINGLE REQUEST:
  ┌─────────────────────────────────────────────────────────────────┐
  │  System prompt                                                  │
  │  ├── Identity ("You are Siy...")                                │
  │  ├── Core memory (always included — curated facts)              │
  │  ├── Episodic memories (top-K similar to current message)       │
  │  └── Rules ("be concise", "never invent facts")                │
  │                                                                 │
  │  Working memory (session history — last 30 messages)            │
  │                                                                 │
  │  New user message                                               │
  └─────────────────────────────────────────────────────────────────┘

  Every token in that stack costs inference time on CPU. That's why:
    - Core memory is compact (structured JSON, not prose)
    - Episodic recall is limited to K=4 memories
    - Session history is capped at 30 messages
    - Episodic memories are truncated to 300 chars each

HOW VECTOR SEARCH WORKS (simplified):
  Text → 384 numbers (a "vector" or "embedding")
  The embedding model converts MEANING into MATH.
  Similar meanings → vectors that point in similar directions.

  Example:
    "giraffe gut microbiome" → [0.12, -0.45, 0.89, ...]
    "what did we discuss about my thesis?" → [0.14, -0.42, 0.85, ...]
    These vectors are CLOSE (high cosine similarity) → match!

    "what's the weather?" → [0.72, 0.31, -0.15, ...]
    This vector is FAR from the thesis one → no match.

  ChromaDB stores the vectors and does the nearest-neighbor search.
  The embedding model (all-MiniLM-L6-v2) is ~80MB, runs on CPU,
  loads in ~4 seconds, and encodes a query in ~10-50ms.
"""

import json
import hashlib
import logging
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR, CORE_MEMORY_PATH, EMBEDDING_MODEL, EPISODIC_RECALL_K
)

logger = logging.getLogger("siy.memory")


# ═══════════════════════════════════════════════════════════════════════
# CORE MEMORY — structured facts in a JSON file
# ═══════════════════════════════════════════════════════════════════════
#
# Core memory is the highest-trust tier. These are curated facts that
# Siy should ALWAYS know, regardless of what conversation is happening.
#
# The JSON file lives at /home/siy_brain/siy/memory/core_memory.json
# and is human-editable. You can also update it via PUT /memory/core.
#
# On first run, if the file doesn't exist, we create it with these
# defaults. Edit them to match your current situation.

DEFAULT_CORE = {
    "identity": {
        "user_name": "Tavis",
        "assistant_name": "Siy",
        "description": (
            "Siy is Tavis's personal assistant, the embodiment of his "
            "technical intelligence. Runs locally on OptiPlex #2."
        ),
    },
    "context": {
        "role": "Computational ecology grad student at UF",
        "research": "Giraffe gut microbiomes, Oxford Nanopore 16S, Disney AKL",
        "homelab": (
            "KittyVERSE ecosystem — Jellyfin on TrueNAS (192.168.8.211), "
            "Home Assistant + Siy on OptiPlex #2 (192.168.8.212), "
            "Uptime Kuma on RPi 3, GL.iNet router, managed switch"
        ),
        "projects": "Goldwire Games, Trophic board game, bioinformatics pipelines",
    },
    "preferences": {
        "style": "Practical, concise unless asked for depth",
        "priorities": "Research first, then homelab, then games, then learning",
    },
    "rules": [
        "If unsure, ask",
        "Never invent facts",
        "Protect privacy",
        "Be direct and honest",
    ],
}


def load_core_memory() -> dict:
    """
    Read core memory from the JSON file on disk.

    BEHAVIOR:
      - File exists and valid   → return parsed dict
      - File missing            → create with DEFAULT_CORE, return defaults
      - File corrupted (bad JSON) → log error, return DEFAULT_CORE
        (doesn't overwrite — you might want to manually fix it)

    Called on every chat request by app.py. The file is small (~1KB)
    so reading it each time is fine — no caching needed.
    """
    try:
        with open(CORE_MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.debug("Core memory loaded from disk")
            return data
    except FileNotFoundError:
        # First run — create the file with defaults
        logger.info(
            f"No core memory file at {CORE_MEMORY_PATH} — creating with defaults"
        )
        save_core_memory(DEFAULT_CORE)
        return DEFAULT_CORE
    except json.JSONDecodeError as e:
        # File exists but isn't valid JSON (manual edit gone wrong?)
        # Return defaults but DON'T overwrite — let the user fix it
        logger.error(
            f"Core memory JSON is corrupted: {e}. "
            f"Using defaults. Fix the file at: {CORE_MEMORY_PATH}"
        )
        return DEFAULT_CORE


def save_core_memory(data: dict):
    """
    Write core memory to disk as pretty-printed JSON.

    Pretty-printing (indent=2) makes the file human-readable and
    editable in nano/vim. ensure_ascii=False preserves Unicode
    characters (emoji, accented text, etc.) instead of escaping them.
    """
    with open(CORE_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Core memory saved to disk")


def format_core_for_prompt(core: dict) -> str:
    """
    Convert the core memory dict into a compact string for the system prompt.

    We keep it compact because every token costs inference time.
    The format is:
      key: value
      key: value
    No JSON brackets, no quotes — just readable key-value pairs.

    Example output:
      user_name: Tavis
      assistant_name: Siy
      description: Siy is Tavis's personal assistant...
      role: Computational ecology grad student at UF
      ...
    """
    lines = []
    for section, content in core.items():
        if isinstance(content, dict):
            # Nested dict → flatten each key-value pair
            for key, val in content.items():
                lines.append(f"  {key}: {val}")
        elif isinstance(content, list):
            # List → join into comma-separated string
            lines.append(f"  {section}: {', '.join(str(item) for item in content)}")
        else:
            # Plain value
            lines.append(f"  {section}: {content}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# EPISODIC MEMORY — past conversation summaries in ChromaDB
# ═══════════════════════════════════════════════════════════════════════

class EpisodicMemory:
    """
    Stores and retrieves past conversation summaries using vector similarity.

    LIFECYCLE:
      1. You chat with Siy for a while (messages accumulate in a session)
      2. Session expires (30 min inactivity)
      3. app.py summarizes the session → calls episodic.store(summary)
      4. Summary gets embedded (text → 384-dim vector) and saved to ChromaDB
      5. Next time you chat, episodic.recall(your_message) finds relevant
         past conversations and injects them into the system prompt

    WHY ChromaDB:
      - Embedded database (no separate server process to manage)
      - Persists to disk at /home/siy_brain/siy/memory/chroma/
      - Survives restarts and reboots
      - Fast similarity search (~10ms per query)
      - Simple Python API

    STORAGE FORMAT:
      Each entry in ChromaDB has:
        - id:        SHA-256 hash of the text (first 16 chars)
                     Ensures the same text is never stored twice.
        - document:  the actual text (conversation summary)
        - embedding: 384-dim vector from all-MiniLM-L6-v2
        - metadata:  {"session_id": "...", "stored_at": "2026-03-21T..."}
    """

    def __init__(self):
        """
        Initialize the embedding model and ChromaDB connection.

        This runs once on server startup (called from app.py's lifespan).
        The embedding model load takes ~4 seconds on CPU. After that,
        encoding a query is ~10-50ms.
        """
        # ── Load embedding model ────────────────────────────────────
        # all-MiniLM-L6-v2 produces 384-dimensional vectors.
        # It's ~80MB, runs on CPU, and is the standard choice for
        # lightweight semantic search. If you need better accuracy
        # at the cost of speed, try "all-mpnet-base-v2" (768-dim).
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # ── Connect to ChromaDB ─────────────────────────────────────
        # PersistentClient stores data on disk (not just in RAM).
        # The path is /home/siy_brain/siy/memory/chroma/ — created
        # by config.py's directory initialization at import time.
        logger.info(f"Opening ChromaDB at: {CHROMA_DIR}")
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)

        # ── Get or create the collection ────────────────────────────
        # A "collection" is like a table in a database. We use one
        # collection for all episodic memories. If it already exists
        # (from a previous run), we reopen it with all existing data.
        self.collection = self.client.get_or_create_collection(
            name="siy_episodic",
            metadata={"description": "Summaries of past conversations"},
        )
        logger.info(
            f"Episodic memory ready. {self.collection.count()} entries stored."
        )

    def store(self, text: str, metadata: dict | None = None):
        """
        Store a text snippet (usually a conversation summary) in episodic memory.

        DEDUPLICATION:
          The document ID is a SHA-256 hash of the text. If you store
          the exact same text twice, ChromaDB silently ignores the
          duplicate (same ID = same document). This prevents the
          same conversation from being stored multiple times if the
          archival process runs twice somehow.

        ARGS:
          text:     The text to store (conversation summary)
          metadata: Optional dict of extra info. We always add a
                    "stored_at" timestamp. The caller can add
                    "session_id" or anything else useful.
        """
        # ── Generate stable ID from content ─────────────────────────
        # SHA-256 hash, truncated to 16 hex chars (64 bits).
        # Collision probability is negligible for our use case.
        doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        # ── Compute embedding ───────────────────────────────────────
        # .encode() returns a numpy array; .tolist() converts to plain
        # Python list, which is what ChromaDB expects.
        embedding = self.embedder.encode(text).tolist()

        # ── Attach metadata ─────────────────────────────────────────
        meta = metadata or {}
        meta["stored_at"] = datetime.now().isoformat(timespec="seconds")

        # ── Store in ChromaDB ───────────────────────────────────────
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[meta],
        )
        logger.debug(f"Stored episodic memory: {text[:80]}...")

    def recall(self, query: str, k: int | None = None) -> list[str]:
        """
        Find the K most relevant memories for a given query.

        HOW IT WORKS:
          1. Encode the query text → 384-dim vector
          2. ChromaDB finds the K nearest stored vectors (cosine similarity)
          3. Return the text documents attached to those vectors

        BUG FIX FROM WINDOWS VERSION:
          The old code had a subtle bug where results["documents"] returned
          [[doc1, doc2, ...]] (a list inside a list). If the caller didn't
          handle the nesting, memories wouldn't render in the prompt.
          We flatten it here: return docs[0] to get [doc1, doc2, ...].

        ARGS:
          query: the text to search for (usually the user's latest message)
          k:     how many memories to return (default: EPISODIC_RECALL_K=4)

        RETURNS:
          List of strings, each a past conversation summary.
          Empty list if no memories exist yet (first run).
        """
        k = k or EPISODIC_RECALL_K

        # No memories stored yet — nothing to search
        if self.collection.count() == 0:
            return []

        # Don't ask for more results than actually exist.
        # ChromaDB throws an error if n_results > total documents.
        actual_k = min(k, self.collection.count())

        # ── Encode query and search ─────────────────────────────────
        embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=actual_k,
        )

        # ── Flatten the nested list ─────────────────────────────────
        # ChromaDB returns: {"documents": [[doc1, doc2, ...]], ...}
        # We want: [doc1, doc2, ...]
        docs = results.get("documents", [[]])
        return docs[0] if docs else []

    def count(self) -> int:
        """How many episodic memories are stored. Used by /health endpoint."""
        return self.collection.count()


# ═══════════════════════════════════════════════════════════════════════
# SESSION SUMMARIZATION — converts a session into storable text
# ═══════════════════════════════════════════════════════════════════════

def summarize_session_for_storage(messages: list[dict]) -> str:
    """
    Convert a session's message history into a compact summary string
    for episodic storage.

    CURRENT APPROACH (Phase 2):
      Simple concatenation of messages with role labels.
      Each message truncated to 200 chars to keep summaries compact.

      Example output:
        Tavis: hey can you check my Downloads folder for any CSV files
        Siy: I found 3 CSV files in your Downloads: experiment_data.csv...
        Tavis: read the first one
        Siy: Here are the first 50 lines of experiment_data.csv...

    FUTURE IMPROVEMENT (Phase 3):
      Use the LLM itself to generate a summary. Instead of storing
      the raw back-and-forth, we'd ask qwen3:8b to produce something
      like: "Discussed CSV files in Downloads. Found experiment_data.csv
      containing giraffe microbiome sequencing results."

      This would make vector search MUCH better because the summary
      captures the topic/intent, not just the words used.

    ARGS:
      messages: list of {"role": "user"|"assistant", "content": "..."}

    RETURNS:
      A single string suitable for vector embedding and storage.
    """
    lines = []
    for msg in messages:
        # Map role names to readable labels
        role = "Tavis" if msg["role"] == "user" else "Siy"
        # Truncate long messages to keep the summary compact.
        # 200 chars is enough to capture the gist without bloating
        # the episodic memory database.
        content = msg["content"][:200]
        lines.append(f"{role}: {content}")

    return "\n".join(lines)
