"""
session_manager.py — Tracks conversations across multiple messages.
══════════════════════════════════════════════════════════════════════════════

WHY THIS EXISTS:
  Without sessions, every message Siy receives is isolated. If you said:
    "What's 2+2?"  → "4"
    "Add 3 to that" → ??? (Siy has no idea what "that" refers to)

  With sessions, each conversation is a list of messages that builds up
  as you talk. Siy sees the full history every time it responds.

HOW IT WORKS:
  1. HA sends a message with a session_id (or we generate one)
  2. The message gets appended to that session's history
  3. The full history goes to Ollama so it has context
  4. Siy's response gets appended too
  5. After SESSION_TIMEOUT (30 min) of silence, the session expires
  6. On expiry, app.py summarizes it and stores in episodic memory

WHAT'S A SESSION?
  Think of it like a single conversation. You walk up to Siy, chat for
  10 minutes about your thesis, walk away. That's one session. Later
  you come back and ask about the weather — that's a new session.

  Home Assistant uses a fixed session_id ("ha_main") so all HA messages
  share one ongoing conversation. This means "turn on the light" followed
  by "now dim it" works naturally.

MEMORY LIFECYCLE:
  ┌─────────────────────────────────────────────────────────────────┐
  │  You say something                                              │
  │       ↓                                                         │
  │  Message added to WORKING MEMORY (this session's message list)  │
  │       ↓                                                         │
  │  Session stays active for 30 min of inactivity                  │
  │       ↓                                                         │
  │  Session expires → summarized → stored in EPISODIC MEMORY       │
  │       ↓                                                         │
  │  Session removed from RAM (but searchable in ChromaDB forever)  │
  └─────────────────────────────────────────────────────────────────┘

PERSISTENCE:
  Sessions live in RAM only. When the Siy server restarts, active sessions
  are lost. This is fine because:
    - Important conversations get archived to episodic memory on expiry
    - HA just starts a new session on the next message
    - The systemd service auto-restarts Siy, so downtime is brief
"""

import time
import uuid
import logging
from dataclasses import dataclass, field

from config import MAX_SESSION_MESSAGES, SESSION_TIMEOUT

logger = logging.getLogger("siy.sessions")


# ═══════════════════════════════════════════════════════════════════════
# SESSION DATA CLASS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Session:
    """
    One conversation session.

    This is a Python dataclass — like a struct in C or a plain object.
    The @dataclass decorator auto-generates __init__, __repr__, etc.
    from the field definitions below.

    Attributes:
        session_id:  Unique identifier. Either a UUID (auto-generated)
                     or a fixed string like "ha_main" (from HA).

        messages:    Ordered list of conversation turns. Each entry:
                     {"role": "user"|"assistant", "content": "..."}

                     NOTE: system messages are NOT stored here. The system
                     prompt (identity, memory, rules) is assembled fresh
                     on every request by app.py. This keeps sessions clean
                     and lets the system prompt evolve without invalidating
                     old session data.

        created_at:  Unix timestamp (float) when the session started.
                     Used for debugging and logging, not for expiry.

        last_active: Unix timestamp (float) of the most recent message.
                     This is what determines expiry — if
                     (now - last_active) > SESSION_TIMEOUT, the session
                     is considered expired and eligible for archival.
    """
    session_id: str
    messages: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════
# SESSION MANAGER
# ═══════════════════════════════════════════════════════════════════════

class SessionManager:
    """
    Manages all active sessions in memory.

    This is intentionally simple — sessions live in a Python dict.
    No database, no persistence, no distributed state. When the server
    restarts, active sessions are gone (that's fine — episodic memory
    catches the important stuff).

    Thread safety: FastAPI runs requests in a thread pool by default.
    For a single-user assistant like Siy, concurrent access to the
    same session is extremely unlikely. If you ever need thread safety,
    wrap _sessions access in a threading.Lock.
    """

    def __init__(self):
        # The dict that holds everything. Key = session_id, Value = Session.
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str | None = None) -> Session:
        """
        Get an existing session by ID, or create a new one.

        BEHAVIOR:
          - session_id=None     → create new session with random UUID
          - session_id="ha_main" and exists → return it, refresh last_active
          - session_id="ha_main" and missing → create it with that ID

        The _cleanup() call at the top removes zombie sessions (2x timeout)
        that somehow weren't archived. This is a safety net, not the
        primary expiry mechanism (that's _archive_expired_sessions in app.py).
        """
        # Housekeeping: remove any truly stale sessions
        self._cleanup()

        if session_id and session_id in self._sessions:
            # Existing session — touch the timestamp so it stays alive
            session = self._sessions[session_id]
            session.last_active = time.time()
            return session

        # Create a new session
        # If no session_id provided, generate a random UUID.
        # If a specific ID was given (like "ha_main"), use it.
        new_id = session_id or str(uuid.uuid4())
        session = Session(session_id=new_id)
        self._sessions[new_id] = session
        logger.info(f"New session: {new_id}")
        return session

    def add_message(self, session_id: str, role: str, content: str):
        """
        Append a message to a session's history.

        SLIDING WINDOW:
          If the history exceeds MAX_SESSION_MESSAGES (30), we trim from
          the front (oldest messages removed first). This keeps the context
          window from growing forever.

          Why 30? qwen3:8b has a 32K token context window, but practical
          usage is ~8K tokens once you include system prompt + core memory
          + episodic memories. 30 messages ≈ 3-4K tokens of conversation,
          leaving room for everything else.

          The trimmed messages aren't lost forever — when the session
          eventually expires, the FULL history (before trimming) would
          ideally be summarized. Current implementation summarizes whatever
          is left in the session at expiry time. Phase 3 improvement:
          summarize incrementally as messages are trimmed.

        ARGS:
          session_id: which session to append to
          role:       "user" or "assistant"
          content:    the message text
        """
        if session_id not in self._sessions:
            # Session doesn't exist — this shouldn't happen in normal flow
            # but can if the server restarted mid-conversation.
            logger.warning(f"Session {session_id} not found, creating new")
            self.get_or_create(session_id)

        session = self._sessions[session_id]
        session.messages.append({"role": role, "content": content})
        session.last_active = time.time()

        # ── Sliding window trim ─────────────────────────────────────
        if len(session.messages) > MAX_SESSION_MESSAGES:
            trimmed = len(session.messages) - MAX_SESSION_MESSAGES
            session.messages = session.messages[trimmed:]
            logger.debug(f"Trimmed {trimmed} old messages from session {session_id}")

    def get_messages(self, session_id: str) -> list[dict]:
        """
        Get all messages in a session for sending to Ollama.

        Returns a COPY (via list()) so the caller can modify it
        (e.g., prepend a system message) without mutating our state.
        """
        if session_id in self._sessions:
            return list(self._sessions[session_id].messages)
        return []

    def get_expired_sessions(self) -> list[Session]:
        """
        Return sessions that have been inactive longer than SESSION_TIMEOUT.

        Called by app.py's _archive_expired_sessions() after every chat
        request. Expired sessions get summarized and stored in episodic
        memory, then removed via remove_session().

        Note: this returns the Session objects themselves, not copies.
        The caller reads .messages for summarization, then calls
        remove_session() to delete them.
        """
        now = time.time()
        expired = []
        for session in self._sessions.values():
            if now - session.last_active > SESSION_TIMEOUT:
                expired.append(session)
        return expired

    def remove_session(self, session_id: str):
        """
        Remove a session after it's been archived to episodic memory.

        Uses .pop() with a default so it doesn't crash if the session
        was already removed (e.g., by _cleanup running in parallel).
        """
        self._sessions.pop(session_id, None)
        logger.info(f"Removed session: {session_id}")

    def _cleanup(self):
        """
        Safety net: remove sessions that are WAY past expired (2x timeout).

        Normal flow: session expires → app.py archives it → remove_session()
        This catches any that slip through (e.g., if archival threw an
        error). At 2x timeout (60 min with default settings), we just
        drop them — better to lose a session than leak memory.
        """
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > SESSION_TIMEOUT * 2
        ]
        for sid in stale:
            logger.info(f"Cleaning up stale session: {sid}")
            self._sessions.pop(sid)

    @property
    def active_count(self) -> int:
        """How many sessions are currently active. Used by /health endpoint."""
        return len(self._sessions)
