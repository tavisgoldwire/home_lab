"""
tools/file_tools.py — File system tools (browse, search, read).
══════════════════════════════════════════════════════════════════════════════

THESE TOOLS LET SIY INTERACT WITH THE FILE SYSTEM.

Without these, Siy can only talk — it can't look at files, find documents,
or read content. With these, conversations like this work:

  "Find any CSV files in my documents"
    → tool_search_files(root="/home/siy_brain/documents", pattern="*.csv")
    → "Found 3 files: data.csv, results.csv, budget.csv"

  "Read the first 50 lines of data.csv"
    → tool_read_file(path="/home/siy_brain/documents/data.csv", max_lines=50)
    → [file content]

SAFETY BOUNDARY:
  All tools check that the requested path is inside one of the
  ALLOWED_FILE_ROOTS defined in config.py. This prevents the LLM
  from reading sensitive system files (/etc/shadow, SSH keys, etc.)
  even if a prompt injection attack tries to trick it.

  The check uses os.path.realpath() to resolve symlinks — so you
  can't bypass the boundary with "../../../etc/shadow" tricks either.

WHAT CHANGED FROM WINDOWS:
  - All paths are Linux (forward slashes, /home/siy_brain/...)
  - os.path.realpath() instead of os.path.abspath() (resolves symlinks)
  - No drive letters (C:\\ → /)
  - File permissions matter on Linux (user siy_brain must have read access)
"""

import os
import fnmatch
import logging

from config import ALLOWED_FILE_ROOTS, MAX_FILE_READ_SIZE

logger = logging.getLogger("siy.tools.files")


# ═══════════════════════════════════════════════════════════════════════
# SAFETY CHECK — is this path inside an allowed root?
# ═══════════════════════════════════════════════════════════════════════

def _is_path_allowed(path: str) -> bool:
    """
    Check if a file path is inside one of the allowed root directories.

    SECURITY:
      We use os.path.realpath() which:
        1. Converts relative paths to absolute ("/home/siy_brain/./docs" → "/home/siy_brain/docs")
        2. Resolves symlinks (a symlink to /etc/ won't bypass the check)
        3. Normalizes ".." traversal ("/home/siy_brain/docs/../../etc" → "/etc")

      Then we check if the resolved path starts with any allowed root.
      This means even a cleverly crafted path can't escape the sandbox.

    ARGS:
      path: the path to check (can be relative, contain .., or be a symlink)

    RETURNS:
      True if the resolved path is inside an allowed root, False otherwise.
    """
    # Resolve to absolute path, following symlinks
    real = os.path.realpath(path)

    for root in ALLOWED_FILE_ROOTS:
        # Normalize the root too, for consistent comparison
        real_root = os.path.realpath(root)
        # os.path.commonpath would work too, but startswith is simpler
        # and handles the edge case where root="/home/siy" and
        # path="/home/siy_brain" (they share a prefix but aren't nested)
        # by checking for the trailing separator.
        if real == real_root or real.startswith(real_root + os.sep):
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════
# TOOL: LIST FILES — show contents of a directory
# ═══════════════════════════════════════════════════════════════════════

def tool_list_files(path: str) -> str:
    """
    List files and subdirectories in a given directory.

    The LLM calls this when the user asks things like:
      "What's in my documents folder?"
      "Show me the files in ~/siy/tools/"

    RETURNS:
      A formatted string listing each entry with [DIR] or [FILE] prefix
      and file size for files. Example:
        [DIR]  tools/
        [FILE] app.py (4.2 KB)
        [FILE] config.py (2.1 KB)

    ARGS:
      path: directory to list (must be inside ALLOWED_FILE_ROOTS)
    """
    # ── Safety check ────────────────────────────────────────────────
    if not _is_path_allowed(path):
        return (
            f"Access denied: '{path}' is outside allowed directories. "
            f"Allowed roots: {ALLOWED_FILE_ROOTS}"
        )

    # ── Verify it's a directory ─────────────────────────────────────
    if not os.path.isdir(path):
        if os.path.isfile(path):
            return f"'{path}' is a file, not a directory. Use read_file to see its contents."
        return f"'{path}' does not exist."

    # ── List contents ───────────────────────────────────────────────
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return f"Permission denied: cannot read directory '{path}'"

    if not entries:
        return f"Directory '{path}' is empty."

    lines = [f"Contents of {path} ({len(entries)} items):"]

    for entry in entries:
        full_path = os.path.join(path, entry)

        if os.path.isdir(full_path):
            lines.append(f"  [DIR]  {entry}/")
        else:
            # Show file size in human-readable format
            try:
                size = os.path.getsize(full_path)
                size_str = _format_size(size)
                lines.append(f"  [FILE] {entry} ({size_str})")
            except OSError:
                lines.append(f"  [FILE] {entry} (size unknown)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# TOOL: SEARCH FILES — find files matching a pattern
# ═══════════════════════════════════════════════════════════════════════

def tool_search_files(root: str, pattern: str = "*") -> str:
    """
    Recursively search for files matching a glob pattern.

    The LLM calls this when the user asks things like:
      "Find all CSV files in my documents"
      "Search for anything with 'giraffe' in the filename"

    PATTERN EXAMPLES:
      "*.csv"         → all CSV files
      "*.py"          → all Python files
      "*giraffe*"     → anything with "giraffe" in the name
      "*.md"          → all Markdown files

    LIMITS:
      Returns at most 50 results to avoid flooding the LLM context.
      If there are more matches, tells the user to narrow the search.

    ARGS:
      root:    directory to search from (must be inside ALLOWED_FILE_ROOTS)
      pattern: glob pattern to match filenames against (default: "*" = all)
    """
    # ── Safety check ────────────────────────────────────────────────
    if not _is_path_allowed(root):
        return (
            f"Access denied: '{root}' is outside allowed directories. "
            f"Allowed roots: {ALLOWED_FILE_ROOTS}"
        )

    if not os.path.isdir(root):
        return f"'{root}' is not a directory."

    # ── Walk the directory tree ─────────────────────────────────────
    matches = []
    max_results = 50  # Safety cap to prevent context flooding

    for dirpath, dirnames, filenames in os.walk(root):
        # Security: skip any directory that resolves outside allowed roots.
        # This handles symlinks inside the search tree that point elsewhere.
        if not _is_path_allowed(dirpath):
            dirnames.clear()  # Don't descend into this subtree
            continue

        for filename in filenames:
            # fnmatch does Unix-style glob matching (case-sensitive on Linux)
            if fnmatch.fnmatch(filename, pattern):
                full_path = os.path.join(dirpath, filename)
                # Show path relative to the search root for readability
                rel_path = os.path.relpath(full_path, root)
                try:
                    size = os.path.getsize(full_path)
                    matches.append(f"  {rel_path} ({_format_size(size)})")
                except OSError:
                    matches.append(f"  {rel_path}")

                if len(matches) >= max_results:
                    matches.append(f"  ... (stopped at {max_results} results, narrow your search)")
                    return f"Search for '{pattern}' in {root}:\n" + "\n".join(matches)

    if not matches:
        return f"No files matching '{pattern}' found in {root}"

    return f"Found {len(matches)} files matching '{pattern}' in {root}:\n" + "\n".join(matches)


# ═══════════════════════════════════════════════════════════════════════
# TOOL: READ FILE — read contents of a text file
# ═══════════════════════════════════════════════════════════════════════

def tool_read_file(path: str, max_lines: int = 100) -> str:
    """
    Read and return the contents of a text file.

    The LLM calls this when the user asks things like:
      "Read my config file"
      "Show me the first 50 lines of data.csv"

    SAFETY LIMITS:
      - Path must be inside ALLOWED_FILE_ROOTS
      - File must be under MAX_FILE_READ_SIZE (500KB)
      - Output limited to max_lines (default 100)

    WHY LIMIT LINES?
      The LLM's context window is finite (~8K usable tokens for qwen3:8b
      after system prompt + memory). A 10,000-line log file would blow
      through the context window and confuse the model. 100 lines is
      enough to show the user meaningful content while leaving room
      for the model to respond intelligently.

    ARGS:
      path:      file to read (must be inside ALLOWED_FILE_ROOTS)
      max_lines: maximum number of lines to return (default: 100)
    """
    # ── Safety check: allowed path? ─────────────────────────────────
    if not _is_path_allowed(path):
        return (
            f"Access denied: '{path}' is outside allowed directories. "
            f"Allowed roots: {ALLOWED_FILE_ROOTS}"
        )

    # ── Does the file exist? ────────────────────────────────────────
    if not os.path.isfile(path):
        if os.path.isdir(path):
            return f"'{path}' is a directory, not a file. Use list_files to see its contents."
        return f"File not found: '{path}'"

    # ── Safety check: file size ─────────────────────────────────────
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return f"Cannot check file size: {e}"

    if size > MAX_FILE_READ_SIZE:
        return (
            f"File too large: {_format_size(size)} "
            f"(limit is {_format_size(MAX_FILE_READ_SIZE)}). "
            f"This is a safety limit to prevent loading huge files into memory."
        )

    # ── Read the file ───────────────────────────────────────────────
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            # Read line by line up to max_lines to avoid loading
            # the entire file into memory at once
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"\n... (truncated at {max_lines} lines, file has more)")
                    break
                lines.append(line.rstrip("\n"))  # Remove trailing newlines
    except PermissionError:
        return f"Permission denied: cannot read '{path}'"
    except UnicodeDecodeError:
        return f"Cannot read '{path}': file appears to be binary, not text."
    except Exception as e:
        return f"Error reading '{path}': {type(e).__name__}: {e}"

    # ── Format output ───────────────────────────────────────────────
    # Include the filename and line count in the header so the LLM
    # can reference them in its response to the user.
    total_lines = len(lines)
    header = f"File: {path} ({_format_size(size)}, showing {total_lines} lines)"
    content = "\n".join(lines)

    return f"{header}\n{'─' * 40}\n{content}"


# ═══════════════════════════════════════════════════════════════════════
# HELPER — human-readable file sizes
# ═══════════════════════════════════════════════════════════════════════

def _format_size(size_bytes: int) -> str:
    """
    Convert bytes to human-readable string.
    1024 → "1.0 KB", 1048576 → "1.0 MB", etc.
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ═══════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS — sent to Ollama so the model knows these tools exist
# ═══════════════════════════════════════════════════════════════════════
#
# These follow Ollama's tool definition format (OpenAI-compatible).
# The "description" field is critical — it's what the LLM reads to
# decide when to use each tool. Write it like you're explaining the
# tool to a coworker who needs to know when and how to use it.
#
# The "parameters" field uses JSON Schema to describe the arguments.
# "required" lists which params the LLM must provide.

DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories in a given path. "
                "Use this when the user asks what's in a folder or wants to browse files. "
                "Returns file names and sizes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory to list",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Recursively search for files matching a filename pattern. "
                "Use this when the user asks to find files by name or extension. "
                "Supports glob patterns like *.csv, *giraffe*, *.py"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": "Absolute path to the root directory to search from",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match filenames (e.g., '*.csv', '*report*')",
                    },
                },
                "required": ["root", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a text file. "
                "Use this when the user asks to see, read, or inspect a file's contents. "
                "Returns the first N lines (default 100). Cannot read binary files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to return (default: 100)",
                    },
                },
                "required": ["path"],
            },
        },
    },
]
