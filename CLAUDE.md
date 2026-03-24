# KittyVERSE — Claude Code Context

## What This Is
Tavis's home lab. Siy is a local AI assistant running on OptiPlex #2.
All development happens locally on the Mac, then deployed to the server via SCP.

---

## Server Access
- **SSH alias:** `ssh siy` (configured in ~/.ssh/config)
- **Tailscale IP:** `ip` (use this, not ip — not reachable directly from Mac)
- **User:** `siy_brain`
- **Key:** `~/.ssh/id_ed25519`
- **sudo** requires interactive password — ask the user to run sudo commands in their own terminal

---

## Project Structure

### Local files (Mac) → Server mapping
```
Desktop/Home Lab/Siy/
├── app.py              → ~/siy/app.py           (FastAPI server — main entry point)
├── router.py           → ~/siy/router.py         (fast path + intent classifier — NEW)
├── config.py           → ~/siy/config.py         (all settings: model, ports, paths)
├── ollama_client.py    → ~/siy/ollama_client.py  (Ollama REST client)
├── session_manager.py  → ~/siy/session_manager.py
├── memory.py           → ~/siy/memory.py         (3-tier: working/episodic/core)
├── __init__.py         → ~/siy/tools/__init__.py (tool registry + per-agent subsets)
├── file_tools.py       → ~/siy/tools/file_tools.py
├── ha_tools.py         → ~/siy/tools/ha_tools.py
└── requirements.txt    → ~/siy/requirements.txt
```

### Deploy workflow
```bash
# After editing any file locally:
scp "Siy/<filename>" siy_brain@ip:~/siy/<server-path>

# Then restart (user must do this in their own terminal):
ssh siy "sudo systemctl restart siy"
ssh siy "journalctl -u siy -f --no-pager -n 20"
```

---

## Current Architecture (as of this session)

### Request flow
```
POST /chat
  → router.handle_fast(text)        # regex match → instant HA call, NO LLM
      ├─ matched → return in ~0.2 sec
      └─ no match → router.classify(text)
                      ├─ "home"    → HOME_TOOL_DEFINITIONS + HOME_AGENT_CONTEXT
                      ├─ "file"    → FILE_TOOL_DEFINITIONS + FILE_AGENT_CONTEXT
                      └─ "general" → ALL_TOOL_DEFINITIONS
                            ↓
                      threading.Lock()   # one Ollama call at a time
                            ↓
                      run_chat_with_tools(messages, tool_defs, tool_funcs)
```

### Fast path patterns (in router.py)
- `"turn on/off the <device>"` → instant HA call
- `"dim the <device> to N%"` → instant HA call
- Entity cache loaded at startup from HA REST API (19 devices as of last run)

### Key constants (config.py on server)
- Model: `qwen3:1.7b` (fast but limited; `qwen3:8b` also pulled)
- Ollama timeout: 180 sec
- Server port: 8000
- HA URL: `http://127.0.0.1:8123`
- HA token: set in systemd override, NOT in code

---

## What's Complete

| Task | Status |
|---|---|
| SIY-01: FastAPI gateway, /chat, /health, systemd, ChromaDB memory | DONE |
| SIY-02: ha_tools.py deployed, HA token wired, light control working | DONE |
| SIY-03: Fast path router, threading lock, per-agent tool subsets | DONE |
| SSH key auth from Mac → server | DONE |

---

## Known Issues / Next Steps

### Critical: LLM path times out on complex queries
- `"what's the status of my lights?"` → routes to home agent → LLM calls `ha_list_entities` → takes 3+ min → timeout
- **Root cause:** Tool calling on qwen3:1.7b with large tool schemas is too slow on CPU
- **Fix (not yet done):** For the home agent, pre-fetch HA states and inject into the system prompt instead of tool calling. Model summarizes text instead of calling tools.
- See `app.py: chat_endpoint` and `router.py: classify()`

### Next priorities (from roadmap)
1. **Home agent pre-fetch fix** — inject HA state into prompt, skip tool calling for status queries
2. **SIY-04:** Cron tasks + daily briefing (APScheduler, morning weather + energy summary)
3. **HA-01:** HA → Siy webhook (HA triggers Siy on events: arrival home, energy spike)
4. **MON-02:** Uptime Kuma alerts (Discord/email notifications)

---

## Server Services
```bash
# Siy
ssh siy "systemctl status siy"
ssh siy "journalctl -u siy -f"           # live logs

# Ollama
ssh siy "ollama ps"                       # check if model is loaded/generating
ssh siy "ollama list"                     # qwen3:1.7b (active), qwen3:8b (available)

# Test fast path (should return in <1 sec)
curl -s -X POST http://ip:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "turn on the floor lamp", "session_id": "test"}'

# Health check
curl -s http://ip:8000/health | python3 -m json.tool
```

---

## Hardware Context
- **OptiPlex #2** (Siy host): cpu, 8GB ram, 1 empty RAM slot, 256GB memory
- **RAM upgrade path:** Add 1x 8GB ram-2400 SODIMM → 16GB total (one empty slot confirmed)
- **IoT VLAN:** ip/24, lights at .159 and .191, HS300 at .180
- **Jellyfin public:** watch.jellyfin.com (Cloudflare Tunnel → OptiPlex #1 :30013)
- **Monitoring:** Uptime Kuma on RPi3 (ratpatrol) at ip:3001
