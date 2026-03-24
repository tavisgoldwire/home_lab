# Siy Brain (SIY-01) — Deployment Guide for OptiPlex #2

## Machine Details

| Field        | Value                          |
|--------------|--------------------------------|
| Hostname     | `siy`                          |
| IP           |                                |
| User         |                                |
| OS           | Ubuntu Server 24.04            |
| CPU          | Intel i5-6500T (4C/4T, no GPU) |
| RAM          | 8 GB DDR4                      |
| Also running | Home Assistant Docker (:8123), Ollama (:11434) |

## What's in the Box

```
~/siy/
├── app.py               ← FastAPI server (main entry point)
├── config.py            ← all settings in one place
├── ollama_client.py     ← talks to Ollama /api/chat
├── session_manager.py   ← multi-turn conversation tracking
├── memory.py            ← 3-tier memory (core + episodic + working)
├── requirements.txt     ← Python dependencies
├── SETUP.md             ← you are here
├── tools/
│   ├── __init__.py      ← tool registry + dispatcher
│   └── file_tools.py    ← file browse/search/read tools
├── memory/              ← persistent data (created automatically)
│   ├── core_memory.json ← curated facts (auto-created on first run)
│   └── chroma/          ← ChromaDB vector database
└── logs/                ← application logs (created automatically)
```

## Step-by-Step Deployment

### 1. SSH into OptiPlex #2

```bash
ssh 
```

### 2. Verify Ollama is Running

```bash
# Check the service
systemctl status ollama

# Verify the model is pulled
ollama list
# Should show qwen3:8b

# If not pulled yet:
ollama pull qwen3:8b
```

### 3. Copy Files into Place

From your local machine (where you downloaded these files):

```bash
# Copy all project files to the server
scp app.py config.py ollama_client.py session_manager.py memory.py \
    requirements.txt SETUP.md \
    @ip:~/siy/

# Copy the tools directory
scp tools/__init__.py tools/file_tools.py \
    @ip:~/siy/tools/
```

Or if you have the files on a USB drive / git repo, clone/copy them
directly on the server into `~/siy/`.

### 4. Create Python Virtual Environment

```bash
cd ~/siy

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip (avoids warnings)
pip install --upgrade pip

# Install dependencies
# NOTE: This will take 5-10 minutes on first run because
# sentence-transformers pulls PyTorch (~800MB)
pip install -r requirements.txt
```

### 5. Test It Manually

```bash
cd ~/siy
source .venv/bin/activate

# Start the server in foreground (Ctrl+C to stop)
python app.py
```

You should see:

```
Siy Brain (SIY-01) starting up...
  Host: 0.0.0.0:port
  Model: qwen3:8b
Ollama OK. Model 'qwen3:8b' available.
Loading embedding model: all-MiniLM-L6-v2
Episodic memory ready. 0 entries stored.
Core memory loaded: ['identity', 'context', 'preferences', 'rules']
Siy Brain ready.
```

From another terminal (or your PC), test it:

```bash
# Health check (should return immediately)
curl http://ip:port/health

# Send a test message (will take 15-40 sec on CPU)
curl -X POST http://ip:port/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "hey Siy, who am I?", "session_id": "test"}'

# View core memory
curl http://ip:port/memory/core | python3 -m json.tool

# View active sessions
curl http://ip:port/sessions | python3 -m json.tool
```

Once this works, Ctrl+C the server and proceed to step 6.

### 6. Create systemd Service (Production)

This makes Siy start automatically on boot and restart on crashes.

```bash
sudo nano /etc/systemd/system/siy.service
```

Paste this:

```ini
[Unit]
# ── What is this service? ──────────────────────────────────────────
Description=Siy Brain - Local AI Assistant (SIY-01)

# ── Start order ────────────────────────────────────────────────────
# Wait for networking so Ollama (localhost) is reachable.
# Wait for Ollama specifically so the model is available.
After=network.target ollama.service

[Service]
# ── Run as the  user (not root!) ──────────────────────────
User=
Group=

# ── Working directory ──────────────────────────────────────────────
WorkingDirectory=/home//siy

# ── The actual command ─────────────────────────────────────────────
# Uses the venv's Python directly so we don't need to activate it.
# uvicorn runs the FastAPI app on 0.0.0.0:port.
# --no-access-log suppresses per-request HTTP logs (we have our own).
ExecStart=/home//siy/.venv/bin/uvicorn app:app \
    --host 0.0.0.0 \
    --port port \
    --no-access-log

# ── Restart policy ─────────────────────────────────────────────────
# If Siy crashes, wait 5 seconds then restart automatically.
# "on-failure" means only restart on crashes, not on clean shutdown.
Restart=on-failure
RestartSec=5

# ── Environment ────────────────────────────────────────────────────
# Set the HA token here so it's not hardcoded in config.py.
# Replace YOUR_TOKEN_HERE with an actual long-lived access token
# from Home Assistant (Profile → Long-Lived Access Tokens).
# Leave commented out until you're ready to wire up HA tools.
# Environment=SIY_HA_TOKEN=YOUR_TOKEN_HERE

[Install]
# ── Start on boot ─────────────────────────────────────────────────
WantedBy=multi-user.target
```

Then enable and start it:

```bash
# Reload systemd to pick up the new service file
sudo systemctl daemon-reload

# Enable (start on boot)
sudo systemctl enable siy

# Start now
sudo systemctl start siy

# Check status
sudo systemctl status siy

# View live logs
journalctl -u siy -f
```

### 7. Verify Uptime Kuma Goes Green

Once the service is running, check Uptime Kuma (on the RPi 3).
The "Siy (future)" monitor watching `http://ip:port`
should flip from red to green within 60 seconds.

### 8. Wire Up Home Assistant

In your HA configuration, add a REST command to talk to Siy:

```yaml
# In configuration.yaml (or a split file if you use packages)
rest_command:
  siy_chat:
    url: "http://127.0.0.1:port/chat"
    method: POST
    content_type: "application/json"
    payload: '{"text": "{{ text }}", "session_id": "ha_main"}'
    timeout: 180
```

After adding the REST command, restart HA:

```bash
docker restart homeassistant
```

### 9. Set the HA Token (When Ready for HA Tools)

When you're ready to let Siy control lights, switches, etc.:

1. Open HA web UI → Profile (bottom-left) → Long-Lived Access Tokens
2. Create one, name it "Siy"
3. Copy the token
4. Edit the systemd service:

```bash
sudo systemctl edit siy
```

Add:

```ini
[Service]
Environment=SIY_HA_TOKEN=your_long_token_here
```

Then restart:

```bash
sudo systemctl restart siy
```

---

## Quick Reference

### Service Commands

```bash
sudo systemctl start siy      # start
sudo systemctl stop siy       # stop
sudo systemctl restart siy    # restart
sudo systemctl status siy     # status
journalctl -u siy -f          # live logs
journalctl -u siy --since "1 hour ago"  # recent logs
```

### API Endpoints

```bash
# Health check
curl http://ip:port/health

# Chat
curl -X POST http://ip:port/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "hello", "session_id": "test"}'

# View core memory
curl http://ip:port/memory/core

# Update core memory
curl -X PUT http://ip:port/memory/core \
  -H "Content-Type: application/json" \
  -d '{"preferences": {"style": "more casual"}}'

# List sessions
curl http://ip:port/sessions
```

### File Locations

| What                  | Where                                    |
|-----------------------|------------------------------------------|
| Project code          | `/home//siy/`                   |
| Virtual environment   | `/home//siy/.venv/`             |
| Core memory           | `/home//siy/memory/core_memory.json` |
| Episodic memory (DB)  | `/home//siy/memory/chroma/`     |
| Logs                  | `/home//siy/logs/`              |
| systemd service       | `/etc/systemd/system/siy.service`        |
| Ollama models         | `/usr/share/ollama/.ollama/models/`      |

### Troubleshooting

**Siy won't start:**
```bash
journalctl -u siy -n 50    # check last 50 log lines
systemctl status ollama     # is Ollama running?
ollama list                 # is qwen3:8b pulled?
```

**Slow responses (>60 sec):**
- Normal for complex prompts on CPU. qwen3:8b on i5-6500T = 15-40 sec.
- If consistently slow, try a smaller model:
  ```bash
  ollama pull qwen3:1.7b
  # Then edit config.py: MODEL_NAME = "qwen3:1.7b"
  sudo systemctl restart siy
  ```

**"Cannot reach Ollama" errors:**
```bash
systemctl status ollama
# If not running:
sudo systemctl start ollama
sudo systemctl restart siy
```

**Core memory got corrupted:**
```bash
# View the file
cat ~/siy/memory/core_memory.json

# If it's garbled, delete it (defaults will regenerate on next start)
rm ~/siy/memory/core_memory.json
sudo systemctl restart siy
```

**Episodic memory seems wrong:**
```bash
# Nuclear option: wipe episodic memory and start fresh
rm -rf ~/siy/memory/chroma/
sudo systemctl restart siy
```

---

## What's Next

- **HA tools**: Add `tools/ha_tools.py` with light/switch control via HA REST API
- **LLM summarization**: Use qwen3 to generate conversation summaries (better episodic recall)
- **Streaming responses**: Show tokens as they generate (faster perceived latency)
- **Gateway/sub-agent architecture**: Siy as orchestrator managing specialized agents
