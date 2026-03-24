# KittyVERSE Home Lab — Siy AI Assistant

A locally-hosted, multi-agent AI assistant built on bare metal. Siy runs entirely on a home server — no cloud, no subscriptions, no data leaving the network.

## What It Does

Siy is a personal assistant that controls smart home devices, manages files, and handles research tasks through a unified chat interface. Home Assistant sends messages to Siy via HTTP; Siy routes them to the right agent, executes tools, and responds — all within the local network.

```
POST /chat
  → Fast path router (regex match → instant HA call, ~0.2 sec, no LLM)
      └─ No match → Intent classifier
                      ├─ "home"    → Home agent (HA tools only)
                      ├─ "file"    → File agent (filesystem tools only)
                      └─ "general" → General agent (all tools)
                            ↓
                      Ollama (local LLM inference)
                            ↓
                      Response → Home Assistant
```

## Architecture

```
home_lab/
└── siy/
    ├── app.py               # FastAPI gateway — main entry point
    ├── router.py            # Fast path + intent classifier + entity cache
    ├── config.py            # All settings in one place
    ├── ollama_client.py     # Ollama REST client
    ├── session_manager.py   # Multi-turn conversation tracking
    ├── memory.py            # 3-tier memory system
    └── tools/
        ├── __init__.py      # Tool registry + per-agent subsets
        ├── file_tools.py    # File browse/search/read
        └── ha_tools.py      # Home Assistant device control
```

## Key Features

**Fast path router** — Simple HA commands ("turn off the floor lamp", "dim bedside lamp to 40%") bypass the LLM entirely via regex matching against a cached entity map. Response time ~0.2 sec vs 15–40 sec through the LLM.

**Per-agent tool scoping** — Each agent only sees the tools it needs. The home agent can't accidentally call file tools; the file agent can't call HA services. Fewer tools in context = less model confusion and faster inference on CPU.

**3-tier memory system:**
- *Working memory* — current session (SessionManager, in-RAM sliding window)
- *Episodic memory* — past conversation summaries (ChromaDB vector search)
- *Core memory* — curated facts injected into every request (JSON file)

**HA state pre-fetch** — Before the home agent responds, the server fetches current device states from HA and injects them into the system prompt. Status queries ("are my lights on?") resolve in one LLM pass with no tool-calling round-trips.

**Ollama threading lock** — Serializes LLM calls so concurrent requests queue instead of timing out on CPU-only hardware.

## Stack

| Component | Technology |
|-----------|-----------|
| API gateway | FastAPI + Uvicorn |
| LLM inference | Ollama (local, CPU) |
| Models | qwen3:1.7b / qwen3:8b |
| Vector memory | ChromaDB + sentence-transformers |
| Home automation | Home Assistant REST API |
| Process management | systemd |
| Remote access | Tailscale (admin) + Cloudflare Tunnel (public) |
| Monitoring | Uptime Kuma |

## Hardware

Runs on a repurposed Dell OptiPlex (i5, 8GB DDR4) — no GPU. CPU-only inference with qwen3 models via Ollama. The fast path router and HA state pre-fetch were both designed specifically to minimize LLM round-trips on constrained hardware.

## Quick Start

```bash
# On the server
git clone https://github.com/tavisgoldwire/home_lab
cd home_lab/siy
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set your HA token
export SIY_HA_TOKEN=your_token_here

# Run
uvicorn app:app --host 0.0.0.0 --port 8000

# Test
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "turn on the floor lamp", "session_id": "test"}'

curl http://localhost:8000/health
```

See `siy/SETUP.md` for full deployment guide including systemd service setup.

## Roadmap

- [ ] Voice interface (Whisper STT + Piper TTS, fully local)
- [ ] Proactive alerts (energy spikes, uptime events, calendar awareness)
- [ ] Research agent (PubMed/arXiv search, literature mining, ChromaDB storage)
- [ ] Thesis data pipeline agent (file watcher, metadata validation, analysis templating)
- [ ] Knowledge graph (entity/relationship extraction from conversations)
