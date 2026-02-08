# Discord Technical Knowledge Pipeline (DCE + Clustering + LLM)

Local-first pipeline to:
- Export Discord messages using **DiscordChatExporter (DCE)**
- Cluster messages using:
  - Reply graph clustering
  - Semantic clustering (FAISS cosine similarity)
- Summarize clusters using **Local LLM (Qwen2.5 7B via Ollama)**
- Generate structured technical knowledge → Mindmap-ready Markdown

---

# Security First
- Never commit `.env`
- Treat Discord token like password
- Prefer bot tokens over user tokens when possible
- If using user token → ensure server rules allow exporting

---

# Architecture Overview

```
Discord Server
   ↓
DiscordChatExporter (JSON)
   ↓
Preprocessing + Filtering
   ↓
Reply Clustering (Union-Find Graph)
   ↓
Semantic Clustering (FAISS + Embeddings)
   ↓
LLM Summarization (Qwen via Ollama)
   ↓
Structured Technical Output (Markdown Mindmap)
```

---

# Requirements

## System
- Linux (Tested Ubuntu 22+)
- Python 3.10+
- Node (optional → for Markmap HTML)
- 16GB RAM recommended (LLM local)

---

# Installation

## Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
```

## Core Packages
```bash
pip install \
faiss-cpu \
numpy \
rapidfuzz \
pydantic \
requests \
tqdm \
sentence-transformers \
dspy-ai
```

## Optional
```bash
pip install lmql
```

---

# Install Ollama + Model

## Install Ollama
Follow official install or:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Pull Model
```bash
ollama pull qwen2.5:7b-instruct
```

## Start Service
```bash
ollama serve
```

---

# DiscordChatExporter (DCE) Setup

## Download Correct Architecture

Check system arch:
```bash
uname -m
```

If:
- `x86_64` → download `linux-x64`
- `aarch64` → download `linux-arm64`

Extract:
```bash
mkdir -p ~/markbot/bin
unzip DiscordChatExporter*.zip -d ~/markbot/bin
chmod +x ~/markbot/bin/DiscordChatExporter*
```

Verify:
```bash
file ~/markbot/bin/DiscordChatExporter*
```

Must show:
```
ELF 64-bit ... x86-64
```

---

# Environment Configuration

Use `env.test` in /cron and update the below credentials 

```env
DISCORD_TOKEN=YOUR_TOKEN
CHANNEL_IDS="123 456"
EXPORT_DIR=./exports
OUT_MD=./out/mindmap.md
OLLAMA_MODEL=qwen2.5:7b-instruct
```

Protect it:
```bash
chmod 600 .env
```

---

# DCE Export Command

Manual Test:
```bash
./DiscordChatExporter.Cli export \
-t "$DISCORD_TOKEN" \
-c CHANNEL_ID \
-f json \
-o output.json
```

---

# Cron Automation (Optional)

Create script:
```bash
nano ~/markbot/cron/cron.sh
```

```
#!/usr/bin/env bash
source ~/markbot/.env
cd ~/markbot/bin

TS=$(date +"%Y%m%d_%H%M")

./DiscordChatExporter.Cli export \
-t "$DISCORD_TOKEN" \
-c CHANNEL_ID \
-f json \
-o ~/markbot/exports/export_$TS.json
```

Make executable:
```bash
chmod +x cron.sh
```

Add cron:
```bash
crontab -e
```

Example (every 30 min):
```
*/30 * * * * /home/user/markbot/cron/cron.sh
```

---

# Clustering Pipeline

## Stage 1 — Reply Graph Clustering
Uses:
```
type == "Reply"
reference.messageId
```

Connected components → conversation clusters.

---

## Stage 2 — Semantic Clustering
Uses:
- bge-small-en-v1.5 embeddings
- FAISS cosine similarity
- Auto threshold:
  - Tight → P93 similarity
  - Loose → P85 similarity

---

# LLM Summarization

Uses:
- Qwen2.5 7B Instruct (Local)
- DSPy structured prompting
- Pydantic schema guardrails

Output schema:
```
Problem
Environment
Symptoms
Root Cause
Fix Steps
Commands
Code/Config
Links
Open Questions
Confidence
```

---

# Output

Generated:
```
out/mindmap.md
```

Ready for:
- Markmap
- Obsidian
- Knowledge Graph import

---

# Run Pipeline - Immediate execution

```bash
source .venv/bin/activate
./cron/cron.sh
```
if you just need the mapping ( i.e. you've the DCE export in you export folder ) : 

```bash
python cron/mapper.py
```

---

# Debug Tips

## DCE Fails
Check:
```
Exec format error → wrong architecture binary
401 → token invalid
403 → channel permission
```

## LLM Fails
Check:
```
curl localhost:11434/api/tags
```

---

# Performance Tips

## Faster
- Use MiniLM embeddings

## Better Quality
- Keep bge embeddings
- Increase context window
- Lower temperature (0.2)

---

# Future Improvements

- Vector DB (Chroma / Qdrant)
- Incremental embedding cache
- Web UI
- Multi-model ensemble validation

---
