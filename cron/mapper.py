import os, glob, json, re, hashlib, time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from rapidfuzz import fuzz

# Embeddings
from sentence_transformers import SentenceTransformer

# FAISS
import faiss

# Guardrails (schema)
from pydantic import BaseModel, Field, ValidationError

# DSPy (structured prompting)
import dspy
import requests

from dotenv import load_dotenv

load_dotenv() 

# -----------------------------
# Functionality
# -----------------------------

# Load exported Discord JSON files
# Filter “obvious junk”
# Build reply clusters
# Embed cluster reps, merge semantically with FAISS cosine (auto-threshold)
# For each merged cluster, call Qwen via Ollama
# Validate output with Pydantic guardrails
# Write one markdown file with “Issue Cards” (mindmap-friendly)

# -----------------------------
# Config
# -----------------------------
# Default EXPORT_DIR is relative to script location (../exports from cron/ directory)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_default_export_dir = os.path.abspath(os.path.join(os.path.dirname(_script_dir), "exports"))
EXPORT_DIR = os.path.abspath(os.environ.get("EXPORT_DIR", _default_export_dir))
OUT_MD = os.environ.get("OUT_MD", "./out/mindmap.md")
OUT_DIR = os.path.abspath(os.environ.get("OUT_DIR"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # stronger
USELESS_THRESHOLD = float(os.environ.get("USELESS_THRESHOLD", "0.18"))
DEDUP_SIMILARITY = int(os.environ.get("DEDUP_SIMILARITY", "92"))
MAX_MSG_PER_CLUSTER = int(os.environ.get("MAX_MSG_PER_CLUSTER", "80"))
MAX_CHARS_PER_CLUSTER = int(os.environ.get("MAX_CHARS_PER_CLUSTER", "14000"))
SEMANTIC_MODE = os.environ.get("SEMANTIC_MODE", "tight")  # tight|loose
TIME_WINDOW_DAYS = int(os.environ.get("TIME_WINDOW_DAYS", "14"))

# -----------------------------
# Utilities
# -----------------------------
def iso_to_key(ts: str) -> str:
    return ts or ""

def normalize(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = re.sub(r"https?://\S+", "URL", t)
    return t[:1200]

def short_summary(text: str, limit=180) -> str:
    s = re.sub(r"\s+", " ", text.strip())
    # return s if len(s) <= limit else s[:limit-3] + "..."
    return s

USELESS_PATTERNS = [
    r'^\s*(lol+|lmao+|rofl+)\s*$',
    r'^\s*(ok(ay)?|kk|k)\s*$',
    r'^\s*(thanks|thx|ty)\s*$',
    r'^\s*(good (morning|night)|gm|gn)\s*$',
    r'^\s*(hi|hello|hey)\s*$',
    r'^\s*(\+1|same)\s*$',
    r'^\s*🙏+\s*$',
]

def useless_score(text: str) -> float:
    s = text.strip()
    if not s:
        return 1.0
    if any(re.match(p, s.lower()) for p in USELESS_PATTERNS):
        return 1.0
    if len(re.sub(r"[\W_]+", "", s)) < 3 and len(s) <= 12:
        return 0.9
    low_signal = (len(s) < 20 and "http" not in s.lower() and "```" not in s
                  and not re.search(r'\b(error|fix|issue|build|install|command|stack|trace)\b', s.lower()))
    return 0.35 if low_signal else 0.0

def extract_code_blocks(text: str) -> List[str]:
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, flags=re.S)
    return [b.strip() for b in blocks if b.strip()]

def extract_commands(text: str) -> List[str]:
    cmds = []
    for line in text.splitlines():
        l = line.strip()
        if re.match(r'^(sudo\s+)?(apt|dnf|pacman|brew|pip|npm|yarn|pnpm|git|cmake|make|ninja|cargo|go|python|pytest|bash|sh)\b', l):
            cmds.append(l)
    return cmds[:5]

def stable_id(channel: str, timestamp: str, author: str, content: str) -> str:
    base = f"{channel}|{timestamp}|{author}|{normalize(content)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]

# -----------------------------
# Data model for messages
# -----------------------------
@dataclass
class Msg:
    id: str
    channel: str
    timestamp: str
    author: str
    url: str
    content: str
    reply_to: Optional[str] = None
    code_blocks: Optional[List[str]] = None
    commands: Optional[List[str]] = None
    norm: Optional[str] = None

# -----------------------------
# Reply clustering (Union-Find)
# -----------------------------
class DSU:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

def reply_clusters(msgs: List[Msg]) -> List[List[Msg]]:
    dsu = DSU()
    by_id = {m.id: m for m in msgs}

    for m in msgs:
        dsu.find(m.id)
        if m.reply_to:
            # connect even if referenced msg not present
            dsu.find(m.reply_to)
            dsu.union(m.id, m.reply_to)

    clusters = defaultdict(list)
    for m in msgs:
        clusters[dsu.find(m.id)].append(m)

    out = []
    for arr in clusters.values():
        arr.sort(key=lambda x: iso_to_key(x.timestamp))
        out.append(arr)
    out.sort(key=len, reverse=True)
    return out

# -----------------------------
# Semantic merge with FAISS cosine (on cluster representatives)
# -----------------------------
def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def auto_threshold(top1_sims: np.ndarray, mode: str) -> float:
    if top1_sims.size == 0:
        return 0.85
    q = 0.93 if mode == "tight" else 0.85
    return float(np.quantile(top1_sims, q))

def faiss_merge_clusters(cluster_reps: List[str], mode: str) -> Tuple[List[List[int]], float]:
    # embed reps
    embedder = SentenceTransformer(EMBED_MODEL)
    X = embedder.encode(cluster_reps, normalize_embeddings=True, batch_size=64)
    X = np.asarray(X, dtype="float32")

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    D, I = index.search(X, 2)
    top1 = D[:, 1]
    thr = auto_threshold(top1, mode)

    dsu = DSU()
    for i in range(len(cluster_reps)):
        dsu.find(i)

    for i in range(len(cluster_reps)):
        j = int(I[i, 1])
        sim = float(D[i, 1])
        if sim >= thr:
            dsu.union(i, j)

    merged = defaultdict(list)
    for i in range(len(cluster_reps)):
        merged[dsu.find(i)].append(i)

    groups = list(merged.values())
    groups.sort(key=len, reverse=True)
    return groups, thr

# -----------------------------
# Guardrails output schema
# -----------------------------
class IssueCard(BaseModel):
    title: str = Field(..., description="Short problem title")
    problem: str
    environment: Optional[str] = ""
    symptoms: List[str] = Field(default_factory=list)
    root_cause: Optional[str] = ""
    fix_steps: List[str] = Field(default_factory=list)
    commands: List[str] = Field(default_factory=list)
    code_or_config: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    confidence: str = Field(..., description="one of: high|medium|low")

# -----------------------------
# Ollama client + DSPy program
# -----------------------------

class OllamaLM(dspy.LM):
    def __init__(self, model: str, base_url: str):
        super().__init__(model=model)
        self.model = model
        self.base_url = base_url.rstrip("/")

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Handle DSPy calling with messages parameter
        if messages is not None:
            # Convert messages list to prompt string if needed
            if isinstance(messages, list):
                # Format messages list to prompt string
                prompt_parts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if content:
                            prompt_parts.append(f"{role}: {content}")
                    elif isinstance(msg, str):
                        prompt_parts.append(msg)
                prompt = "\n".join(prompt_parts)
            elif isinstance(messages, str):
                prompt = messages
        elif prompt is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        
        # deterministic-ish settings (quality > speed)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_ctx": 8192
            }
        }
        url = f"{self.base_url}/api/generate"
        try:
            r = requests.post(url, json=payload, timeout=180)
            r.raise_for_status()
            response_data = r.json()
            if "response" not in response_data:
                raise ValueError(f"Unexpected response format from Ollama: {response_data}")
            return [response_data["response"]]
        except requests.exceptions.HTTPError as e:
            # Try to get error details from response
            error_msg = f"HTTP {r.status_code} error from Ollama at {url}"
            try:
                error_data = r.json()
                if "error" in error_data:
                    error_msg += f": {error_data['error']}"
            except:
                error_msg += f": {r.text[:200]}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Ollama at {url}: {e}") from e

class SummarizeCluster(dspy.Signature):
    """Turn a Discord message cluster into a strict JSON IssueCard. Do not hallucinate facts."""
    cluster_text: str = dspy.InputField()
    json_issuecard: str = dspy.OutputField(desc="Return ONLY valid JSON matching the IssueCard schema. No markdown.")

class IssueCardProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SummarizeCluster)

    def forward(self, cluster_text: str):
        # Strong guardrails in prompt: schema + no hallucinations
        schema_hint = {
            "title": "string",
            "problem": "string",
            "environment": "string (optional)",
            "symptoms": ["string"],
            "root_cause": "string (optional; empty if unknown)",
            "fix_steps": ["string"],
            "commands": ["string"],
            "code_or_config": ["string"],
            "links": ["string"],
            "open_questions": ["string"],
            "confidence": "high|medium|low"
        }
        instruction = (
            "You are extracting ONLY technical context from Discord debugging chat.\n"
            "Rules:\n"
            "1) Use ONLY the provided messages. If unknown, leave fields empty or add to open_questions.\n"
            "2) Prefer concrete errors, commands, config, and verified steps.\n"
            "3) Output MUST be a single JSON object matching this schema exactly.\n"
            f"Schema: {json.dumps(schema_hint)}\n"
        )
        prompt = instruction + "\nCLUSTER:\n" + cluster_text
        out = self.predict(cluster_text=prompt).json_issuecard
        return out

# -----------------------------
# Cluster text formatting for LLM
# -----------------------------
URL_RE = re.compile(r"https?://\S+")

def format_cluster_for_llm(cluster: List[Msg]) -> str:
    # Keep it readable, chronological, minimal noise.
    lines = []
    total = 0
    for i, m in enumerate(cluster[-MAX_MSG_PER_CLUSTER:], start=1):
        # preserve code blocks & commands separately
        content = m.content.strip()
        # keep links but also collect separately later
        entry = f"[{i}] {m.timestamp} | {m.author} | #{m.channel}\n{content}\n"
        if m.reply_to:
            entry = f"[{i}] (reply_to:{m.reply_to}) {m.timestamp} | {m.author} | #{m.channel}\n{content}\n"
        if total + len(entry) > MAX_CHARS_PER_CLUSTER:
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines)

def extract_links_from_cluster(cluster: List[Msg]) -> List[str]:
    links = []
    for m in cluster:
        links.extend(URL_RE.findall(m.content or ""))
    # unique preserve order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u); out.append(u)
    return out[:20]

# -----------------------------
# Main
# -----------------------------
def load_msgs() -> List[Msg]:
    msgs: List[Msg] = []

    export_file_path = os.path.join(EXPORT_DIR, "*.json")
    print("exportfile path pattern:", export_file_path)
    
    json_files = glob.glob(export_file_path)
    if not json_files:
        print(f"Warning: No JSON files found matching pattern: {export_file_path}")
        return msgs
    
    for fp in json_files:
        print(f"Processing file: {fp}")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"JSON loaded successfully from: {fp}")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file {fp}: {e}")
            continue
        except Exception as e:
            print(f"Error: Failed to load file {fp}: {e}")
            continue
        channel_name = data.get("channel", {}).get("name") or os.path.basename(fp)
        for m in data.get("messages", []):
            content = (m.get("content") or "").strip()
            if not content:
                continue

            if useless_score(content) >= USELESS_THRESHOLD:
                continue

            mid = str(m.get("id") or m.get("messageId") or "")
            if not mid:
                # fallback stable id
                mid = stable_id(channel_name, m.get("timestamp",""), (m.get("author") or {}).get("name",""), content)

            reply_to = None
            # Your specific logic:
            # if type == Reply then reference.messageId
            if (m.get("type") == "Reply") and m.get("reference"):
                reply_to = str((m.get("reference") or {}).get("messageId") or "") or None

            msg = Msg(
                id=mid,
                channel=channel_name,
                timestamp=m.get("timestamp",""),
                author=(m.get("author") or {}).get("name",""),
                url=m.get("url",""),
                content=content,
                reply_to=reply_to,
                code_blocks=extract_code_blocks(content),
                commands=extract_commands(content),
                norm=normalize(content),
            )
            msgs.append(msg)

    # In-run dedupe for near-identical consecutive pastes within same channel
    msgs.sort(key=lambda x: (x.channel, x.timestamp))
    deduped = []
    for m in msgs:
        if not deduped:
            deduped.append(m); continue
        p = deduped[-1]
        if m.channel == p.channel:
            if fuzz.ratio(m.norm or "", p.norm or "") >= DEDUP_SIMILARITY:
                continue
        deduped.append(m)
    return deduped

def safe_parse_issuecard(text: str) -> Optional[IssueCard]:
    # Extract first JSON object if model wraps it
    t = text.strip()
    # crude JSON extraction guard
    if not t.startswith("{"):
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            t = m.group(0)
    try:
        obj = json.loads(t)
        return IssueCard.model_validate(obj)
    except (json.JSONDecodeError, ValidationError):
        return None

def write_md(issuecards: List[Tuple[IssueCard, List[Msg]]], semantic_thr: float):
    os.makedirs(os.path.dirname(OUT_MD) or ".", exist_ok=True)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Discord Technical Digest Map\n\n")
        f.write(f"_Generated: {time.strftime('%Y-%m-%d %H:%M %Z')}_\n\n")
        f.write(f"Semantic merge cosine threshold (auto): **{semantic_thr:.3f}**\n\n")

        for card, cluster in issuecards:
            f.write(f"## {card.title}\n\n")
            f.write(f"**Confidence:** {card.confidence}\n\n")
            f.write(f"**Problem:** {card.problem}\n\n")

            if card.environment:
                f.write(f"**Environment:** {card.environment}\n\n")

            if card.symptoms:
                f.write("**Symptoms / Errors:**\n")
                for s in card.symptoms[:12]:
                    f.write(f"- {s}\n")
                f.write("\n")

            if card.root_cause:
                f.write(f"**Root cause:** {card.root_cause}\n\n")

            if card.fix_steps:
                f.write("**Fix / Steps:**\n")
                for s in card.fix_steps[:12]:
                    f.write(f"- {s}\n")
                f.write("\n")

            cmds = card.commands[:12]
            if cmds:
                f.write("**Commands:**\n")
                for c in cmds:
                    f.write(f"- `{c}`\n")
                f.write("\n")

            if card.code_or_config:
                f.write("**Code / Config (snippets):**\n")
                for sn in card.code_or_config[:6]:
                    f.write("```text\n")
                    f.write(sn.strip()[:1200] + "\n")
                    f.write("```\n")
                f.write("\n")

            links = card.links[:15]
            if links:
                f.write("**Links:**\n")
                for u in links:
                    f.write(f"- {u}\n")
                f.write("\n")

            if card.open_questions:
                f.write("**Open questions:**\n")
                for q in card.open_questions[:10]:
                    f.write(f"- {q}\n")
                f.write("\n")

            # Minimal provenance (jump links)
            f.write("**Source messages:**\n")
            for m in cluster[:8]:
                line = f"- {m.timestamp} — {m.author} — #{m.channel}: {short_summary(m.content)}"
                if m.url:
                    line += f" ({m.url})"
                f.write(line + "\n")
            f.write("\n---\n\n")

def main():
    msgs = load_msgs()
    if not msgs:
        print("No messages found.")
        return

    # 1) reply clustering
    rc = reply_clusters(msgs)

    # 2) representative per reply cluster (last few messages)
    reps = []
    for cl in rc:
        tail = cl[-5:]
        rep = "\n".join([f"{m.author}: {m.content}" for m in tail])
        reps.append(rep)

    # 3) semantic merge on reps
    groups, thr = faiss_merge_clusters(reps, mode=SEMANTIC_MODE)

    # 4) build merged clusters
    merged_clusters: List[List[Msg]] = []
    for g in groups:
        allm = []
        for idx in g:
            allm.extend(rc[idx])
        # sort + unique by message id
        allm.sort(key=lambda x: (x.channel, x.timestamp))
        seen = set()
        uniq = []
        for m in allm:
            if m.id in seen: continue
            seen.add(m.id)
            uniq.append(m)
        # chronological for reading
        uniq.sort(key=lambda x: x.timestamp)
        merged_clusters.append(uniq)

    # Write a markmap-friendly markdown from merged_clusters
    out_path = os.path.join(os.path.abspath(OUT_DIR), f"mindmap_{int(time.time())}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Discord Debugging Issue Mindmap\n")
        f.write(f"**Clustering threshold:** `{thr:.3f}`\n\n")

        for idx, cluster in enumerate(tqdm(merged_clusters, desc="Writing clusters"), 1):
            # Heading for the cluster
            summary = short_summary(cluster[0].content, 60) if cluster else "Cluster"
            title = f"Cluster {idx}: {summary}"
            f.write(f"## {title}\n")

            # # Top-level: Participants and message count
            # authors = {m.author for m in cluster}
            # f.write(f"- **Participants:** {', '.join(authors)}\n")
            # f.write(f"- **Messages:** {len(cluster)}\n")

            # # Show cluster by message (chronological)
            # f.write(f"- **Cluster messages:**\n")
            for m in cluster:
                content = m.content.strip().replace("\n", "  ")
                msg_line = f"- {short_summary(content,120)}"
                if m.url:
                    msg_line += f"\n- ([link]({m.url}))"
                # Include code blocks if any
                if m.code_blocks:
                    for cb in m.code_blocks:
                        msg_line += f"\n- `code`: {cb}"
                # Include commands if any
                if m.commands:
                    for cmd in m.commands:
                        # summarized_cmd = short_summary(cmd, 80)
                        msg_line += f"\n- `command`: {cmd}"
                f.write(msg_line + "\n")
            f.write("\n")
    print(f"Wrote simple markmap to {out_path}")

    # # 5) LLM summarization with guardrails
    # lm = OllamaLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
    # dspy.settings.configure(lm=lm)
    # prog = IssueCardProgram()
    # issuecards = []
    # for cluster in tqdm(merged_clusters[:200], desc="LLM summarizing clusters"):  # cap for sanity
    #     cluster_text = format_cluster_for_llm(cluster)
    #     # Add extracted links into prompt context (so model can place them into links field)
    #     links = extract_links_from_cluster(cluster)
    #     if links:
    #         cluster_text += "\n\nLINKS FOUND:\n" + "\n".join(links)

    #     raw = prog(cluster_text=cluster_text)
    #     card = safe_parse_issuecard(raw)
    #     if card is None:
    #         # fallback: minimal card from heuristic to avoid pipeline break
    #         card = IssueCard(
    #             title="Unparsed cluster (needs review)",
    #             problem=short_summary(cluster[0].content, 200),
    #             environment="",
    #             symptoms=[],
    #             root_cause="",
    #             fix_steps=[],
    #             commands=[],
    #             code_or_config=[],
    #             links=links,
    #             open_questions=["LLM output did not validate. Review cluster manually."],
    #             confidence="low",
    #         )
    #     issuecards.append((card, cluster))

    # write_md(issuecards, thr)
    # print(f"Wrote: {OUT_MD}")

if __name__ == "__main__":
    main()
