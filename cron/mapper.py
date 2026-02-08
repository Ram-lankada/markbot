import os, json, glob, re, hashlib
from datetime import datetime, timedelta, timezone
from rapidfuzz import fuzz
from collections import defaultdict
from datetime import datetime
import numpy as np

EXPORT_DIR = os.environ["EXPORT_DIR"]
OUT_DIR = os.environ["OUT_DIR"]
EXPORT_MODE = os.environ.get("EXPORT_MODE","all").strip()
LAST_N_DAYS = os.environ.get("LAST_N_DAYS","").strip()
USELESS_THRESHOLD = float(os.environ.get("USELESS_THRESHOLD","0.18"))
DEDUP_SIMILARITY = int(os.environ.get("DEDUP_SIMILARITY","92"))
MAX_NODES = int(os.environ.get("MAX_NODES","2200"))
MAX_BYTES = int(os.environ.get("MAX_BYTES","900000"))

os.makedirs(OUT_DIR, exist_ok=True)

def now_str():
    return datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')

# ---- Time window cutoff (best effort) ----
cutoff = None
if EXPORT_MODE == "last_n_days":
    try:
        n = int(LAST_N_DAYS or "0")
        if n > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=n)
    except:
        cutoff = None

# ---- "Useless chat" filter: conservative (drop only obvious junk) ----
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
    # pure short acknowledgements
    if any(re.match(p, s.lower()) for p in USELESS_PATTERNS):
        return 1.0
    # mostly emojis / punctuation
    if len(re.sub(r'[\W_]+', '', s)) < 3 and len(s) <= 12:
        return 0.9
    # super short with no signal
    if len(s) <= 8 and not re.search(r'[a-zA-Z0-9]', s):
        return 0.8
    # lightweight heuristic: if very short and no links/code/keywords, likely low value
    low_signal = (len(s) < 20
                  and "http" not in s.lower()
                  and "```" not in s
                  and not re.search(r'\b(error|fix|issue|build|install|command|stack|trace)\b', s.lower()))
    if low_signal:
        return 0.35
    return 0.0

# ---- Tagging: keep broad & inclusive ----
TAG_RULES = {
    "Errors & Debugging": [
        r'\b(error|exception|traceback|panic|stack|segfault|failed|failure)\b'
    ],
    "Commands / How-to": [
        r'(^|\n)\s*(sudo\s+)?(apt|dnf|pacman|brew|pip|npm|yarn|pnpm|git|cmake|make|ninja|cargo|go|python|pytest|bash|sh)\b'
    ],
    "Code / Logs": [
        r'```', r'\b(diff|patch)\b', r'\b(0x[0-9a-fA-F]+)\b'
    ],
    "Bitcoin / Core": [
        r'\b(bitcoin|bitcoind|core|signet|testnet|regtest|descriptor|psbt|utxo|mempool|rpc|wallet|txid)\b'
    ],
    "Docker / Env": [
        r'\b(docker|podman|container|compose|volume|wsl|permission denied)\b'
    ],
    "Git / PRs": [
        r'\b(github|pull request|merge|rebase|commit|branch|issue #|pr #)\b'
    ],
    "CI / Build": [
        r'\b(ci|github actions|jenkins|build|compile|linker|cmake|ninja|make|gcc|clang)\b'
    ],
    "Resources / Links": [
        r'https?://'
    ],
}

def tag_text(text: str):
    s = text.lower()
    tags = []
    for tag, patterns in TAG_RULES.items():
        for p in patterns:
            if re.search(p, s, flags=re.I|re.M):
                tags.append(tag)
                break
    if not tags:
        tags = ["General (Keep)"]
    # cap to keep map tidy
    return tags[:4]

def extract_code_blocks(text: str):
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, flags=re.S)
    return [b.strip() for b in blocks if b.strip()]

def extract_commands(text: str):
    cmds = []
    for line in text.splitlines():
        l = line.strip()
        if re.match(r'^(sudo\s+)?(apt|dnf|pacman|brew|pip|npm|yarn|pnpm|git|cmake|make|ninja|cargo|go|python|pytest|bash|sh)\b', l):
            cmds.append(l)
    return cmds[:5]

def normalize(text: str):
    t = re.sub(r'\s+', ' ', text.strip().lower())
    t = re.sub(r'https?://\S+', 'URL', t)
    return t[:1200]

def short_summary(text: str, limit=160):
    s = re.sub(r'\s+', ' ', text.strip())
    # return s if len(s) <= limit else s[:limit-3] + "..."
    return s

def stable_id(channel: str, timestamp: str, author: str, content: str):
    base = f"{channel}|{timestamp}|{author}|{normalize(content)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]

# ---- Read current mindmap file (or choose one) ----
def existing_maps():
    return sorted(glob.glob(os.path.join(OUT_DIR, "mindmap*.md")))

def current_map_path():
    maps = existing_maps()
    if not maps:
        return os.path.join(OUT_DIR, "mindmap_001.md")
    # highest suffix
    def suf(p):
        m = re.search(r"mindmap_(\d+)\.md$", os.path.basename(p))
        return int(m.group(1)) if m else 0
    maps.sort(key=suf)
    return maps[-1]

def count_nodes(md_text: str) -> int:
    # approximate nodes by bullet count
    return sum(1 for line in md_text.splitlines() if line.lstrip().startswith("- "))

def rotate_if_needed(path: str) -> str:
    if not os.path.exists(path):
        return path
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    nodes = count_nodes(txt)
    sizeb = len(txt.encode("utf-8"))
    if nodes < MAX_NODES and sizeb < MAX_BYTES:
        return path
    # rotate
    m = re.search(r"mindmap_(\d+)\.md$", os.path.basename(path))
    idx = int(m.group(1)) if m else 1
    newp = os.path.join(OUT_DIR, f"mindmap_{idx+1:03d}.md")
    return newp


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

def cluster_by_replies(messages):
    """
    messages: list[dict] from DiscordChatExporter JSON (data["messages"]).
    Returns: list[list[dict]] clusters, each is a chronological list of message dicts.
    Logic:
      - If message['type'] == 'Reply', look up message['reference']['messageId']
      - Union current messageId with referenced messageId
      - Output connected components as clusters
    """

    dsu = DSU()
    by_id = {}

    # 1) index messages by messageId (DiscordChatExporter uses "id" usually; but you might have messageId)
    # We'll support both.
    def get_mid(m):
        return str(m.get("id") or m.get("messageId") or "")

    for m in messages:
        mid = get_mid(m)
        if not mid:
            continue
        by_id[mid] = m
        dsu.find(mid)

    # 2) union reply edges
    for m in messages:
        mid = get_mid(m)
        if not mid:
            continue

        mtype = m.get("type", "")
        if mtype == "Reply":
            ref = m.get("reference") or {}
            ref_mid = str(ref.get("messageId") or "")
            if ref_mid:
                # even if referenced msg isn't in the export, create node so chain stays connected
                dsu.find(ref_mid)
                dsu.union(mid, ref_mid)

    # 3) collect clusters
    clusters = defaultdict(list)
    for m in messages:
        mid = get_mid(m)
        if not mid:
            continue
        root = dsu.find(mid)
        clusters[root].append(m)

    # 4) sort inside each cluster by timestamp (best effort)
    def parse_ts(m):
        ts = m.get("timestamp")
        if not ts:
            return ""
        return ts

    out = []
    for _, arr in clusters.items():
        arr.sort(key=parse_ts)
        out.append(arr)

    # 5) sort clusters by size (desc)
    out.sort(key=len, reverse=True)
    return out

def build_reply_clusters_from_items(items):
    dsu = DSU()
    by_id = {it["id"]: it for it in items if it.get("id")}

    for it in items:
        mid = it.get("id")
        if not mid:
            continue
        dsu.find(mid)
        if it.get("reply_to"):
            ref = it["reply_to"]
            dsu.find(ref)
            dsu.union(mid, ref)

    clusters = defaultdict(list)
    for it in items:
        mid = it.get("id")
        if not mid:
            continue
        clusters[dsu.find(mid)].append(it)

    for k in list(clusters.keys()):
        clusters[k].sort(key=lambda x: x.get("timestamp",""))

    out = list(clusters.values())
    out.sort(key=len, reverse=True)
    return out

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def auto_threshold(top1_sims, mode="tight"):
    """
    top1_sims: list/np array of best-neighbor cosine sims.
    mode: 'tight' or 'loose'
    """
    top1_sims = np.asarray(top1_sims)
    if len(top1_sims) == 0:
        return 0.85
    q = 0.93 if mode == "tight" else 0.85
    return float(np.quantile(top1_sims, q))

def faiss_cluster(items, embed_fn, threshold=None, mode="tight"):
    """
    items: list of dicts with 'content'
    embed_fn: function(list[str]) -> np.ndarray [N, D]
    threshold: cosine similarity threshold; if None, auto-tune from data
    Returns list of clusters (list[list[item]])
    """
    import faiss  # pip install faiss-cpu

    texts = [it["content"] for it in items]
    X = embed_fn(texts).astype("float32")
    X = l2_normalize(X)

    index = faiss.IndexFlatIP(X.shape[1])  # inner product == cosine after normalization
    index.add(X)

    # compute top-1 neighbor similarity (excluding self)
    D, I = index.search(X, 2)  # self + nearest
    top1 = D[:, 1]
    if threshold is None:
        threshold = auto_threshold(top1, mode=mode)

    # union-find by similarity edges above threshold
    dsu = DSU()
    for i in range(len(items)):
        dsu.find(i)

    for i in range(len(items)):
        j = int(I[i, 1])
        sim = float(D[i, 1])
        if sim >= threshold:
            dsu.union(i, j)

    clusters = defaultdict(list)
    for i, it in enumerate(items):
        clusters[dsu.find(i)].append(it)

    # Sort inside clusters
    out = []
    for _, arr in clusters.items():
        arr.sort(key=lambda x: x.get("timestamp",""))
        out.append(arr)

    # Sort clusters by size desc
    out.sort(key=len, reverse=True)
    return out, threshold


def cluster_pipeline(items, embed_fn):
    # 1) Reply clustering
    reply_clusters = build_reply_clusters_from_items(items)

    # 2) Build representative text per reply cluster
    reps = []
    for idx, cl in enumerate(reply_clusters):
        # representative = last 5 messages (keeps “resolution”)
        tail = cl[-5:]
        rep_text = "\n".join([f"{m.get('author','')}: {m.get('content','')}" for m in tail])
        reps.append({"cluster_idx": idx, "content": rep_text})

    # 3) Semantic clustering of representatives
    rep_clusters, thr = faiss_cluster(reps, embed_fn, threshold=None, mode="tight")

    # 4) Map back to message clusters
    merged = []
    for rc in rep_clusters:
        merged_msgs = []
        for rep in rc:
            merged_msgs.extend(reply_clusters[rep["cluster_idx"]])
        # optional: sort + dedupe by original message id
        seen = set()
        uniq = []
        for m in sorted(merged_msgs, key=lambda x: x.get("timestamp","")):
            if m["id"] in seen: 
                continue
            seen.add(m["id"])
            uniq.append(m)
        merged.append(uniq)

    merged.sort(key=len, reverse=True)
    return merged, thr


mpath = rotate_if_needed(current_map_path())

# ---- Load known IDs from all existing maps to avoid duplicates across rotations ----
known = set()
for mp in existing_maps():
    with open(mp, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r'\[id:([0-9a-f]{12})\]', line)
            if m:
                known.add(m.group(1))

# ---- Collect messages from exports ----
items = []
for fp in glob.glob(os.path.join(EXPORT_DIR, "*.json")):
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    channel_name = data.get("channel", {}).get("name") or os.path.basename(fp)
    messages = data.get("messages", [])

    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue

        ts = m.get("timestamp")
        dt = None
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except:
                dt = None
        if cutoff and dt and dt < cutoff:
            continue

        # filter only obvious useless
        us = useless_score(content)
        if us >= USELESS_THRESHOLD:
            continue

        author = (m.get("author") or {}).get("name", "")
        msg_id = stable_id(channel_name, ts or "", author, content)
        if msg_id in known:
            continue

        code = extract_code_blocks(content)
        cmds = extract_commands(content)
        tags = tag_text(content)

        ref_mid = ""
        if (m.get("type") == "Reply") and m.get("reference"):
            ref_mid = str(m["reference"].get("messageId",""))

        items.append({
            "id": msg_id,
            "reply_to": ref_mid or None,
            "channel": channel_name,
            "timestamp": ts or "",
            "author": author,
            "url": m.get("url",""),
            "content": content,
            "norm": normalize(content),
            "tags": tags,
            "code_blocks": code,
            "commands": cmds,
        })

# ---- Dedupe near-identical within this run (avoid repeated pastes) ----
items.sort(key=lambda x: (x["channel"], x["timestamp"]))
deduped = []
for it in items:
    if not deduped:
        deduped.append(it); continue
    prev = deduped[-1]
    if it["channel"] == prev["channel"]:
        sim = fuzz.ratio(it["norm"], prev["norm"])
        if sim >= DEDUP_SIMILARITY:
            continue
    deduped.append(it)

# ---- Build append block in required format (single md) ----
header_needed = not os.path.exists(mpath)
lines = []
if header_needed:
    lines.append("# Discord Tagged Map")
    lines.append("")
    lines.append(f"_Created: {now_str()}_")
    lines.append("")
else:
    lines.append("")
    lines.append(f"---")
    lines.append(f"_Update: {now_str()}_")
    lines.append("")

# Group: Tag -> Channel
tag_map = {}
for it in deduped:
    for t in it["tags"]:
        tag_map.setdefault(t, []).append(it)

# stable ordering: bigger tags first
for tag, arr in sorted(tag_map.items(), key=lambda kv: len(kv[1]), reverse=True):
    lines.append(f"## {tag}")
    ch_map = {}
    for it in arr:
        ch_map.setdefault(it["channel"], []).append(it)
    for ch, msgs in sorted(ch_map.items(), key=lambda kv: len(kv[1]), reverse=True):
        lines.append(f"### #{ch}")
        for it in msgs[:120]:
            summ = short_summary(it["content"])
            link = f" ([jump]({it['url']}))" if it["url"] else ""
            extra = ""
            if it["commands"]:
                extra += f" — `cmd:` {it['commands'][0]}"
            if it["code_blocks"]:
                extra += " — `code/logs`"
            lines.append(f"- {summ}{link}{extra}")
        lines.append("")

# Write append
with open(mpath, "a", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(mpath)
