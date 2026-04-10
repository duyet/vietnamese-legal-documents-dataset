"""
Vietnamese Legal Documents → Unsloth Instruction Dataset

Modes (auto-detected from flags):
  --upload <repo>   Upload checkpoint to HuggingFace
  --generate        Use LLMs to create QA pairs (default: simple convert)
  --resume          Continue from last checkpoint

Features:
  - HTML → Markdown for better text quality
  - Local parquet cache (fast re-runs)
  - JSONL checkpointing (safe overnight runs)
  - Multi-turn conversations per document
  - Auto-detect OpenRouter / NVIDIA from .env.local
  - Model rotation with rate limit handling

Usage:
  python generate.py                              # Build cache + preview
  python generate.py --upload duyet/vietnamese-legal-instruct
  python generate.py --generate --limit 50        # LLM QA pairs
  python generate.py --generate --limit 500 --resume
  python generate.py --upload duyet/vietnamese-legal-instruct --resume
"""

import argparse
import json
import os
import random
import re
import sys
import time
from html import unescape
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env.local")

BASE = Path(__file__).parent
CACHE = BASE / "cache"
CKPT = BASE / "checkpoint.jsonl"

# ── Models ───────────────────────────────────────────────────────────

MODELS = [
    {"name": "google/gemma-4-31b-it:free",            "provider": "openrouter", "ctx": 262144, "rpm": 10},
    {"name": "google/gemma-4-26b-a4b-it:free",        "provider": "openrouter", "ctx": 262144, "rpm": 10},
    {"name": "qwen/qwen3-next-80b-a3b-instruct:free", "provider": "openrouter", "ctx": 262144, "rpm": 10},
    {"name": "nvidia/nemotron-3-super-120b-a12b:free", "provider": "openrouter", "ctx": 262144, "rpm": 10},
    {"name": "google/gemma-3-27b-it:free",             "provider": "openrouter", "ctx": 131072, "rpm": 20},
    {"name": "google/gemma-4-31b-it",                  "provider": "nvidia", "ctx": 131072, "rpm": 10},
    {"name": "qwen/qwen3.5-397b-a17b",                "provider": "nvidia", "ctx": 131072, "rpm": 5},
    {"name": "qwen/qwen3.5-122b-a10b",                "provider": "nvidia", "ctx": 131072, "rpm": 10},
    {"name": "meta/llama-4-maverick-17b-128e-instruct","provider": "nvidia", "ctx": 131072, "rpm": 10},
    {"name": "google/gemma-3-27b-it",                  "provider": "nvidia", "ctx": 131072, "rpm": 10},
    {"name": "deepseek-ai/deepseek-v3.2",              "provider": "nvidia", "ctx": 131072, "rpm": 5},
]

# ── QA Templates ─────────────────────────────────────────────────────

PROMPTS = [
    {
        "type": "summarize",
        "system": "Bạn là chuyên gia pháp luật Việt Nam. Tóm tắt chính xác, ngắn gọn. Giữ thuật ngữ pháp lý. Tiếng Việt.",
        "user": "Tóm tắt văn bản sau trong 3-5 câu (mục đích, đối tượng, quy định chính, phạm vi):\n\n```\n{content}\n```",
    },
    {
        "type": "key_provisions",
        "system": "Bạn là chuyên gia phân tích pháp luật Việt Nam. Trình bày có hệ thống. Tiếng Việt.",
        "user": "Phân tích các quy định chính:\n\n```\n{content}\n```\n\n1. Các điều/khoản quan trọng\n2. Ý nghĩa pháp lý\n3. Phạm vi áp dụng",
    },
    {
        "type": "qa_practical",
        "system": "Bạn là luật sư tư vấn pháp luật Việt Nam. Trả lời chính xác, trích dẫn điều khoản. Tiếng Việt.",
        "user": "Từ văn bản:\n\n```\n{content}\n```\n\n1. Ai phải tuân thủ?\n2. Nghĩa vụ cụ thể?\n3. Xử lý vi phạm?\n4. Cơ quan thẩm quyền?",
    },
    {
        "type": "explain_simple",
        "system": "Bạn phiên dịch pháp luật cho công chúng. Giải thích đơn giản, ví dụ thực tế. Tiếng Việt.",
        "user": "Giải thích cho người không biết pháp luật:\n\n```\n{content}\n```\n\nDùng từ đơn giản, ví dụ thực tế, tóm tắt quyền/nghĩa vụ.",
    },
    {
        "type": "scope",
        "system": "Bạn là chuyên gia pháp luật hành chính. Phân tích phạm vi áp dụng, hiệu lực. Tiếng Việt.",
        "user": "Phạm vi áp dụng:\n\n```\n{content}\n```\n\n1. Đối tượng\n2. Phạm vi địa lý\n3. Thời gian hiệu lực\n4. Văn bản liên quan\n5. Điều kiện áp dụng",
    },
    {
        "type": "rights",
        "system": "Bạn chuyên tư vấn quyền/nghĩa vụ pháp lý. Phân tích chi tiết. Tiếng Việt.",
        "user": "Quyền và nghĩa vụ:\n\n```\n{content}\n```\n\n1. Quyền lợi\n2. Nghĩa vụ\n3. Thủ tục\n4. Rủi ro pháp lý",
    },
]


# ── HTML → Markdown ──────────────────────────────────────────────────

def html_to_text(html: str) -> str:
    """Convert legal document HTML to clean structured text."""
    if not html:
        return ""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = []
    for line in text.splitlines():
        line = unescape(line).strip()
        line = re.sub(r"[\xa0\u200b]+", " ", line).strip()
        if line:
            lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(lines)).strip()


# ── Cached Document Loader ──────────────────────────────────────────

def load_documents(limit=None, min_len=200, max_len=30000):
    """Load with local parquet cache. First run downloads + converts."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / "documents.parquet"

    if cache_file.exists():
        print(f"Cache hit: {cache_file}", flush=True)
        docs = pq.read_table(cache_file).to_pylist()
        docs = [d for d in docs if min_len <= len(d["text"]) <= max_len]
        if limit:
            docs = docs[:limit]
        print(f"  {len(docs)} docs", flush=True)
        return docs

    # Download
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    print("Loading metadata…", flush=True)
    meta_ds = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data")
    meta = {str(r["id"]): dict(r) for r in meta_ds}
    print(f"  {len(meta)} entries", flush=True)

    print("Downloading content…", flush=True)
    path = hf_hub_download(
        "th1nhng0/vietnamese-legal-documents", "data/content.parquet", repo_type="dataset",
    )
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    print(f"  {total} rows → converting HTML to Markdown…", flush=True)

    docs = []
    loaded = 0
    for batch in pf.iter_batches(batch_size=2000):
        for doc_id, html in zip(batch.column("id").to_pylist(), batch.column("content_html").to_pylist()):
            m = meta.get(str(doc_id))
            if not m:
                continue
            t = html_to_text(html)
            if not t or len(t) < 50:
                continue
            docs.append({
                "id": str(doc_id), "text": t,
                "title": m.get("title") or "", "loai_van_ban": m.get("loai_van_ban") or "",
                "so_ky_hieu": m.get("so_ky_hieu") or "", "co_quan_ban_hanh": m.get("co_quan_ban_hanh") or "",
                "ngay_ban_hanh": m.get("ngay_ban_hanh") or "", "pham_vi": m.get("pham_vi") or "",
                "tinh_trang_hieu_luc": m.get("tinh_trang_hieu_luc") or "",
                "linh_vuc": m.get("linh_vuc") or "", "nganh": m.get("nganh") or "",
            })
        loaded += len(batch)
        print(f"  {loaded}/{total} → {len(docs)} valid", flush=True)

    # Save cache
    print(f"Caching {len(docs)} docs…", flush=True)
    pq.write_table(pa.Table.from_pylist(docs), cache_file)

    docs = [d for d in docs if min_len <= len(d["text"]) <= max_len]
    if limit:
        docs = docs[:limit]
    print(f"  → {len(docs)} docs", flush=True)
    return docs


# ── LLM Client ──────────────────────────────────────────────────────

class LLM:
    def __init__(self, models):
        self.models = models
        self._t = {}
        self._dead = set()

    def call(self, msgs, idx=0):
        for att in range(8):
            i = (idx + att) % len(self.models)
            m = self.models[i]
            if m["name"] in self._dead:
                continue
            self._throttle(m)
            try:
                r = self._api(msgs, m)
                self._t[m["name"]] = time.time()
                return r, i
            except requests.exceptions.HTTPError as e:
                self._t[m["name"]] = time.time()
                c = e.response.status_code if e.response else 0
                if c == 429:
                    w = min(2 ** (att + 2), 120)
                    print(f" ⚠️429({w}s)", end="", flush=True)
                    time.sleep(w)
                elif c == 503:
                    print(" ⚠️503", end="", flush=True)
                    time.sleep(5)
                elif c == 402:
                    self._dead.add(m["name"])
                elif c in (400, 401, 403):
                    return None, i
                else:
                    time.sleep(10)
            except Exception:
                time.sleep(5)
        return None, idx

    def _throttle(self, m):
        gap = 60.0 / m.get("rpm", 10)
        wait = gap - (time.time() - self._t.get(m["name"], 0)) + 0.1
        if wait > 0:
            print(f" ⏳{wait:.0f}s", end="", flush=True)
            time.sleep(wait)

    def _api(self, msgs, m):
        if m["provider"] == "openrouter":
            url, key = "https://openrouter.ai/api/v1/chat/completions", os.environ["OPENROUTER_API_KEY"]
        else:
            url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com") + "/v1/chat/completions"
            key = os.environ["NVIDIA_API_KEY"]
        r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json={"model": m["name"], "messages": msgs, "max_tokens": 4096, "temperature": 0.7}, timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ── Checkpoint ──────────────────────────────────────────────────────

def ckpt_append(recs):
    with open(CKPT, "a") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def ckpt_done_ids():
    if not CKPT.exists():
        return set()
    return {json.loads(l)["source_id"] for l in CKPT.read_text().splitlines() if l.strip()}

def ckpt_load():
    if not CKPT.exists():
        return []
    return [json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]


# ── Helpers ─────────────────────────────────────────────────────────

def pick_models():
    ov = os.environ.get("MODEL_LIST")
    if ov:
        out = []
        for entry in ov.split(","):
            entry = entry.strip()
            if "/" not in entry:
                continue
            p, n = entry.split("/", 1)
            out.append({"name": n, "provider": p, "ctx": 131072, "rpm": 10})
        return out
    has_or, has_nv = bool(os.environ.get("OPENROUTER_API_KEY")), bool(os.environ.get("NVIDIA_API_KEY"))
    return [m for m in MODELS if (m["provider"] == "openrouter" and has_or) or (m["provider"] == "nvidia" and has_nv)]


def meta_header(doc):
    parts = [f"Loại văn bản: {doc.get('loai_van_ban') or 'Không rõ'}", f"Tiêu đề: {doc.get('title') or ''}"]
    for k, l in [("so_ky_hieu","Số/Ký hiệu"),("co_quan_ban_hanh","Cơ quan ban hành"),
                  ("ngay_ban_hanh","Ngày ban hành"),("pham_vi","Phạm vi"),
                  ("tinh_trang_hieu_luc","Trạng thái"),("linh_vuc","Lĩnh vực"),("nganh","Ngành")]:
        if doc.get(k):
            parts.append(f"{l}: {doc[k]}")
    return "\n".join(parts)


def upload(repo, private, variant=None):
    """Upload checkpoint to HF. If variant is set, uploads as a named config/subset."""
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi
    recs = ckpt_load()
    if not recs:
        print("No records!")
        return

    tag = f" ({variant})" if variant else ""
    print(f"Uploading {len(recs)} records → {repo}{tag}…", flush=True)
    ds = Dataset.from_list(recs)
    split = ds.train_test_split(test_size=0.05, seed=42)

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=private)

    if variant:
        # Upload as a named config: variant/train and variant/test parquet files
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        import tempfile

        config_dir = variant
        for split_name, split_ds in split.items():
            buf = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
            split_ds.to_parquet(buf.name)
            path_in_repo = f"{config_dir}/{split_name}/{split_name}-00000-of-00001.parquet"
            api.upload_file(path_or_fileobj=buf.name, path_in_repo=path_in_repo,
                            repo_id=repo, repo_type="dataset")
            buf.close()
            print(f"  → {path_in_repo}", flush=True)

        # Update README with config metadata if not present
        _update_dataset_card(api, repo, variant, len(recs))
    else:
        split.push_to_hub(repo, private=private, token=token)

    print(f"Done! https://huggingface.co/datasets/{repo}")


def _update_dataset_card(api, repo, variant, num_records):
    """Ensure the dataset card lists available configs."""
    try:
        card = api.repo_info(repo, repo_type="dataset").card_data or {}
        # Just log — the README in the repo is the source of truth
        print(f"  Config '{variant}' uploaded ({num_records} records)", flush=True)
    except Exception:
        pass


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Vietnamese Legal → Unsloth Dataset")
    ap.add_argument("--generate", action="store_true", help="Use LLMs for QA pairs (default: simple convert)")
    ap.add_argument("--upload", type=str, default=None, help="Upload to HF dataset repo")
    ap.add_argument("--resume", action="store_true", help="Continue from checkpoint")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Max documents to process")
    ap.add_argument("--qa-types", type=int, default=2, help="QA variations per doc (generate mode)")
    ap.add_argument("--min-length", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=30000)
    ap.add_argument("--variant", type=str, default=None,
                    help="Named config for multi-variant upload (e.g. '10k', '50k', 'full')")
    ap.add_argument("--clear-cache", action="store_true", help="Delete local cache and exit")
    args = ap.parse_args()

    if args.clear_cache:
        import shutil
        shutil.rmtree(CACHE, ignore_errors=True)
        print("Cache cleared.")
        return

    # Load docs (cached after first run)
    docs = load_documents(limit=args.limit, min_len=args.min_length, max_len=args.max_length)

    if args.generate:
        # ── LLM Generation Mode ──
        models = pick_models()
        if not models:
            print("❌ Set OPENROUTER_API_KEY or NVIDIA_API_KEY in .env.local")
            sys.exit(1)
        print(f"\nModels ({len(models)}):")
        for m in models:
            print(f"  {m['provider']:12} {m['name']:<55} rpm={m['rpm']}")

        done = ckpt_done_ids() if args.resume else set()
        if done:
            docs = [d for d in docs if d["id"] not in done]
            print(f"Resuming: {len(done)} done, {len(docs)} remaining", flush=True)

        if not docs:
            print("Nothing to process!")
            if args.upload:
                upload(args.upload, args.private, args.variant)
            return
        llm = LLM(models)
        mi, batch, ok, fail = 0, [], 0, 0
        print(f"\n▶ {len(docs)} docs × {args.qa_types} QA\n", flush=True)

        for i, doc in enumerate(docs):
            text = doc["text"]
            cap = models[mi % len(models)]["ctx"] - 6000
            if len(text) > cap:
                text = text[:cap] + "\n\n…(cắt ngắn)"

            qas = random.sample(PROMPTS, min(args.qa_types, len(PROMPTS)))
            for qa in qas:
                user_msg = qa["user"].format(content=text)
                msgs = [{"role": "system", "content": qa["system"]}, {"role": "user", "content": user_msg}]
                print(f"  [{i+1}/{len(docs)}] {qa['type']:<18}", end="", flush=True)
                resp, mi = llm.call(msgs, mi)

                if not resp:
                    print(" ❌", flush=True)
                    fail += 1
                    continue
                print(f" ✅ {len(resp):>5}c", flush=True)
                batch.append({
                    "source_id": doc["id"], "document_type": doc.get("loai_van_ban") or "",
                    "qa_type": qa["type"],
                    "conversations": [
                        {"role": "system", "content": qa["system"]},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": resp},
                    ],
                })
                ok += 1

            if batch and (i + 1) % 10 == 0:
                ckpt_append(batch)
                batch = []
                print(f"  💾 {ok} ok, {fail} fail", flush=True)

        if batch:
            ckpt_append(batch)
        print(f"\n✅ {ok} ok, {fail} fail", flush=True)

    else:
        # ── Simple Convert Mode (no LLM) ──
        done = ckpt_done_ids() if args.resume else set()
        if done:
            docs = [d for d in docs if d["id"] not in done]
        if not docs:
            print("Nothing to process!")
            if args.upload:
                upload(args.upload, args.private, args.variant)
            return

        print(f"\nConverting {len(docs)} docs…", flush=True)
        batch = []
        for i, doc in enumerate(docs):
            batch.append({
                "source_id": doc["id"], "document_type": doc.get("loai_van_ban") or "",
                "qa_type": "full_text",
                "conversations": [
                    {"role": "user", "content": f"Cho văn bản pháp luật sau:\n\n{meta_header(doc)}\n\nHãy trình bày nội dung đầy đủ."},
                    {"role": "assistant", "content": doc["text"]},
                ],
            })
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(docs)}", flush=True)

        ckpt_append(batch)
        print(f"  {len(batch)} records saved", flush=True)

    # Upload
    if args.upload:
        upload(args.upload, args.private, args.variant)


if __name__ == "__main__":
    main()
