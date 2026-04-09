"""
Generate instruction-following conversations from Vietnamese legal documents
using free LLM APIs (OpenRouter / NVIDIA).

Features:
- Auto-detects available providers from env keys
- Model rotation with rate limit handling and exponential backoff
- Incremental checkpointing (safe to interrupt/resume)
- Configurable via .env.local

Usage:
  python generate.py --limit 100                    # generate 100 conversations
  python generate.py --limit 1000 --resume          # resume from checkpoint
  python generate.py --upload duyet/vietnamese-legal-instruct
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
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env.local")

CHECKPOINT_FILE = Path(__file__).parent / "checkpoint.jsonl"

# Default models — ranked by quality for Vietnamese text generation.
# Override via MODEL_LIST env var (comma-separated provider/model pairs).
# Example: MODEL_LIST="nvidia/google/gemma-4-31b-it,nvidia/qwen/qwen3.5-397b-a17b"
DEFAULT_MODELS = [
    # ── OpenRouter free ──
    {"name": "google/gemma-4-31b-it:free",           "provider": "openrouter", "context": 262144, "rpm": 10},
    {"name": "google/gemma-4-26b-a4b-it:free",       "provider": "openrouter", "context": 262144, "rpm": 10},
    {"name": "qwen/qwen3-next-80b-a3b-instruct:free","provider": "openrouter", "context": 262144, "rpm": 10},
    {"name": "qwen/qwen3-coder:free",                "provider": "openrouter", "context": 262000, "rpm": 10},
    {"name": "nvidia/nemotron-3-super-120b-a12b:free","provider": "openrouter", "context": 262144, "rpm": 10},
    {"name": "minimax/minimax-m2.5:free",             "provider": "openrouter", "context": 196608, "rpm": 10},
    {"name": "google/gemma-3-27b-it:free",            "provider": "openrouter", "context": 131072, "rpm": 20},
    {"name": "meta-llama/llama-3.3-70b-instruct:free","provider": "openrouter", "context": 65536,  "rpm": 20},
    # ── NVIDIA API ──
    {"name": "google/gemma-4-31b-it",                 "provider": "nvidia", "context": 131072, "rpm": 10},
    {"name": "qwen/qwen3.5-397b-a17b",               "provider": "nvidia", "context": 131072, "rpm": 5},
    {"name": "qwen/qwen3.5-122b-a10b",               "provider": "nvidia", "context": 131072, "rpm": 10},
    {"name": "meta/llama-4-maverick-17b-128e-instruct","provider": "nvidia", "context": 131072, "rpm": 10},
    {"name": "google/gemma-3-27b-it",                 "provider": "nvidia", "context": 131072, "rpm": 10},
    {"name": "deepseek-ai/deepseek-v3.2",             "provider": "nvidia", "context": 131072, "rpm": 5},
]

# ── QA Prompt Templates ─────────────────────────────────────────────

QA_PROMPTS = [
    {
        "type": "summarize",
        "system": "Bạn là trợ lý pháp luật Việt Nam. Hãy tóm tắt chính xác, ngắn gọn bằng tiếng Việt. Giữ nguyên các thuật ngữ pháp lý quan trọng.",
        "user_template": (
            "Dưới đây là nội dung một văn bản pháp luật Việt Nam:\n\n"
            "---\n{content}\n---\n\n"
            "Hãy tóm tắt văn bản trên trong 3-5 câu, bao gồm:\n"
            "- Mục đích chính của văn bản\n"
            "- Các đối tượng chịu tác động\n"
            "- Những quy định quan trọng nhất"
        ),
    },
    {
        "type": "key_provisions",
        "system": "Bạn là chuyên gia pháp luật Việt Nam. Hãy phân tích và liệt kê chi tiết các điều khoản quan trọng. Trả lời bằng tiếng Việt.",
        "user_template": (
            "Phân tích văn bản pháp luật sau và liệt kê các quy định chính:\n\n"
            "---\n{content}\n---\n\n"
            "Hãy:\n"
            "1. Liệt kê các điều/khoản quan trọng nhất\n"
            "2. Giải thích ngắn gọn ý nghĩa của từng điều\n"
            "3. Nhận xét về phạm vi áp dụng"
        ),
    },
    {
        "type": "qa_practical",
        "system": "Bạn là luật sư tư vấn pháp luật Việt Nam. Hãy trả lời câu hỏi thực tế dựa trên văn bản pháp luật được cung cấp. Trả lời bằng tiếng Việt, chính xác và có trích dẫn điều khoản khi cần.",
        "user_template": (
            "Dựa vào văn bản pháp luật sau:\n\n"
            "---\n{content}\n---\n\n"
            "Hãy trả lời các câu hỏi thực tế sau:\n"
            "1. Đối tượng/person nào phải tuân thủ văn bản này?\n"
            "2. Nghĩa vụ cụ thể của từng đối tượng là gì?\n"
            "3. Hình thức xử lý vi phạm (nếu có)?\n"
            "4. Cơ quan nào có thẩm quyền thực thi?"
        ),
    },
    {
        "type": "explain_simple",
        "system": "Bạn là người phiên dịch pháp luật. Hãy giải thích văn bản pháp luật bằng ngôn ngữ đơn giản, dễ hiểu cho người bình thường. Dùng ví dụ thực tế nếu cần.",
        "user_template": (
            "Hãy giải thích văn bản pháp luật sau bằng ngôn ngữ đơn giản, như đang giải thích cho một người không có kiến thức pháp lý:\n\n"
            "---\n{content}\n---\n\n"
            "Vui lòng:\n"
            "- Dùng từ ngữ đơn giản, tránh thuật ngữ pháp lý phức tạp\n"
            "- Cho ví dụ thực tế nếu có thể\n"
            "- Tóm tắt quyền và nghĩa vụ của người dân liên quan"
        ),
    },
    {
        "type": "scope_application",
        "system": "Bạn là chuyên gia pháp luật hành chính Việt Nam. Hãy phân tích phạm vi áp dụng và hiệu lực của văn bản pháp luật.",
        "user_template": (
            "Phân tích phạm vi áp dụng của văn bản pháp luật sau:\n\n"
            "---\n{content}\n---\n\n"
            "Hãy xác định:\n"
            "1. Phạm vi đối tượng áp dụng\n"
            "2. Phạm vi địa lý áp dụng\n"
            "3. Thời hiệu lực của văn bản\n"
            "4. Các văn bản có liên quan được dẫn chiếu\n"
            "5. Điều kiện để áp dụng văn bản này"
        ),
    },
    {
        "type": "rights_obligations",
        "system": "Bạn là chuyên gia tư vấn pháp luật Việt Nam. Hãy phân tích chi tiết quyền và nghĩa vụ của các bên liên quan.",
        "user_template": (
            "Từ văn bản pháp luật sau:\n\n"
            "---\n{content}\n---\n\n"
            "Hãy phân tích chi tiết:\n"
            "1. Quyền lợi của tổ chức/cá nhân theo văn bản\n"
            "2. Nghĩa vụ của tổ chức/cá nhân theo văn bản\n"
            "3. Trình tự, thủ tục thực hiện (nếu có)\n"
            "4. Lợi ích và rủi ro pháp lý cần lưu ý"
        ),
    },
]


# ── HTML Cleaning ────────────────────────────────────────────────────

def strip_html(html: str) -> str:
    if not html:
        return ""
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
    text = "\n\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── LLM Client ──────────────────────────────────────────────────────

class LLMClient:
    """Unified LLM client with rate limit handling and model rotation."""

    def __init__(self, models: list[dict]):
        self.models = models
        self.last_request_time: dict[str, float] = {}
        self.dead_models: set[str] = set()  # models that returned 402

    def _wait_for_rate_limit(self, model: dict):
        rpm = model.get("rpm", 10)
        min_interval = 60.0 / rpm
        last = self.last_request_time.get(model["name"], 0)
        elapsed = time.time() - last
        if elapsed < min_interval:
            wait = min_interval - elapsed + 0.1
            print(f"    ⏳ {wait:.1f}s rate limit wait", flush=True)
            time.sleep(wait)

    def call(self, messages: list[dict], model_idx: int = 0, max_retries: int = 6) -> tuple[str | None, int]:
        """Call LLM, return (response_text, model_index_used)."""
        for attempt in range(max_retries):
            # Rotate through models starting from model_idx
            idx = (model_idx + attempt) % len(self.models)
            model = self.models[idx]

            if model["name"] in self.dead_models:
                continue

            self._wait_for_rate_limit(model)

            try:
                result = self._api_call(messages, model)
                self.last_request_time[model["name"]] = time.time()
                return result, idx
            except requests.exceptions.HTTPError as e:
                self.last_request_time[model["name"]] = time.time()
                status = e.response.status_code if e.response else 0

                if status == 429:
                    wait = min(2 ** (attempt + 2), 120)
                    print(f"    ⚠️ 429 rate limit, {wait}s backoff", flush=True)
                    time.sleep(wait)
                elif status == 503:
                    print(f"    ⚠️ 503 overloaded, trying next model", flush=True)
                    time.sleep(5)
                elif status == 402:
                    print(f"    ❌ {model['name']} no longer free", flush=True)
                    self.dead_models.add(model["name"])
                elif status in (400, 401, 403):
                    print(f"    ❌ {status}: {e.response.text[:200]}", flush=True)
                    return None, idx
                else:
                    print(f"    ⚠️ HTTP {status}, retry {attempt+1}/{max_retries}", flush=True)
                    time.sleep(10)
            except Exception as e:
                print(f"    ⚠️ {type(e).__name__}: {e}", flush=True)
                time.sleep(5)

        return None, model_idx

    def _api_call(self, messages: list[dict], model: dict) -> str | None:
        provider = model["provider"]
        if provider == "openrouter":
            return self._openrouter(messages, model)
        elif provider == "nvidia":
            return self._nvidia(messages, model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _openrouter(self, messages: list[dict], model: dict) -> str:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set")
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model["name"], "messages": messages, "max_tokens": 4096, "temperature": 0.7},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _nvidia(self, messages: list[dict], model: dict) -> str:
        key = os.environ.get("NVIDIA_API_KEY")
        base = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com")
        if not key:
            raise ValueError("NVIDIA_API_KEY not set")
        resp = requests.post(
            f"{base}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model["name"], "messages": messages, "max_tokens": 4096, "temperature": 0.7},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Data Loading ────────────────────────────────────────────────────

def load_documents(limit=None, min_length=100, max_length=30000):
    """Load and join metadata + content, streaming to control memory."""
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    print("Loading metadata...", flush=True)
    meta_ds = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data")
    meta_map = {str(row["id"]): dict(row) for row in meta_ds}
    print(f"  {len(meta_map)} metadata entries", flush=True)

    print("Streaming content...", flush=True)
    content_path = hf_hub_download(
        "th1nhng0/vietnamese-legal-documents", "data/content.parquet", repo_type="dataset",
    )

    documents = []
    pf = pq.ParquetFile(content_path)
    for batch in pf.iter_batches(batch_size=1000):
        ids = batch.column("id").to_pylist()
        htmls = batch.column("content_html").to_pylist()
        for doc_id, html in zip(ids, htmls):
            if limit and len(documents) >= limit:
                break
            meta = meta_map.get(str(doc_id))
            if not meta:
                continue
            text = strip_html(html)
            if not text or len(text) < min_length or len(text) > max_length:
                continue
            documents.append({"id": doc_id, "meta": meta, "text": text})
        if limit and len(documents) >= limit:
            break

    print(f"  {len(documents)} documents loaded", flush=True)
    return documents


# ── Checkpoint ──────────────────────────────────────────────────────

def save_checkpoint(records: list[dict], path: Path = CHECKPOINT_FILE):
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_processed_ids(path: Path = CHECKPOINT_FILE) -> set:
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line).get("source_id"))
    return ids


def load_all_records(path: Path = CHECKPOINT_FILE) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ── Model Selection ─────────────────────────────────────────────────

def get_available_models() -> list[dict]:
    """Filter DEFAULT_MODELS by available API keys. Override with MODEL_LIST env."""
    env_override = os.environ.get("MODEL_LIST")
    if env_override:
        models = []
        for entry in env_override.split(","):
            entry = entry.strip()
            if "/" not in entry:
                continue
            provider, name = entry.split("/", 1)
            models.append({"name": name, "provider": provider, "context": 131072, "rpm": 10})
        return models

    available = []
    has_or = bool(os.environ.get("OPENROUTER_API_KEY"))
    has_nv = bool(os.environ.get("NVIDIA_API_KEY"))

    for m in DEFAULT_MODELS:
        if (m["provider"] == "openrouter" and has_or) or (m["provider"] == "nvidia" and has_nv):
            available.append(m)
    return available


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate legal QA conversations")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--upload", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--qa-types", type=int, default=2, help="QA variations per document")
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=30000)
    args = parser.parse_args()

    # Models
    models = get_available_models()
    if not models:
        print("❌ No API keys or models configured!")
        print("   Set OPENROUTER_API_KEY or NVIDIA_API_KEY in .env.local")
        print("   Or set MODEL_LIST=provider/model,provider/model")
        sys.exit(1)

    print(f"Available models ({len(models)}):")
    for m in models:
        print(f"  {m['provider']:12} {m['name']:<55} ctx={m['context']:>8} rpm={m['rpm']}")
    print()

    # Load data
    processed_ids = get_processed_ids() if args.resume else set()
    print(f"Previously processed: {len(processed_ids)}", flush=True)

    documents = load_documents(limit=None, min_length=args.min_length, max_length=args.max_length)

    # Filter already processed
    if processed_ids:
        before = len(documents)
        documents = [d for d in documents if d["id"] not in processed_ids]
        print(f"Skipped {before - len(documents)} already-processed", flush=True)

    if args.limit:
        documents = documents[:args.limit]

    if not documents:
        print("No documents to process!")
        if args.upload:
            print("Uploading existing records...")
            _upload(args)
        sys.exit(0)

    print(f"\nProcessing {len(documents)} documents × {args.qa_types} QA types = ~{len(documents) * args.qa_types} calls\n", flush=True)

    # Generate
    client = LLMClient(models)
    model_idx = 0
    batch = []
    total = 0
    failed = 0

    for i, doc in enumerate(documents):
        text = doc["text"]
        meta = doc["meta"]

        # Truncate to fit model context
        model = models[model_idx % len(models)]
        max_input = model["context"] - 6000
        if len(text) > max_input:
            text = text[:max_input] + "\n\n...(nội dung được cắt ngắn)"

        qa_types = random.sample(QA_PROMPTS, min(args.qa_types, len(QA_PROMPTS)))

        for qa in qa_types:
            user_msg = qa["user_template"].format(content=text)
            messages = [
                {"role": "system", "content": qa["system"]},
                {"role": "user", "content": user_msg},
            ]

            print(f"  [{i+1}/{len(documents)}] {qa['type']:<20} ", end="", flush=True)
            response, model_idx = client.call(messages, model_idx)

            if not response:
                print("❌ all models failed", flush=True)
                failed += 1
                continue

            print(f"✅ {len(response):>5} chars via {models[model_idx]['name'][:40]}", flush=True)

            batch.append({
                "source_id": doc["id"],
                "document_type": meta.get("loai_van_ban") or "",
                "qa_type": qa["type"],
                "conversations": [
                    {"role": "system", "content": qa["system"]},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": response},
                ],
            })
            total += 1

        # Checkpoint every 10 docs
        if batch and (i + 1) % 10 == 0:
            save_checkpoint(batch)
            print(f"  💾 checkpoint: {total} records, {failed} failed\n", flush=True)
            batch = []

    # Save remaining
    if batch:
        save_checkpoint(batch)

    print(f"\n✅ Done: {total} generated, {failed} failed", flush=True)

    # Upload
    if args.upload:
        _upload(args)


def _upload(args):
    from datasets import Dataset
    records = load_all_records()
    if not records:
        print("No records to upload!")
        return
    print(f"Uploading {len(records)} records to {args.upload}...")
    ds = Dataset.from_list(records)
    split = ds.train_test_split(test_size=0.05, seed=42)
    split.push_to_hub(args.upload, private=args.private, token=os.environ.get("HF_TOKEN"))
    print(f"Done! https://huggingface.co/datasets/{args.upload}")


if __name__ == "__main__":
    main()
