"""
Convert th1nhng0/vietnamese-legal-documents to Unsloth-compatible instruction format.

Optimized: parallel HTML stripping with multiprocessing.

Usage:
  python convert.py                          # preview only
  python convert.py --upload duyet/vietnamese-legal-instruct
  python convert.py --workers 8              # parallel processing (default: CPU count)
"""

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from html import unescape
from multiprocessing import cpu_count

import pyarrow.parquet as pq
from bs4 import BeautifulSoup
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download


def strip_html(html: str) -> str:
    """Convert legal document HTML to clean plain text."""
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


def build_user_prompt(meta: dict) -> str:
    parts = [f"Loại văn bản: {meta.get('loai_van_ban') or 'Không rõ'}"]
    parts.append(f"Tiêu đề: {meta.get('title') or ''}")
    for key, label in [
        ("so_ky_hieu", "Số/Ký hiệu"),
        ("co_quan_ban_hanh", "Cơ quan ban hành"),
        ("ngay_ban_hanh", "Ngày ban hành"),
        ("pham_vi", "Phạm vi"),
        ("tinh_trang_hieu_luc", "Trạng thái hiệu lực"),
        ("linh_vuc", "Lĩnh vực"),
        ("nganh", "Ngành"),
    ]:
        if meta.get(key):
            parts.append(f"{label}: {meta[key]}")
    header = "\n".join(parts)
    return f"Cho văn bản pháp luật sau:\n\n{header}\n\nHãy trình bày nội dung đầy đủ của văn bản."


def process_batch(batch_items: list[tuple]) -> list[dict]:
    """Process a batch of (doc_id, html, meta, min_len, max_len) tuples."""
    results = []
    for doc_id, html, meta, min_len, max_len in batch_items:
        clean = strip_html(html)
        if not clean or len(clean) < min_len or len(clean) > max_len:
            continue
        results.append({
            "id": doc_id,
            "meta": meta,
            "text": clean,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Convert Vietnamese legal dataset")
    parser.add_argument("--upload", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=cpu_count(),
                        help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    # 1. Load metadata
    print("Loading metadata...", flush=True)
    meta_ds = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data")
    meta_map = {str(row["id"]): dict(row) for row in meta_ds}
    print(f"  {len(meta_map)} entries", flush=True)

    # 2. Stream content, collect (id, html, meta) for matching docs
    print("Downloading content...", flush=True)
    content_path = hf_hub_download(
        "th1nhng0/vietnamese-legal-documents", "data/content.parquet", repo_type="dataset",
    )
    pf = pq.ParquetFile(content_path)
    total_content = pf.metadata.num_rows
    print(f"  {total_content} content rows", flush=True)

    # Collect all matching items
    print("Joining metadata + content...", flush=True)
    all_items = []
    for batch in pf.iter_batches(batch_size=5000):
        ids = batch.column("id").to_pylist()
        htmls = batch.column("content_html").to_pylist()
        for doc_id, html in zip(ids, htmls):
            meta = meta_map.get(str(doc_id))
            if meta:
                all_items.append((str(doc_id), html, meta, args.min_length, args.max_length))
        print(f"  {len(all_items)} matched so far...", flush=True)

    print(f"  Total matched: {len(all_items)}", flush=True)

    if args.max_samples:
        all_items = all_items[:args.max_samples]

    # 3. Parallel HTML stripping
    workers = args.workers
    print(f"\nStripping HTML with {workers} workers...", flush=True)

    # Split into chunks for parallel processing
    chunk_size = max(100, len(all_items) // (workers * 4))
    chunks = [all_items[i:i + chunk_size] for i in range(0, len(all_items), chunk_size)]

    documents = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_batch, chunk) for chunk in chunks]
        for future in as_completed(futures):
            results = future.result()
            documents.extend(results)
            done += 1
            if done % 5 == 0 or done == len(chunks):
                print(f"  {len(documents)} docs converted ({done}/{len(chunks)} chunks)", flush=True)

    print(f"\n  Valid documents: {len(documents)}", flush=True)

    if not documents:
        print("No documents!")
        sys.exit(1)

    # 4. Build conversation records
    print("Building conversations...", flush=True)
    records = []
    for doc in documents:
        records.append({
            "conversations": [
                {"role": "user", "content": build_user_prompt(doc["meta"])},
                {"role": "assistant", "content": doc["text"]},
            ],
            "source_id": int(doc["id"]),
            "document_type": doc["meta"].get("loai_van_ban") or "",
        })

    # 5. Sample
    print("\n" + "=" * 60)
    sample = records[0]
    for msg in sample["conversations"]:
        print(f"\n[{msg['role'].upper()}]:\n{msg['content'][:300]}...")

    # 6. Split and stats
    print("\nCreating dataset...", flush=True)
    ds = Dataset.from_list(records)
    split = ds.train_test_split(test_size=0.05, seed=42)
    print(f"  Train: {len(split['train'])}")
    print(f"  Test:  {len(split['test'])}")

    types = {}
    for r in records:
        t = r["document_type"] or "Khác"
        types[t] = types.get(t, 0) + 1
    print("\nDocument types (top 15):")
    for t, c in sorted(types.items(), key=lambda x: -x[1])[:15]:
        print(f"  {t}: {c}")

    # 7. Upload
    if args.upload:
        print(f"\nUploading to {args.upload}...", flush=True)
        split.push_to_hub(args.upload, private=args.private, token=os.environ.get("HF_TOKEN"))
        print(f"Done! https://huggingface.co/datasets/{args.upload}")
    else:
        print("\nUse --upload <repo> to push to HuggingFace")


if __name__ == "__main__":
    main()
