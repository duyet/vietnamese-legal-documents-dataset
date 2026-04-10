"""
Vietnamese Legal Documents → Unsloth Instruction Dataset

Single script: downloads, caches (DuckDB), generates QA pairs, uploads to HF.

9 QA types with Vietnamese legal hierarchy knowledge:
  summarize, explain, practical, provisions, classify, scope, legal_basis, amounts, full_text

Usage:
  python generate.py                                      # Build cache + preview
  python generate.py --upload duyet/vietnamese-legal-instruct
  python generate.py --generate --limit 100               # LLM-powered QA
  python generate.py --generate --resume --upload duyet/vietnamese-legal-instruct
  python generate.py --clear-cache
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
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

# ── Vietnamese Legal Hierarchy ───────────────────────────────────────

DOC_HIERARCHY = {
    "Luật":               {"level": 1, "issuer": "Quốc hội", "desc": "Văn bản quy phạm pháp luật có hiệu lực cao nhất sau Hiến pháp.", "scope": "toàn quốc"},
    "Bộ luật":            {"level": 1, "issuer": "Quốc hội", "desc": "Hệ thống quy phạm điều chỉnh toàn diện một lĩnh vực (hình sự, dân sự, lao động).", "scope": "toàn quốc"},
    "Pháp lệnh":          {"level": 2, "issuer": "UBTVQH", "desc": "Do UBTVQH ban hành, có hiệu lực cho đến khi Quốc hội ban hành Luật thay thế.", "scope": "toàn quốc"},
    "Lệnh":               {"level": 2, "issuer": "Chủ tịch nước", "desc": "Do Chủ tịch nước ban hành để công bố Luật, Pháp lệnh.", "scope": "toàn quốc"},
    "Sắc lệnh":           {"level": 2, "issuer": "Chủ tịch nước (trước 1992)", "desc": "Văn bản pháp luật giai đoạn đầu lập nước.", "scope": "toàn quốc"},
    "Nghị định":           {"level": 3, "issuer": "Chính phủ", "desc": "Quy định chi tiết, hướng dẫn thi hành Luật, Pháp lệnh.", "scope": "toàn quốc"},
    "Nghị quyết":          {"level": 3, "issuer": "Quốc hội / Chính phủ / HĐND", "desc": "Thể hiện chủ trương, chính sách, kế hoạch phát triển.", "scope": "tùy cấp"},
    "Nghị Quyết":          {"level": 3, "issuer": "Quốc hội / Chính phủ / HĐND", "desc": "Chủ trương, chính sách.", "scope": "tùy cấp"},
    "Quyết định":          {"level": 4, "issuer": "Thủ tướng / Bộ trưởng / Chủ tịch UBND", "desc": "Giải quyết công việc cụ thể, áp dụng trực tiếp.", "scope": "tùy cấp"},
    "Thông tư":            {"level": 4, "issuer": "Bộ trưởng / cơ quan ngang Bộ", "desc": "Hướng dẫn chi tiết thi hành Nghị định, quy định chuyên ngành.", "scope": "toàn quốc trong lĩnh vực"},
    "Thông tư liên tịch":  {"level": 4, "issuer": "Nhiều Bộ phối hợp", "desc": "Hướng dẫn thi hành vấn đề liên ngành.", "scope": "toàn quốc"},
    "Chỉ thị":             {"level": 5, "issuer": "Thủ tướng / Chủ tịch UBND", "desc": "Chỉ đạo, đôn đốc cấp dưới. Từ 2016 không còn là VBQPPL.", "scope": "tùy cấp"},
}

SYSTEM_PROMPTS = {
    "expert": "Bạn là chuyên gia pháp luật Việt Nam với kinh nghiệm phân tích văn bản quy phạm pháp luật. Trả lời chính xác, có căn cứ, giữ thuật ngữ pháp lý. Tiếng Việt.",
    "analyst": "Bạn là chuyên gia phân tích pháp luật Việt Nam. Trình bày có hệ thống, rõ ràng, phân tích sâu. Tiếng Việt.",
    "lawyer": "Bạn là luật sư tư vấn pháp luật Việt Nam. Trả lời chính xác, trích dẫn điều khoản cụ thể, tư vấn thực tiễn. Tiếng Việt.",
    "translator": "Bạn phiên dịch pháp luật cho người dân. Giải thích đơn giản, dùng ngôn ngữ phổ thông, lấy ví dụ thực tế. Tiếng Việt.",
    "admin": "Bạn là chuyên gia pháp luật hành chính Việt Nam. Phân tích phạm vi áp dụng, thẩm quyền, hiệu lực. Tiếng Việt.",
    "classifier": "Bạn là chuyên gia phân loại và hệ thống hóa văn bản pháp luật Việt Nam. Xác định chính xác loại, cấp, vị trí trong hệ thống. Tiếng Việt.",
    "researcher": "Bạn là nhà nghiên cứu pháp luật Việt Nam. Phân tích căn cứ pháp lý, mối liên hệ giữa các văn bản. Tiếng Việt.",
}

# ── LLM QA Prompts (for --generate mode) ─────────────────────────────

LLM_PROMPTS = [
    {"type": "summarize", "system": SYSTEM_PROMPTS["expert"],
     "user": "Tóm tắt văn bản sau trong 3-5 câu (mục đích, đối tượng, quy định chính, phạm vi):\n\n```\n{content}\n```"},
    {"type": "key_provisions", "system": SYSTEM_PROMPTS["analyst"],
     "user": "Phân tích các quy định chính:\n\n```\n{content}\n```\n\n1. Các điều/khoản quan trọng\n2. Ý nghĩa pháp lý\n3. Phạm vi áp dụng"},
    {"type": "qa_practical", "system": SYSTEM_PROMPTS["lawyer"],
     "user": "Từ văn bản:\n\n```\n{content}\n```\n\n1. Ai phải tuân thủ?\n2. Nghĩa vụ cụ thể?\n3. Xử lý vi phạm?\n4. Cơ quan thẩm quyền?"},
    {"type": "explain_simple", "system": SYSTEM_PROMPTS["translator"],
     "user": "Giải thích cho người không biết pháp luật:\n\n```\n{content}\n```\n\nDùng từ đơn giản, ví dụ thực tế, tóm tắt quyền/nghĩa vụ."},
    {"type": "scope", "system": SYSTEM_PROMPTS["admin"],
     "user": "Phạm vi áp dụng:\n\n```\n{content}\n```\n\n1. Đối tượng\n2. Phạm vi địa lý\n3. Thời gian hiệu lực\n4. Văn bản liên quan"},
    {"type": "rights", "system": SYSTEM_PROMPTS["researcher"],
     "user": "Quyền và nghĩa vụ:\n\n```\n{content}\n```\n\n1. Quyền lợi\n2. Nghĩa vụ\n3. Thủ tục\n4. Rủi ro pháp lý"},
]


# ── Text Extraction ──────────────────────────────────────────────────

def html_to_text(html):
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


def extract_articles(text):
    return re.findall(r"(Điều\s+\d+[a-z]?\.?\s*[^\n]*)", text)

def extract_legal_bases(text):
    return [m.strip() for m in re.findall(r"(Căn cứ\s+[^;]+)", text) if len(m.strip()) > 20]

def extract_amounts(text):
    return re.findall(r"(\d[\d.,]*\s*(?:đồng|đ|VND|triệu|tỷ)[^\n]{0,80})", text)[:8]

def extract_percentages(text):
    return re.findall(r"(\d+(?:[.,]\d+)?%[^\n]{0,80})", text)[:5]

def extract_effective_date(text):
    for p in [r"có hiệu lực.*?(?:từ|kể từ)\s+(?:ngày\s+)?([^\n.;]{5,40})",
              r"hiệu lực thi hành.*?(?:từ|kể từ)\s+(?:ngày\s+)?([^\n.;]{5,40})"]:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def count_articles(text):
    return len(re.findall(r"Điều\s+\d+", text))

def truncate(text, n=8000):
    return text if len(text) <= n else text[:n] + "\n\n…(nội dung còn lại được lược bỏ)"

def infer_scope(doc):
    organ = (doc.get("co_quan_ban_hanh") or "").lower()
    if any(k in organ for k in ["quốc hội", "chính phủ", "thủ tướng", "chủ tịch nước"]):
        return "toàn quốc"
    if "bộ " in organ:
        return "toàn quốc trong lĩnh vực quản lý của Bộ"
    if "tỉnh" in organ or "thành phố" in organ:
        return f"phạm vi địa phương ({doc.get('co_quan_ban_hanh', '')})"
    return "theo thẩm quyền cơ quan ban hành"

def meta_block(doc):
    parts = []
    for key, label in [("loai_van_ban","Loại văn bản"),("title","Tiêu đề"),("so_ky_hieu","Số/Ký hiệu"),
                        ("co_quan_ban_hanh","Cơ quan ban hành"),("ngay_ban_hanh","Ngày ban hành"),
                        ("linh_vuc","Lĩnh vực"),("tinh_trang_hieu_luc","Tình trạng hiệu lực")]:
        if doc.get(key):
            parts.append(f"{label}: {doc[key]}")
    return "\n".join(parts)

def _explain_doc_type(doc_type):
    m = {"Luật": "một bộ luật do Quốc hội thông qua", "Nghị định": "quy định chi tiết do Chính phủ ban hành",
         "Quyết định": "văn bản hành chính giải quyết việc cụ thể", "Thông tư": "hướng dẫn chi tiết do Bộ ban hành",
         "Nghị quyết": "quyết định chính sách", "Pháp lệnh": "do UBTVQH ban hành, gần như luật",
         "Chỉ thị": "văn bản chỉ đạo từ cấp trên", "Sắc lệnh": "văn bản pháp luật thời kỳ đầu lập nước"}
    return m.get(doc_type, "một văn bản pháp luật")


# ── 9 QA Generators (local, no LLM needed) ───────────────────────────

def gen_summarize(doc):
    text, title, doc_type = doc["text"], doc.get("title",""), doc.get("loai_van_ban","văn bản")
    articles, bases, n_art, eff = extract_articles(text), extract_legal_bases(text), count_articles(text), extract_effective_date(text)
    parts = [f"Đây là {doc_type} số {doc.get('so_ky_hieu','')} do {doc.get('co_quan_ban_hanh','')} ban hành ngày {doc.get('ngay_ban_hanh','')}."]
    if title: parts.append(f"Nội dung: {title.strip().rstrip('.')}.")
    if bases: parts.append(f"Căn cứ {len(bases)} văn bản pháp lý, trong đó có {bases[0]}." if len(bases) > 1 else f"Căn cứ {bases[0]}.")
    if n_art > 0: parts.append(f"Cấu trúc gồm {n_art} điều.")
    if articles and len(articles) >= 2:
        key_arts = [a.strip() for a in articles[:3] if len(a.strip()) > 15]
        if key_arts: parts.append(f"Nội dung chính: {'; '.join(key_arts)}.")
    if eff: parts.append(f"Có hiệu lực từ {eff}.")
    if doc.get("tinh_trang_hieu_luc"): parts.append(f"Tình trạng: {doc['tinh_trang_hieu_luc']}.")
    return {"qa_type": "summarize", "conversations": [
        {"role": "system", "content": SYSTEM_PROMPTS["expert"]},
        {"role": "user", "content": f"Tóm tắt văn bản pháp luật sau trong 3-5 câu:\n\n{truncate(text, 6000)}"},
        {"role": "assistant", "content": " ".join(parts)}]}

def gen_provisions(doc):
    text, articles = doc["text"], extract_articles(doc["text"])
    if len(articles) < 3: return None
    bases, n_art, title = extract_legal_bases(text), count_articles(text), doc.get("title","")
    parts = [f"## Phân tích \"{title}\"\n", f"### 1. Các điều khoản chính ({n_art} điều)"]
    for a in articles[:10]: parts.append(f"- {a.strip()}")
    if bases:
        parts.append(f"\n### 2. Cơ sở pháp lý ({len(bases)} văn bản)")
        for b in bases[:6]: parts.append(f"- {b.strip()}")
    parts.append(f"\n### 3. Phạm vi\n- Cơ quan: {doc.get('co_quan_ban_hanh','')}\n- Phạm vi: {infer_scope(doc)}")
    eff = extract_effective_date(text)
    if eff: parts.append(f"- Hiệu lực từ: {eff}")
    return {"qa_type": "key_provisions", "conversations": [
        {"role": "system", "content": SYSTEM_PROMPTS["analyst"]},
        {"role": "user", "content": f"Phân tích các quy định chính:\n\n{truncate(text, 6000)}"},
        {"role": "assistant", "content": "\n".join(parts)}]}

def gen_practical(doc):
    text, articles, amounts = doc["text"], extract_articles(doc["text"]), extract_amounts(doc["text"])
    title, doc_type, organ = doc.get("title",""), doc.get("loai_van_ban","văn bản"), doc.get("co_quan_ban_hanh","")
    parts = [f"## Tư vấn: {doc_type} \"{title}\"\n", "### 1. Ai phải tuân thủ?"]
    parts.append(f"Các tổ chức, cá nhân thuộc phạm vi điều chỉnh trong {infer_scope(doc)}.")
    parts.append("\n### 2. Nghĩa vụ")
    for a in amounts[:4]: parts.append(f"- {a.strip()}")
    oblig = [a for a in articles if any(kw in a.lower() for kw in ["phải","trách nhiệm","nghĩa vụ","cấm"])]
    for a in oblig[:3]: parts.append(f"- {a.strip()}")
    parts.append(f"\n### 3. Cơ quan thẩm quyền\n- {organ}")
    viols = [a for a in articles if any(kw in a.lower() for kw in ["vi phạm","xử phạt","xử lý"])]
    if viols:
        parts.append("\n### 4. Xử lý vi phạm")
        for a in viols[:3]: parts.append(f"- {a.strip()}")
    eff = extract_effective_date(text)
    if eff: parts.append(f"\n### Hiệu lực: từ {eff}")
    if doc.get("tinh_trang_hieu_luc"): parts.append(f"Tình trạng: {doc['tinh_trang_hieu_luc']}")
    return {"qa_type": "qa_practical", "conversations": [
        {"role": "system", "content": SYSTEM_PROMPTS["lawyer"]},
        {"role": "user", "content": f"Phân tích thực tiễn:\n\n{truncate(text, 5000)}"},
        {"role": "assistant", "content": "\n".join(parts)}]}

def gen_explain(doc):
    text, title, doc_type = doc["text"], doc.get("title",""), doc.get("loai_van_ban","văn bản")
    amounts, articles = extract_amounts(text), extract_articles(text)
    parts = ["## Giải thích đơn giản\n", "**Đây là gì?**", f"Đây là {doc_type} — {_explain_doc_type(doc_type)}.",
             f"Do {doc.get('co_quan_ban_hanh','')} ban hành."]
    if title: parts.append(f"Nội dung: {title.strip()}.\n")
    parts.append(f"**Ai cần quan tâm?**\nÁp dụng {infer_scope(doc)}.\n")
    if amounts:
        parts.append("**Mức tiền/phí:**")
        for a in amounts[:4]: parts.append(f"- {a.strip()}")
    if articles and len(articles) >= 2:
        parts.append("\n**Nội dung chính:**")
        for a in articles[:5]: parts.append(f"- {a.strip()}")
    status = doc.get("tinh_trang_hieu_luc","")
    if status:
        parts.append(f"\n**Còn hiệu lực?** {'⚠️ ' + status + '. Đã hết hiệu lực.' if 'hết hiệu lực' in status.lower() else '✅ ' + status}")
    return {"qa_type": "explain_simple", "conversations": [
        {"role": "system", "content": SYSTEM_PROMPTS["translator"]},
        {"role": "user", "content": f"Giải thích đơn giản:\n\n{truncate(text, 4000)}"},
        {"role": "assistant", "content": "\n".join(parts)}]}

def gen_classify(doc):
    doc_type = doc.get("loai_van_ban","")
    if doc_type not in DOC_HIERARCHY: return None
    info = DOC_HIERARCHY[doc_type]
    hl = {1:"cấp Luật",2:"cấp Pháp lệnh/Lệnh",3:"cấp Nghị định",4:"cấp Thông tư/Quyết định",5:"cấp Chỉ thị"}
    mk = {i: ("  ← **văn bản này**" if i == info["level"] else "") for i in range(1,6)}
    answer = f"""## Phân loại văn bản\n\n**Loại:** {doc_type}\n**Số:** {doc.get('so_ky_hieu','')}\n**Cơ quan:** {doc.get('co_quan_ban_hanh',info['issuer'])}\n\n{info['desc']}\n\n**Thứ bậc:**\n1. Hiến pháp\n2. Luật, Bộ luật{mk[1]}\n3. Pháp lệnh, Lệnh{mk[2]}\n4. Nghị định, Nghị quyết{mk[3]}\n5. Thông tư, Quyết định{mk[4]}\n6. Chỉ thị{mk[5]}\n\nVăn bản này thuộc {hl[info['level']]}. Phạm vi: {info['scope']}."""
    return {"qa_type": "classify", "conversations": [
        {"role": "system", "content": SYSTEM_PROMPTS["classifier"]},
        {"role": "user", "content": f"Phân loại văn bản:\n\n{truncate(doc['text'], 3000)}"},
        {"role": "assistant", "content": answer}]}

def gen_scope(doc):
    text, title, bases, eff = doc["text"], doc.get("title",""), extract_legal_bases(doc["text"]), extract_effective_date(doc["text"])
    parts = [f"## Phạm vi áp dụng: \"{title}\"\n", "### 1. Đối tượng"]
    sm = re.search(r"(?:Phạm vi|Đối tượng)[^\n]*\n([^\n]+(?:\n[^\n]+){0,5})", text, re.IGNORECASE)
    parts.append(sm.group(0).strip() if sm else f"Các tổ chức, cá nhân thuộc phạm vi điều chỉnh.")
    parts.append(f"\n### 2. Địa lý\n- {infer_scope(doc)}")
    parts.append(f"\n### 3. Hiệu lực")
    if doc.get("ngay_ban_hanh"): parts.append(f"- Ban hành: {doc['ngay_ban_hanh']}")
    if eff: parts.append(f"- Hiệu lực từ: {eff}")
    if doc.get("tinh_trang_hieu_luc"): parts.append(f"- Tình trạng: {doc['tinh_trang_hieu_luc']}")
    if bases:
        parts.append(f"\n### 4. Văn bản liên quan ({len(bases)})")
        for b in bases[:6]: parts.append(f"- {b.strip()}")
    return {"qa_type": "scope", "conversations": [
        {"role": "system", "content": SYSTEM_PROMPTS["admin"]},
        {"role": "user", "content": f"Phạm vi áp dụng:\n\n{truncate(text, 5000)}"},
        {"role": "assistant", "content": "\n".join(parts)}]}

def gen_legal_basis(doc):
    bases = extract_legal_bases(doc["text"])
    if len(bases) < 2: return None
    parts = [f"## Cơ sở pháp lý\n**\"{doc.get('title','')}\"**\n\nViện dẫn {len(bases)} căn cứ:\n"]
    for i, b in enumerate(bases, 1):
        bl = b.lower()
        level = "Luật" if ("luật " in bl or "bộ luật" in bl) else "Nghị định" if "nghị định" in bl else "Thông tư" if "thông tư" in bl else "Khác"
        parts.append(f"**{i}.** [{level}] {b.strip()}")
    lc = Counter()
    for b in bases:
        bl = b.lower()
        if "luật " in bl or "bộ luật" in bl: lc["Luật"] += 1
        elif "nghị định" in bl: lc["Nghị định"] += 1
    parts.append(f"\n### Đánh giá")
    if lc.get("Luật"): parts.append(f"✅ Căn cứ trực tiếp từ {lc['Luật']} Luật.")
    parts.append(f"Tổng {len(bases)} văn bản viện dẫn → {'cơ sở pháp lý vững chắc' if len(bases) >= 4 else 'đủ cơ sở'}.")
    return {"qa_type": "legal_basis", "conversations": [
        {"role": "system", "content": SYSTEM_PROMPTS["researcher"]},
        {"role": "user", "content": f"Phân tích căn cứ pháp lý:\n\n{truncate(doc['text'], 4000)}"},
        {"role": "assistant", "content": "\n".join(parts)}]}

def gen_amounts(doc):
    amounts, pcts = extract_amounts(doc["text"]), extract_percentages(doc["text"])
    if not amounts and not pcts: return None
    title, doc_type = doc.get("title",""), doc.get("loai_van_ban","")
    parts = [f"## Số liệu trong {doc_type} \"{title}\"\n"]
    if amounts:
        parts.append("### Mức tiền")
        for a in amounts: parts.append(f"- {a.strip()}")
    if pcts:
        parts.append("\n### Tỷ lệ %")
        for p in pcts: parts.append(f"- {p.strip()}")
    parts.append(f"\n### Lưu ý\n- Theo {doc_type} ({doc.get('so_ky_hieu','')}).")
    status = doc.get("tinh_trang_hieu_luc","")
    if "hết hiệu lực" in status.lower(): parts.append(f"- ⚠️ {status}")
    return {"qa_type": "amounts", "conversations": [
        {"role": "system", "content": SYSTEM_PROMPTS["lawyer"]},
        {"role": "user", "content": f"Trích xuất số liệu:\n\n{truncate(doc['text'], 5000)}"},
        {"role": "assistant", "content": "\n".join(parts)}]}

def gen_full_text(doc):
    return {"qa_type": "full_text", "conversations": [
        {"role": "system", "content": SYSTEM_PROMPTS["expert"]},
        {"role": "user", "content": f"Trình bày nội dung đầy đủ:\n\n{meta_block(doc)}"},
        {"role": "assistant", "content": doc["text"]}]}

# Generator registry with weights
GENERATORS = [
    (gen_summarize, 5), (gen_provisions, 4), (gen_practical, 4), (gen_explain, 4),
    (gen_scope, 3), (gen_classify, 3), (gen_legal_basis, 3), (gen_amounts, 2), (gen_full_text, 1),
]

def generate_for_doc(doc, qa_count=2):
    pool = []
    for fn, w in GENERATORS:
        pool.extend([fn] * w)
    random.shuffle(pool)
    results, seen = [], set()
    for fn in pool:
        if len(results) >= qa_count: break
        try:
            rec = fn(doc)
        except Exception:
            continue
        if rec is None or rec["qa_type"] in seen: continue
        # Quality filter: min 100 chars in assistant response
        assistant = rec["conversations"][-1]["content"]
        if len(assistant) < 100: continue
        seen.add(rec["qa_type"])
        rec["source_id"] = doc["id"]
        rec["document_type"] = doc.get("loai_van_ban", "")
        results.append(rec)
    return results


# ── DuckDB Cache ────────────────────────────────────────────────────

META_FIELDS = ["id","title","loai_van_ban","so_ky_hieu","co_quan_ban_hanh",
               "ngay_ban_hanh","pham_vi","tinh_trang_hieu_luc","linh_vuc","nganh"]

def _db_path():
    return CACHE / "documents.duckdb"

def _db(readonly=True):
    import duckdb
    return duckdb.connect(str(_db_path()), read_only=readonly)

def build_cache():
    import duckdb
    CACHE.mkdir(exist_ok=True)
    db = _db_path()
    if db.exists():
        con = duckdb.connect(str(db), read_only=True)
        n = con.execute("SELECT count(*) FROM docs").fetchone()[0]
        con.close()
        print(f"Cache: {n:,} docs", flush=True)
        return
    import pyarrow as pa, pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    print("Loading metadata…", flush=True)
    meta_ds = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data").select_columns(META_FIELDS)
    meta = {str(r["id"]): r for r in meta_ds}
    print(f"  {len(meta):,} entries", flush=True)
    del meta_ds
    print("Downloading content…", flush=True)
    path = hf_hub_download("th1nhng0/vietnamese-legal-documents", "data/content.parquet", repo_type="dataset")
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    print(f"  {total:,} rows → stripping HTML…", flush=True)
    tmp = CACHE / "tmp.parquet"
    writer, loaded, written = None, 0, 0
    for batch in pf.iter_batches(batch_size=2000):
        rows = []
        for doc_id, html in zip(batch.column("id").to_pylist(), batch.column("content_html").to_pylist()):
            m = meta.get(str(doc_id))
            if not m: continue
            t = html_to_text(html)
            if not t or len(t) < 50: continue
            rows.append({"id": str(doc_id), "text": t, **{k: m.get(k) or "" for k in META_FIELDS if k != "id"}})
        if rows:
            tbl = pa.Table.from_pylist(rows)
            if writer is None: writer = pq.ParquetWriter(str(tmp), tbl.schema)
            writer.write_table(tbl)
            written += len(rows)
            del tbl
        loaded += len(batch)
        print(f"  {loaded:,}/{total:,} → {written:,} valid", flush=True)
    if writer: writer.close()
    del meta
    print("Building DuckDB…", flush=True)
    con = duckdb.connect(str(db))
    con.execute(f"""CREATE TABLE docs AS
        SELECT DISTINCT ON (id) *, length(text) as text_len
        FROM read_parquet('{tmp}') ORDER BY id, length(text) DESC""")
    con.execute("CREATE INDEX idx_id ON docs(id)")
    con.close()
    tmp.unlink(missing_ok=True)
    print(f"Cached {written:,} docs", flush=True)

def get_doc_ids(min_len=200, max_len=30000, limit=None, exclude=None):
    con = _db()
    ids = [r[0] for r in con.execute("SELECT id FROM docs WHERE text_len BETWEEN ? AND ? ORDER BY id", [min_len, max_len]).fetchall()]
    con.close()
    if exclude: ids = [i for i in ids if i not in exclude]
    if limit: ids = ids[:limit]
    return ids

def read_docs(doc_ids):
    if not doc_ids: return {}
    con = _db()
    ph = ",".join(["?"] * len(doc_ids))
    rows = con.execute(f"SELECT * FROM docs WHERE id IN ({ph})", doc_ids).fetchdf().to_dict("records")
    con.close()
    return {r["id"]: r for r in rows}


# ── LLM Client ──────────────────────────────────────────────────────

class LLM:
    def __init__(self, models):
        self.models, self._t, self._dead = models, {}, set()

    def call(self, msgs, idx=0):
        for att in range(8):
            i = (idx + att) % len(self.models)
            m = self.models[i]
            if m["name"] in self._dead: continue
            self._throttle(m)
            try:
                r = self._api(msgs, m)
                self._t[m["name"]] = time.time()
                return r, i
            except requests.exceptions.HTTPError as e:
                self._t[m["name"]] = time.time()
                c = e.response.status_code if e.response else 0
                if c == 429: time.sleep(min(2 ** (att + 2), 120))
                elif c == 503: time.sleep(5)
                elif c == 402: self._dead.add(m["name"])
                elif c in (400, 401, 403): return None, i
                else: time.sleep(10)
            except Exception: time.sleep(5)
        return None, idx

    def _throttle(self, m):
        w = 60.0 / m.get("rpm", 10) - (time.time() - self._t.get(m["name"], 0)) + 0.1
        if w > 0:
            print(f" ⏳{w:.0f}s", end="", flush=True)
            time.sleep(w)

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
    if not CKPT.exists(): return set()
    with open(CKPT) as f:
        return {json.loads(l)["source_id"] for l in f if l.strip()}

def _ckpt_iter():
    if not CKPT.exists(): return
    with open(CKPT) as f:
        for l in f:
            if l.strip(): yield json.loads(l)


# ── Helpers ─────────────────────────────────────────────────────────

def pick_models():
    ov = os.environ.get("MODEL_LIST")
    if ov:
        out = []
        for e in ov.split(","):
            e = e.strip()
            if "/" not in e: continue
            p, n = e.split("/", 1)
            out.append({"name": n, "provider": p, "ctx": 131072, "rpm": 10})
        return out
    has_or, has_nv = bool(os.environ.get("OPENROUTER_API_KEY")), bool(os.environ.get("NVIDIA_API_KEY"))
    return [m for m in MODELS if (m["provider"] == "openrouter" and has_or) or (m["provider"] == "nvidia" and has_nv)]


def upload(repo, private):
    from datasets import Dataset
    ds = Dataset.from_generator(_ckpt_iter)
    if len(ds) == 0:
        print("No records!"); return
    print(f"Uploading {len(ds):,} records → {repo}…", flush=True)
    split = ds.train_test_split(test_size=0.05, seed=42)
    split.push_to_hub(repo, private=private, token=os.environ.get("HF_TOKEN"))
    print(f"Done! https://huggingface.co/datasets/{repo}")


# ── Main ────────────────────────────────────────────────────────────

BATCH = 50

def main():
    ap = argparse.ArgumentParser(description="Vietnamese Legal → Unsloth Instruction Dataset")
    ap.add_argument("--generate", action="store_true", help="Use LLMs for QA (default: local generation)")
    ap.add_argument("--upload", type=str, default=None, help="Upload to HF repo")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qa-types", type=int, default=2, help="QA variations per doc")
    ap.add_argument("--min-length", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clear-cache", action="store_true")
    args = ap.parse_args()

    if args.clear_cache:
        import shutil
        shutil.rmtree(CACHE, ignore_errors=True); print("Cache cleared."); return

    random.seed(args.seed)
    build_cache()

    done = ckpt_done_ids() if args.resume else set()
    if done: print(f"Already done: {len(done):,}", flush=True)
    doc_ids = get_doc_ids(min_len=args.min_length, max_len=args.max_length, limit=args.limit, exclude=done or None)
    if not doc_ids:
        print("Nothing to process!")
        if args.upload: upload(args.upload, args.private)
        return
    print(f"{len(doc_ids):,} docs to process", flush=True)

    if args.generate:
        # ── LLM Generation ──
        models = pick_models()
        if not models: print("❌ Set OPENROUTER_API_KEY or NVIDIA_API_KEY in .env.local"); sys.exit(1)
        print(f"\nModels ({len(models)}):")
        for m in models: print(f"  {m['provider']:12} {m['name']:<55} rpm={m['rpm']}")
        llm = LLM(models)
        mi, batch, ok, fail, total = 0, [], 0, 0, len(doc_ids)
        print(f"\n▶ {total:,} docs × {args.qa_types} QA (LLM)\n", flush=True)
        for bs in range(0, total, BATCH):
            bids = doc_ids[bs:bs+BATCH]
            dm = read_docs(bids)
            for j, did in enumerate(bids):
                doc = dm.get(did)
                if not doc: continue
                i = bs + j
                text = doc["text"]
                cap = models[mi % len(models)]["ctx"] - 6000
                if len(text) > cap: text = text[:cap] + "\n\n…(cắt ngắn)"
                qas = random.sample(LLM_PROMPTS, min(args.qa_types, len(LLM_PROMPTS)))
                for qa in qas:
                    user_msg = qa["user"].format(content=text)
                    msgs = [{"role": "system", "content": qa["system"]}, {"role": "user", "content": user_msg}]
                    print(f"  [{i+1}/{total}] {qa['type']:<18}", end="", flush=True)
                    resp, mi = llm.call(msgs, mi)
                    if not resp or len(resp) < 100: print(" ❌", flush=True); fail += 1; continue
                    print(f" ✅ {len(resp):>5}c", flush=True)
                    batch.append({"source_id": doc["id"], "document_type": doc.get("loai_van_ban",""),
                                  "qa_type": qa["type"], "conversations": [
                                      {"role": "system", "content": qa["system"]},
                                      {"role": "user", "content": user_msg},
                                      {"role": "assistant", "content": resp}]})
                    ok += 1
                if batch and (i + 1) % 10 == 0: ckpt_append(batch); batch = []; print(f"  💾 {ok} ok, {fail} fail", flush=True)
            del dm
        if batch: ckpt_append(batch)
        print(f"\n✅ {ok:,} ok, {fail:,} fail", flush=True)

    else:
        # ── Local Generation (no LLM, instant) ──
        total = len(doc_ids)
        print(f"\n▶ {total:,} docs × {args.qa_types} QA (local)\n", flush=True)
        batch, n = [], 0
        for bs in range(0, total, BATCH * 10):
            bids = doc_ids[bs:bs + BATCH * 10]
            dm = read_docs(bids)
            for did in bids:
                doc = dm.get(did)
                if not doc: continue
                recs = generate_for_doc(doc, args.qa_types)
                batch.extend(recs)
                n += 1
            del dm
            if batch:
                ckpt_append(batch)
                print(f"  {min(bs + BATCH * 10, total):,}/{total:,} → {n:,} docs, {sum(1 for _ in _ckpt_iter()):,} records", flush=True)
                batch = []
        if batch: ckpt_append(batch)
        # Stats
        all_recs = list(_ckpt_iter())
        print(f"\n✅ {len(all_recs):,} records from {n:,} docs", flush=True)
        print(f"\n── QA Types ──")
        for k, v in Counter(r["qa_type"] for r in all_recs).most_common():
            print(f"  {k:<18} {v:>7,}")
        print(f"\n── Doc Types (top 10) ──")
        for k, v in Counter(r["document_type"] for r in all_recs).most_common(10):
            print(f"  {k:<25} {v:>7,}")

    if args.upload:
        upload(args.upload, args.private)


if __name__ == "__main__":
    main()
