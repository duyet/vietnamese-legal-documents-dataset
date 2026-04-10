---
language:
- vi
license: cc-by-4.0
task_categories:
- text-generation
- question-answering
tags:
- vietnamese
- legal
- instruction-tuning
- unsloth
- gemma-4
- fine-tuning
- law
size_categories:
- 100K<n<1M
---

# Vietnamese Legal Instruction Dataset

**Dataset**: [huggingface.co/datasets/duyet/vietnamese-legal-instruct](https://huggingface.co/datasets/duyet/vietnamese-legal-instruct) | **Source code**: [github.com/duyet/vietnamese-legal-documents-dataset](https://github.com/duyet/vietnamese-legal-documents-dataset)

Instruction-following dataset built from [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) — 127K Vietnamese legal documents from [vbpl.vn](https://vbpl.vn) (Government Legal Document Portal, Ministry of Justice).

**467,732 training pairs** across 14 QA types with deep Vietnamese legal hierarchy knowledge. Every document has a `full_text` pair for content recall, plus 5 short metadata recall types for memorization.

## Statistics

| Metric | Value |
|--------|-------|
| Total records | 467,732 |
| Source documents | 116,933 (from 127,271 unique, filtered by length) |
| QA types | 14 |
| Train split | 444,346 (95%) |
| Test split | 23,386 (5%) |

### QA Type Distribution

| Type | Count | % | Description |
|------|------:|--:|-------------|
| `full_text` | 116,933 | 25.0 | Full document content (1 per doc for content recall) |
| `scope` | 35,715 | 7.6 | Scope, applicability, effective dates |
| `classify` | 35,401 | 7.6 | Document type & hierarchy position |
| `summarize` | 35,226 | 7.5 | 3-5 sentence structured summary |
| `meta_date` | 29,247 | 6.3 | Issue date & effective date (short) |
| `explain_simple` | 29,176 | 6.2 | Plain language for non-lawyers |
| `meta_issuer` | 29,059 | 6.2 | Issuing authority (short) |
| `meta_title` | 29,049 | 6.2 | Title & subject (short) |
| `meta_type` | 29,013 | 6.2 | Document type & hierarchy level (short) |
| `qa_practical` | 28,982 | 6.2 | Practical compliance Q&A |
| `meta_status` | 22,201 | 4.7 | Current legal status (short) |
| `key_provisions` | 22,061 | 4.7 | Key articles and provisions |
| `legal_basis` | 17,867 | 3.8 | Legal basis chain analysis |
| `amounts` | 7,802 | 1.7 | Monetary amounts & percentages |

### Document Type Distribution (top 10)

| Document Type | Count |
|---------------|------:|
| Quyết định (Decisions) | 137,524 |
| Nghị quyết (Resolutions) | 40,404 |
| Thông tư (Circulars) | 23,544 |
| Chỉ thị (Directives) | 16,198 |
| Nghị định (Decrees) | 6,610 |
| Thông tư liên tịch (Joint Circulars) | 4,834 |
| Sắc lệnh (Ordinances) | 1,950 |
| Nghị Quyết | 864 |
| Lệnh (Orders) | 750 |
| Pháp lệnh (Ordinances) | 404 |

## Format

Unsloth-compatible conversation format (`conversations` column with `role`/`content`):

```json
{
  "conversations": [
    {"role": "system", "content": "Bạn là chuyên gia pháp luật Việt Nam..."},
    {"role": "user", "content": "Tóm tắt văn bản sau..."},
    {"role": "assistant", "content": "Văn bản này quy định về..."}
  ],
  "source_id": "12345",
  "document_type": "Nghị định",
  "qa_type": "summarize"
}
```

## Vietnamese Legal Hierarchy

The dataset encodes knowledge of the Vietnamese legal document hierarchy (per Luật ban hành VBQPPL 2015):

```
1. Hiến pháp (Constitution)
2. Luật, Bộ luật (Laws, Codes) — Quốc hội
3. Pháp lệnh, Lệnh (Ordinances, Orders) — UBTVQH / Chủ tịch nước
4. Nghị định, Nghị quyết (Decrees, Resolutions) — Chính phủ
5. Thông tư, Quyết định (Circulars, Decisions) — Bộ trưởng / UBND
6. Chỉ thị (Directives) — Thủ tướng / Chủ tịch UBND
```

## Quick Start with Unsloth

```python
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4B-it",
    max_seq_length=4096,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma")

dataset = load_dataset("duyet/vietnamese-legal-instruct", split="train")

def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in examples["conversations"]
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2, learning_rate=2e-4,
        packing=True,  # 2-5x speedup for mixed-length data
    ),
)
trainer.train()
```

## Generation

GitHub: [duyet/vietnamese-legal-documents-dataset](https://github.com/duyet/vietnamese-legal-documents-dataset) | Source code: [`generate.py`](https://huggingface.co/datasets/duyet/vietnamese-legal-instruct/blob/main/generate.py)

Built with 14 local QA generators (no LLM API calls needed):
- 9 analysis types: summarize, key_provisions, qa_practical, explain_simple, scope, classify, legal_basis, amounts, full_text
- 5 short metadata recall types: meta_type, meta_issuer, meta_date, meta_title, meta_status
- Vietnamese legal hierarchy knowledge baked into system prompts
- Quality-filtered: min 60 char responses, no within-doc duplicates
- Every doc gets 1 full_text + 3 random QA types = 4 records per doc
- DuckDB-backed cache for memory-efficient processing (~145 MB RSS)

```bash
# Reproduce the dataset
pip install requests python-dotenv beautifulsoup4 lxml pyarrow datasets duckdb
python generate.py --fresh --qa-types 3 --upload duyet/vietnamese-legal-instruct
```

## Source

- **Original dataset**: [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) by Thịnh Ngô
- **Data source**: [vbpl.vn](https://vbpl.vn) — Vietnamese Ministry of Justice
- **License**: CC BY 4.0

## Citation

```bibtex
@dataset{vietnamese_legal_instruct_2026,
  title     = {Vietnamese Legal Instruction Dataset},
  author    = {Duyet Le},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/8343},
  url       = {https://huggingface.co/datasets/duyet/vietnamese-legal-instruct},
  note      = {468K instruction pairs from 127K Vietnamese legal documents, 14 QA types}
}
```
