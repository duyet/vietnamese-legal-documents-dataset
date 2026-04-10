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
- gemma
- gemma-3
- gemma-4
- fine-tuning
- law
size_categories:
- 100K<n<1M
---

# Vietnamese Legal Instruction Dataset

Instruction-following dataset built from [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) — 127K Vietnamese legal documents from [vbpl.vn](https://vbpl.vn) (Government Legal Document Portal, Ministry of Justice).

**233,866 training pairs** across 9 QA types with deep Vietnamese legal hierarchy knowledge.

## Statistics

| Metric | Value |
|--------|-------|
| Total records | 233,866 |
| Source documents | 116,933 (from 127,271 unique, filtered by length) |
| QA types | 9 |
| Train split | 222,173 (95%) |
| Test split | 11,693 (5%) |

### QA Type Distribution

| Type | Count | Description |
|------|------:|-------------|
| `summarize` | 42,502 | 3-5 sentence structured summary |
| `qa_practical` | 35,488 | Practical compliance Q&A |
| `explain_simple` | 35,334 | Plain language for non-lawyers |
| `classify` | 27,143 | Document type & hierarchy position |
| `scope` | 27,085 | Scope, applicability, effective dates |
| `key_provisions` | 26,163 | Key articles and provisions |
| `legal_basis` | 21,636 | Legal basis chain analysis |
| `full_text` | 9,401 | Full document text |
| `amounts` | 9,114 | Monetary amounts & percentages |

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

Source code: [`generate.py`](https://huggingface.co/datasets/duyet/vietnamese-legal-instruct/blob/main/generate.py)

Built with 9 local QA generators (no LLM API calls needed):
- Vietnamese legal hierarchy knowledge baked into system prompts
- Quality-filtered: min 100 char responses, no within-doc duplicates
- DuckDB-backed cache for memory-efficient processing (~145 MB RSS)

```bash
# Reproduce the dataset
pip install requests python-dotenv beautifulsoup4 lxml pyarrow datasets duckdb
python generate.py --qa-types 2 --upload duyet/vietnamese-legal-instruct
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
  url       = {https://huggingface.co/datasets/duyet/vietnamese-legal-instruct},
  note      = {233K instruction pairs from 127K Vietnamese legal documents, 9 QA types}
}

@dataset{thinhngo_vietnamese_legal_2025,
  title     = {Vietnamese Legal Documents},
  author    = {Thinh Ngo},
  year      = {2025},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents},
  note      = {Source: vbpl.vn, Ministry of Justice, Vietnam}
}
```
