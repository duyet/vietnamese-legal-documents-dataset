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

Instruction-following dataset converted from [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) — a comprehensive collection of ~150K Vietnamese legal documents from [vbpl.vn](https://vbpl.vn) (Government Legal Document Portal, Ministry of Justice).

## Format

Unsloth-compatible conversation format (`conversations` column with `role`/`content`):

```json
{
  "conversations": [
    {"role": "system", "content": "Bạn là chuyên gia pháp luật Việt Nam..."},
    {"role": "user", "content": "Tóm tắt văn bản sau..."},
    {"role": "assistant", "content": "Văn bản này quy định về..."}
  ],
  "source_id": 12345,
  "document_type": "Nghị định",
  "qa_type": "summarize"
}
```

## QA Types

| Type | Description |
|------|-------------|
| `full_text` | Full document text (convert mode) |
| `summarize` | Summarize in 3-5 sentences |
| `key_provisions` | Key provisions and important articles |
| `qa_practical` | Practical Q&A from document content |
| `explain_simple` | Plain language explanation |
| `scope` | Scope, applicability, effective dates |
| `rights` | Rights and obligations analysis |

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
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
```

## Document Types

Covers all major Vietnamese legal document types:
- Luật (Laws), Nghị quyết (Resolutions), Sắc lệnh (Ordinances)
- Nghị định (Decrees), Thông tư (Circulars), Quyết định (Decisions)
- Chỉ thị (Directives), Thông báo (Notices), and more

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
  note      = {Converted from th1nhng0/vietnamese-legal-documents for instruction fine-tuning}
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
