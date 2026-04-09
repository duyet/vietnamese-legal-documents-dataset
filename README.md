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
size_categories:
- 100K<n<1M
---

# Vietnamese Legal Instruction Dataset

Instruction-following dataset converted from [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) — a comprehensive collection of ~150K Vietnamese legal documents from [vbpl.vn](https://vbpl.vn) (Government Legal Document Portal).

## Format

Unsloth-compatible conversation format with `conversations` column:

```json
{
  "conversations": [
    {"role": "system", "content": "Bạn là trợ lý pháp luật Việt Nam..."},
    {"role": "user", "content": "Hãy tóm tắt văn bản pháp luật sau..."},
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
| `summarize` | Summarize the document in 3-5 sentences |
| `key_provisions` | List the key provisions and important articles |
| `qa` | Answer questions based on the document content |
| `explain_simple` | Explain in simple language for non-lawyers |
| `scope` | Identify scope, applicability, and effective dates |

## Usage with Unsloth

```python
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset

# Load model
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4B-it",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Apply Gemma chat template
tokenizer = get_chat_template(tokenizer, chat_template="gemma")

# Load dataset
dataset = load_dataset("duyet/vietnamese-legal-instruct", split="train")

# Format conversations into training text
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
- Nghị định (Decrees), Thông tư (Circulars), Quyết định (Decisions)
- Luật (Laws), Nghị quyết (Resolutions), Sắc lệnh (Ordinances)
- Chỉ thị (Directives), Thông báo (Notices), and more

## Source

- **Original dataset**: [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents)
- **Data source**: [vbpl.vn](https://vbpl.vn) — Vietnamese Ministry of Justice
- **Curated by**: Thịnh Ngô
- **License**: CC BY 4.0

## Generation

Generated using free LLM APIs (Google Gemma 3 27B via NVIDIA/OpenRouter) with:
- HTML content cleaned to plain text
- Diverse QA prompts for each document
- Rate limit handling with automatic model rotation
- Incremental checkpointing for reliability
