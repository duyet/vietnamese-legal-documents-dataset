---
language:
- vi
license: cc-by-4.0
task_categories:
- text-generation
- conversational
tags:
- vietnamese
- legal
- instruction-tuning
- unsloth
- gemma
size_categories:
- 100K<n<1M
---

# Vietnamese Legal Instruction Dataset

Converted from [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) for Unsloth-compatible fine-tuning.

## Format

Each record contains a `conversations` column with instruction-following pairs:

```json
{
  "conversations": [
    {"role": "user", "content": "Cho văn bản pháp luật sau:\n\nLoại văn bản: Nghị định\nTiêu đề: ...\n\nHãy trình bày nội dung đầy đủ của văn bản."},
    {"role": "assistant", "content": "<cleaned legal document text>"}
  ],
  "source_id": 12345,
  "document_type": "Nghị định"
}
```

## Usage with Unsloth

```python
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt, get_chat_template

dataset = load_dataset("duyet/vietnamese-legal-instruct", split="train")

# Apply Gemma chat template
tokenizer = get_chat_template(tokenizer, chat_template="gemma")

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
```

## Source

- Original dataset: [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents)
- Source: vbpl.vn (Vietnamese Government Legal Document Portal)
- License: CC BY 4.0

## Conversion Details

- HTML content stripped to clean plain text
- Metadata joined with content on document ID
- Filtered: content length 50-50,000 characters
- Split: 95% train / 5% test
