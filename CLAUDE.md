# Vietnamese Legal Documents - Instruction Dataset Generator

Convert [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) to Unsloth-compatible instruction format for fine-tuning Gemma models.

## Scripts

| Script | Purpose |
|--------|---------|
| `convert.py` | Simple conversion: join metadata + content, strip HTML, output conversation format |
| `generate.py` | LLM-powered generation: uses free OpenRouter/NVIDIA APIs to create diverse QA pairs |

## Quick Start

### Option A: Simple Conversion (no LLM needed)
```bash
# Preview
python convert.py --max-samples 100

# Full conversion + upload
HF_TOKEN=your_token python convert.py --upload duyet/vietnamese-legal-instruct
```

### Option B: LLM-Generated QA Pairs
```bash
# Set up API keys in .env.local
cp .env.example .env.local
# Edit .env.local with your keys

# Generate 100 conversations
python generate.py --limit 100 --qa-types 2

# Resume if interrupted
python generate.py --limit 1000 --resume

# Upload when done
HF_TOKEN=your_token python generate.py --limit 0 --resume --upload duyet/vietnamese-legal-instruct
```

## Output Format

Unsloth-compatible conversation format:
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

## Models

| Provider | Model | Context | RPM |
|----------|-------|---------|-----|
| OpenRouter | google/gemma-3-27b-it:free | 128k | 20 |
| OpenRouter | qwen/qwen3-32b:free | 128k | 20 |
| OpenRouter | deepseek/deepseek-r1-0528:free | 160k | 10 |
| NVIDIA | google/gemma-3-27b-it | 128k | 10 |
| NVIDIA | meta/llama-4-maverick-17b-128e-instruct | 128k | 10 |

## Files

| File | Description |
|------|-------------|
| `.env.local` | API keys (gitignored) |
| `checkpoint.jsonl` | Incremental save of generated records |
| `output.jsonl` | Final output file |

## Fine-tuning with Unsloth

```python
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import load_dataset

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4B-it",
    max_seq_length=2048,
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
