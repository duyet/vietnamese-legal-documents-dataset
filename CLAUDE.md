# Vietnamese Legal Documents → Unsloth Instruction Dataset

Convert [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) to Unsloth-compatible format for fine-tuning Gemma 3/4 models.

**Repo**: [github.com/duyet/vietnamese-legal-documents-dataset](https://github.com/duyet/vietnamese-legal-documents-dataset)
**Dataset**: [huggingface.co/datasets/duyet/vietnamese-legal-instruct](https://huggingface.co/datasets/duyet/vietnamese-legal-instruct)

## Single Script: generate.py

| Mode | Command | Description |
|------|---------|-------------|
| Convert | `python generate.py --upload <repo>` | Join metadata+content, no LLM |
| Generate | `python generate.py --generate --limit 100` | LLM QA pairs via NVIDIA/OpenRouter |
| Resume | `python generate.py --generate --resume --limit 1000` | Continue from checkpoint |
| Upload | `python generate.py --resume --upload <repo>` | Upload checkpoint to HF |
| Fresh | `python generate.py --fresh --upload <repo>` | Delete checkpoint and regenerate all |
| Clear cache | `python generate.py --clear-cache` | Delete local parquet cache |

## Architecture

- **Cache**: First run downloads + strips HTML → saves `cache/documents.duckdb` (156K docs). Re-runs are instant.
- **Checkpoint**: `checkpoint.jsonl` — append-only, safe to Ctrl+C. `--resume` skips already-processed docs.
- **Auto-detect**: Reads `.env.local` for API keys, filters models by available providers.
- **Rate limits**: Per-model RPM tracking, exponential backoff on 429/503, auto-rotation on failure.

## Output Format

Unsloth-compatible `conversations` with `role`/`content`:
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

## QA Types (14 variations)

| Type | System Prompt | Focus |
|------|---------------|-------|
| `summarize` | Legal expert | 3-5 sentence summary |
| `key_provisions` | Legal analyst | Key articles and provisions |
| `qa_practical` | Lawyer | Practical Q&A |
| `explain_simple` | Public translator | Plain language for non-lawyers |
| `scope` | Administrative law expert | Scope, applicability, effectivity |
| `classify` | Classification expert | Document type and hierarchy position |
| `legal_basis` | Legal researcher | Legal basis chain analysis |
| `amounts` | Lawyer | Monetary amounts and percentages |
| `full_text` | Legal expert | Full document text (1 per doc, always) |
| `meta_type` | Classifier | Document type + hierarchy level |
| `meta_issuer` | Legal expert | Issuing authority + date |
| `meta_date` | Admin expert | Issue date + effective date + status |
| `meta_title` | Legal expert | Title + subject + field |
| `meta_status` | Admin expert | Current legal status |

## Models

| Provider | Model | Context | RPM |
|----------|-------|---------|-----|
| OpenRouter | google/gemma-4-31b-it:free | 262K | 10 |
| OpenRouter | qwen/qwen3-next-80b-a3b-instruct:free | 262K | 10 |
| OpenRouter | google/gemma-3-27b-it:free | 131K | 20 |
| NVIDIA | google/gemma-4-31b-it | 131K | 10 |
| NVIDIA | qwen/qwen3.5-397b-a17b | 131K | 5 |
| NVIDIA | meta/llama-4-maverick-17b-128e-instruct | 131K | 10 |
| NVIDIA | deepseek-ai/deepseek-v3.2 | 131K | 5 |

Override: `MODEL_LIST=openrouter/google/gemma-4-31b-it:free,nvidia/google/gemma-4-31b-it`

## Fine-tuning with Unsloth

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

## Env Setup

```bash
cp .env.example .env.local
# Edit .env.local — at least one of:
#   NVIDIA_API_KEY=nvapi-...
#   OPENROUTER_API_KEY=sk-or-...
#   HF_TOKEN=hf_... (for upload)
```
