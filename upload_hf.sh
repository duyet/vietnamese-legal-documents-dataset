#!/bin/bash
# Upload all project files to the HF dataset repo
# Usage: ./upload_hf.sh

REPO="duyet/vietnamese-legal-instruct"
FILES="README.md CLAUDE.md generate.py .env.example .gitignore"

for f in $FILES; do
  echo "Uploading $f..."
  python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HF_TOKEN') or open('.env.local').read().split('HF_TOKEN=')[1].strip())
api.upload_file(path_or_fileobj='$f', path_in_repo='$f', repo_id='$REPO', repo_type='dataset')
print('  done')
"
done
echo "All uploaded → https://huggingface.co/datasets/$REPO"
