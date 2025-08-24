# WS-Lite

A tiny weak-supervision toolkit for text classification.

## Quickstart

```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
make reproduce
make ablation
make test
```

### What it does

- Applies label functions to text

- Builds a label matrix with abstentions

- Majority vote and one-step weighted vote

- Trains a logistic regression on denoised labels

- Reports test accuracy and leave-one-LF-out ablation

__NOTE__ : First run tries to download 20 Newsgroups; if SSL blocks it, the code uses a synthetic local dataset so everything still works.
