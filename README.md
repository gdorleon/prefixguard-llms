# PrefixGuard LLMs: Efficient Detection of Hate Speech and Sexism

This repository contains reference code for the paper **PrefixGuard LLMs: Efficient Detection of Hate Speech and Sexism**.  
PrefixGuard uses **prefix tuning** (PEFT) to adapt a **frozen** LLM backbone for abusive language detection with *minimal trainable parameters*.

## The Approach
- **Prefix Tuning (PEFT)** on top of a frozen backbone for binary or multi-class classification.
- **Evaluation**: Accuracy, Macro-F1, confusion matrix.
- **Fairness** (if `group` column exists): per-group F1, worst-group F1, F1-gap, FPR-gap.
- **Robustness**: Obfuscation (homoglyphs/leetspeak) and ΔMacro-F1.
- **Calibration**: ECE and optional temperature scaling fitted on the dev set.
- **Reproducible CLI scripts** for training and evaluation.

## Data Format (CSV)
Required columns:
- `text`: input string
- `label`: integer label (0/1 for binary; 0..K-1 for multi-class)

Optional:
- `group`: subgroup string (e.g., `women`, `immigrants`) for fairness metrics

**Example**
```csv
text,label,group
"She's pretty good for a girl",1,women
"She is a great engineer",0,women

