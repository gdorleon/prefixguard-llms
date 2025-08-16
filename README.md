# PrefixGuard LLMs: Efficient Detection of Hate Speech and Sexism

This repository contains the reference implementation for the paper **PrefixGuard LLMs: Efficient Detection of Hate Speech and Sexism**.  

PrefixGuard uses **prefix tuning** (PEFT) to adapt a **frozen** Large Language Model (LLM) backbone for abusive language detection with *minimal trainable parameters*.

---

## About it
- **Prefix Tuning (PEFT):** lightweight adaptation of frozen LLMs for classification tasks.  
- **Binary and multi-class classification:** supports both hate speech and nuanced sexism detection.  
- **Evaluation metrics:** Accuracy, Macro-F1, confusion matrix.  
- **Fairness:** subgroup evaluation (per-group F1, worst-group F1, F1/FPR gaps) when `group` column is present.  
- **Robustness:** adversarial obfuscation (homoglyphs, leetspeak) and Macro-F1 evaluation.  
- **Calibration:** Expected Calibration Error (ECE) and optional temperature scaling for reliable probabilities. 

---

## Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/prefixguard-llms.git
cd prefixguard-llms
python -m venv .venv && source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate                            # Windows
pip install -r requirements.txt
```

---

## Data Format
The code expects datasets in CSV format.  

**Required columns:**
- `text`: input sentence/post
- `label`: integer label (0/1 for binary; 0..K-1 for multi-class)

**Optional column:**
- `group`: subgroup identifier (e.g., `women`, `immigrants`) for fairness evaluation

**Example (`data/sample_train.csv`):**
```csv
text,label,group
"She's pretty good for a girl",1,women
"She is a great engineer",0,women
```

---

## Training
Train a PrefixGuard model with prefix tuning on a frozen backbone:

```bash
python -m src.prefixguard.train_prefixguard   --train_csv data/edos_train.csv   --dev_csv data/edos_dev.csv   --output_dir outputs/edos_prefixguard   --model_name_or_path roberta-base   --num_labels 2   --prefix_tokens 10   --per_device_train_batch_size 8   --per_device_eval_batch_size 16   --learning_rate 5e-5   --max_seq_length 256   --num_train_epochs 3   --seed 42
```

After training, the adapted model is saved in `outputs/edos_prefixguard/`.

---

## Evaluation
Evaluate on the test set, including fairness, robustness, and calibration:

```bash
python -m src.prefixguard.evaluate_prefixguard   --test_csv data/edos_test.csv   --model_name_or_path outputs/edos_prefixguard   --per_device_eval_batch_size 32   --max_seq_length 256   --calibrate_on_dev_csv data/edos_dev.csv   --evaluate_obfuscation   --report_path outputs/edos_prefixguard/report.json
```

Metrics (Accuracy, Macro-F1, subgroup fairness gaps, robustness, calibration) are printed and stored in `report.json`.

---

## Repository Structure
```
prefixguard-llms/
├── README.md
├── requirements.txt
├── data/                 # place datasets here
│   ├── edos_train.csv
│   ├── edos_dev.csv
│   └── edos_test.csv
└── src/
    └── prefixguard/
        ├── __init__.py
        ├── utils.py
        ├── train_prefixguard.py
        └── evaluate_prefixguard.py
```

---

## Datasets
The paper experiments use:
- **EDOS (sexism)** – SemEval-2023 Task 10  
- **OLID (offense)** – SemEval-2019 Task 6  
- **HatEval (hate targeting women or immigrants)** – SemEval-2019 Task 5  

Prepare CSVs with the above schema and place them in the `data/` folder.

---

## Citation
If you use this code, please cite our paper:

```bibtex
@inproceedings{..,
  title={PrefixGuard LLMs: Efficient Detection of Hate Speech and Sexism},
  author={...},
  booktitle={...},
  year={2025}
}
```

---

## License
MIT
