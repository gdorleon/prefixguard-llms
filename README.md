
## Efficient Detection of Hate Speech and Sexism

This repository contains the reference implementation for the paper on **Efficient Detection of Hate Speech and Sexism**.  

The proposed method uses **prefix tuning** (PEFT) to adapt a **frozen** Large Language Model (LLM) backbone for abusive language detection with *minimal trainable parameters*.  
It supports both encoder-style backbones (BERT, RoBERTa) and decoder-style backbones (LLaMA, Llama 2).

---

## About it
- **Prefix Tuning (PEFT):** lightweight adaptation of frozen LLMs for classification tasks.  
- **Supports multiple backbones:** BERT-base, RoBERTa-large, LLaMA, and Llama 2 with `LlamaForSequenceClassification`.  
- **Binary and multi-class classification:** covers both offensive/hate speech detection and nuanced sexism taxonomy.  
- **Evaluation metrics:** Accuracy, Macro-F1, confusion matrix.  
- **Fairness:** subgroup evaluation (per-group F1, worst-group F1, F1/FPR gaps) when a `group` column is present.  
- **Robustness:** adversarial obfuscation (homoglyphs, leetspeak) and Macro-F1 evaluation.  
- **Calibration:** Expected Calibration Error (ECE) and optional temperature scaling for reliable probabilities.  
- **Optimization:** separate learning rates for prefix parameters and classifier head, early stopping on Macro-F1.  

---

## Installation and Requirements
Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/prefixguard-llms.git
cd prefixguard-llms
python -m venv .venv && source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate                            # Windows
pip install -r requirements.txt
````

Install `requirements.txt`:

```
transformers>=4.41.0
datasets>=2.18.0
peft>=0.10.0
accelerate>=0.30.0
scikit-learn>=1.3.0
scipy>=1.10.0
pandas>=2.0.0
numpy>=1.23.0
torch>=2.1.0
bitsandbytes>=0.43.0
```

---

## Data Format

The code expects datasets in CSV format.

**Required columns:**

* `text`: input sentence/post
* `label`: integer label (0/1 for binary; 0..K-1 for multi-class)

**Optional column:**

* `group`: subgroup identifier (e.g., `women`, `immigrants`) for fairness evaluation

**Example:**

```csv
text,label,group
"She's pretty good for a girl",1,women
"She is a great engineer",0,women
```

---

## Training

### Train with an encoder backbone

```bash
python -m src.prefixguard.train_prefixguard \
  --train_csv data/edos_train.csv \
  --dev_csv data/edos_dev.csv \
  --output_dir outputs/edos_roberta_prefix \
  --model_name_or_path roberta-base \
  --num_labels 2 \
  --prefix_tokens 10 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --lr_head 1e-4 \
  --lr_prefix 5e-5 \
  --max_seq_length 256 \
  --num_train_epochs 3 \
  --early_stopping_patience 2 \
  --seed 42
```

### Train with LLaMA or Llama 2

```bash
python -m src.prefixguard.train_prefixguard \
  --train_csv data/edos_train.csv \
  --dev_csv data/edos_dev.csv \
  --output_dir outputs/edos_llama2_7b_prefix \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --num_labels 2 \
  --prefix_tokens 10 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --learning_rate 5e-5 \
  --lr_head 1e-4 \
  --lr_prefix 5e-5 \
  --max_seq_length 256 \
  --num_train_epochs 3 \
  --early_stopping_patience 2 \
  --seed 42
```

After training, the adapted model is saved in the chosen `output_dir`.

---

## Evaluation

Evaluate on the test set, including fairness, robustness, and calibration:

```bash
python -m src.prefixguard.evaluate_prefixguard \
  --test_csv data/edos_test.csv \
  --model_name_or_path outputs/edos_roberta_prefix \
  --per_device_eval_batch_size 32 \
  --max_seq_length 256 \
  --calibrate_on_dev_csv data/edos_dev.csv \
  --evaluate_obfuscation \
  --report_path outputs/edos_roberta_prefix/report.json
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

* **EDOS (sexism)** – SemEval-2023 Task 10
* **OLID (offense)** – SemEval-2019 Task 6
* **HatEval (hate targeting women or immigrants)** – SemEval-2019 Task 5

Prepare CSVs with the above schema and place them in the `data/` folder.

---

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{..,
  title={Efficient Detection of Hate Speech and Sexism},
  author={...},
  booktitle={...},
  year={2025}
}
```

---

## License

MIT

```
```
