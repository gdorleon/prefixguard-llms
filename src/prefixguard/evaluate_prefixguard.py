import os, json, argparse, numpy as np, pandas as pd, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel
from .utils import (
    load_csv, label_stats, classification_report,
    make_obfuscated_df, per_group_metrics,
    expected_calibration_error, temperature_scale, softmax
)

def build_dataset(tokenizer, df: pd.DataFrame, text_col="text", label_col="label", max_length=256):
    ds = Dataset.from_pandas(df[[text_col, label_col]])
    def tok(ex):
        x = tokenizer(ex[text_col], truncation=True, max_length=max_length)
        x["labels"] = ex[label_col]
        return x
    ds = ds.map(tok, batched=True, remove_columns=[text_col, label_col])
    return ds

@torch.no_grad()
def predict_logits(model, tokenizer, df: pd.DataFrame, max_length: int, batch_size: int = 32):
    model.eval()
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    ds = build_dataset(tokenizer, df, max_length=max_length)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collator)
    all_logits, all_labels = [], []
    for batch in dl:
        labels = batch.pop("labels").numpy()
        if torch.cuda.is_available():
            for k in batch:
                batch[k] = batch[k].cuda()
        outputs = model(**batch)
        logits = outputs.logits.detach().cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels)
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

def main(args):
    # Load base or peft-wrapped model
    cfg_path = os.path.join(args.model_name_or_path, "prefixguard_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        base_model_name = cfg["model_name_or_path"]
        num_labels = cfg["num_labels"]
        max_seq_len = cfg["max_seq_length"]
    else:
        base_model_name = args.model_name_or_path
        num_labels = args.num_labels
        max_seq_len = args.max_seq_length

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path if os.path.isdir(args.model_name_or_path) else base_model_name,
        use_fast=True
    )
    config = AutoConfig.from_pretrained(
        args.model_name_or_path if os.path.isdir(args.model_name_or_path) else base_model_name,
        num_labels=num_labels
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path if os.path.isdir(args.model_name_or_path) else base_model_name,
        config=config
    )

    # Attach PEFT weights if present
    if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")) or \
       os.path.exists(os.path.join(args.model_name_or_path, "adapter_model.bin")):
        model = PeftModel.from_pretrained(base_model, args.model_name_or_path)
    else:
        model = base_model

    if torch.cuda.is_available():
        model.cuda()

    # Test
    test_df = load_csv(args.test_csv)
    print("Test stats:", label_stats(test_df))

    logits, labels = predict_logits(model, tokenizer, test_df, max_length=max_seq_len, batch_size=args.per_device_eval_batch_size)
    probs = softmax(logits)[:, 1] if logits.shape[1] == 2 else np.max(softmax(logits), axis=1)
    preds = np.argmax(logits, axis=1)

    rep = classification_report(labels, preds)
    report = {"test": rep}

    # Fairness
    if "group" in test_df.columns:
        fair = per_group_metrics(labels, preds, groups=test_df["group"].astype(str).tolist())
        report["fairness"] = fair

    # Robustness: obfuscated inputs
    if args.evaluate_obfuscation:
        obf_df = make_obfuscated_df(test_df, mode="mixed")
        logits_obf, labels_obf = predict_logits(model, tokenizer, obf_df, max_length=max_seq_len, batch_size=args.per_device_eval_batch_size)
        preds_obf = np.argmax(logits_obf, axis=1)
        rep_obf = classification_report(labels_obf, preds_obf)
        report["robustness"] = {
            "macro_f1_drop": float(rep["macro_f1"] - rep_obf["macro_f1"]),
            "accuracy_drop": float(rep["accuracy"] - rep_obf["accuracy"]),
            "obfuscated": rep_obf
        }

    # Calibration
    if args.calibrate_on_dev_csv is not None:
        dev_df = load_csv(args.calibrate_on_dev_csv)
        logits_dev, labels_dev = predict_logits(model, tokenizer, dev_df, max_length=max_seq_len, batch_size=args.per_device_eval_batch_size)
        ece_raw = expected_calibration_error(probs, labels, n_bins=10)
        T_star = temperature_scale(logits_dev, labels_dev, init_T=1.0)
        logits_T = logits / T_star
        probs_T = softmax(logits_T)[:, 1] if logits_T.shape[1] == 2 else np.max(softmax(logits_T), axis=1)
        ece_T = expected_calibration_error(probs_T, labels, n_bins=10)
        report["calibration"] = {
            "ece_raw": float(ece_raw),
            "temperature": float(T_star),
            "ece_temp_scaled": float(ece_T)
        }

    # Write report
    if args.report_path is not None:
        os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
        with open(args.report_path, "w") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--num_labels", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--calibrate_on_dev_csv", type=str, default=None)
    p.add_argument("--evaluate_obfuscation", action="store_true")
    p.add_argument("--report_path", type=str, default=None)
    args = p.parse_args()
    main(args)
