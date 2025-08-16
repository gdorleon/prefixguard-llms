import os, json, argparse, numpy as np, pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
)
from peft import get_peft_model, PrefixTuningConfig, TaskType
from .utils import load_csv, label_stats
import torch

def build_dataset(tokenizer, df: pd.DataFrame, text_col="text", label_col="label", max_length=256):
    ds = Dataset.from_pandas(df[[text_col, label_col]])
    def tok(ex):
        x = tokenizer(ex[text_col], truncation=True, max_length=max_length)
        x["labels"] = ex[label_col]
        return x
    ds = ds.map(tok, batched=True, remove_columns=[text_col, label_col])
    return ds

def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = load_csv(args.train_csv)
    dev_df   = load_csv(args.dev_csv)

    n_labels = int(train_df["label"].nunique()) if args.num_labels is None else args.num_labels
    print("Train stats:", label_stats(train_df))
    print("Dev stats:", label_stats(dev_df))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=n_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    # Freeze backbone; leave classifier trainable (will be re-wrapped by PEFT)
    if hasattr(model, "base_model"):
        for p in model.base_model.parameters():
            p.requires_grad = False
    else:
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    # Apply Prefix Tuning (PEFT)
    peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=args.prefix_tokens, inference_mode=False)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Datasets
    train_ds = build_dataset(tokenizer, train_df, max_length=args.max_seq_length)
    dev_ds   = build_dataset(tokenizer, dev_df,   max_length=args.max_seq_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed
    )

    def compute_metrics(p):
        from sklearn.metrics import f1_score, accuracy_score
        preds = np.argmax(p.predictions, axis=1)
        f1 = f1_score(p.label_ids, preds, average="macro")
        acc = accuracy_score(p.label_ids, preds)
        return {"macro_f1": f1, "accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "prefixguard_config.json"), "w") as f:
        json.dump({
            "model_name_or_path": args.model_name_or_path,
            "num_labels": n_labels,
            "prefix_tokens": args.prefix_tokens,
            "max_seq_length": args.max_seq_length
        }, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--dev_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--num_labels", type=int, default=None)
    p.add_argument("--prefix_tokens", type=int, default=10)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
