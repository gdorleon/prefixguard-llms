import os, json, argparse, numpy as np, pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoConfig,
    AutoModelForSequenceClassification, LlamaForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    set_seed, EarlyStoppingCallback
)
from peft import get_peft_model, PrefixTuningConfig, TaskType
from .utils import load_csv, label_stats
import torch

# Helper: wrap a dataframe into a HuggingFace dataset and tokenize text
def build_dataset(tokenizer, df: pd.DataFrame, text_col="text", label_col="label", max_length=256):
    ds = Dataset.from_pandas(df[[text_col, label_col]])
    def tok(ex):
        # Tokenize and attach labels
        x = tokenizer(ex[text_col], truncation=True, max_length=max_length)
        x["labels"] = ex[label_col]
        return x
    return ds.map(tok, batched=True, remove_columns=[text_col, label_col])

# Build model and tokenizer, and handle llama quirks (no pad token etc.)
def make_model_and_tokenizer(model_name_or_path: str, num_labels: int):
    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    is_llama = getattr(config, "model_type", "") == "llama"

    if is_llama:
        # LLaMA doesn’t ship with a pad token, so we borrow EOS
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id
        model = LlamaForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    return model, tokenizer, is_llama

# Freeze everything except classifier head
def freeze_backbone_keep_head(model):
    if hasattr(model, "base_model"):
        for p in model.base_model.parameters():
            p.requires_grad = False
    else:
        for name, p in model.named_parameters():
            if "classifier" not in name and "score" not in name:
                p.requires_grad = False

# Build optimizer with separate learning rates for head vs prefix (as in paper)
def build_optimizer(model, lr_head: float, lr_prefix: float, lr_other: float, weight_decay: float):
    head_keys = ("classifier", "score")
    prefix_keys = ("prefix_encoder", "peft")

    head_params, prefix_params, other_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in head_keys):
            head_params.append(p)
        elif any(k in n for k in prefix_keys):
            prefix_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    if prefix_params:
        param_groups.append({"params": prefix_params, "lr": lr_prefix, "weight_decay": weight_decay})
    if other_params:
        param_groups.append({"params": other_params, "lr": lr_other, "weight_decay": weight_decay})

    return torch.optim.AdamW(param_groups)

def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load training and dev CSVs
    train_df = load_csv(args.train_csv)
    dev_df   = load_csv(args.dev_csv)

    # Make sure labels are integers
    if not np.issubdtype(train_df["label"].dtype, np.number):
        mapping = {v: i for i, v in enumerate(sorted(train_df["label"].unique()))}
        train_df["label"] = train_df["label"].map(mapping)
        dev_df["label"] = dev_df["label"].map(mapping)

    n_labels = int(train_df["label"].nunique()) if args.num_labels is None else args.num_labels
    print("Train stats:", label_stats(train_df))
    print("Dev stats:", label_stats(dev_df))

    model, tokenizer, is_llama = make_model_and_tokenizer(args.model_name_or_path, n_labels)

    # Freeze backbone, leave head trainable
    freeze_backbone_keep_head(model)

    # Add prefix tuning adapters
    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=args.prefix_tokens,
        inference_mode=False
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Tokenize data
    train_ds = build_dataset(tokenizer, train_df, max_length=args.max_seq_length)
    dev_ds   = build_dataset(tokenizer, dev_df,   max_length=args.max_seq_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,  # fallback for any group not explicitly set
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed
    )

    # Macro-F1 and Accuracy reporting
    def compute_metrics(p):
        from sklearn.metrics import f1_score, accuracy_score
        preds = np.argmax(p.predictions, axis=1)
        f1 = f1_score(p.label_ids, preds, average="macro")
        acc = accuracy_score(p.label_ids, preds)
        return {"macro_f1": f1, "accuracy": acc}

    optimizer = build_optimizer(
        model,
        lr_head=args.lr_head,
        lr_prefix=args.lr_prefix,
        lr_other=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Trainer handles training loop and eval
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    trainer.train()

    # Save model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save small JSON config so eval knows how to reload
    with open(os.path.join(args.output_dir, "prefixguard_config.json"), "w") as f:
        json.dump({
            "model_name_or_path": args.model_name_or_path,
            "num_labels": n_labels,
            "prefix_tokens": args.prefix_tokens,
            "max_seq_length": args.max_seq_length,
            "is_llama": is_llama
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
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=5e-5)  # fallback
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--lr_prefix", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
