import os
import time
from datetime import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('grayscale')  # автоматично переводить усі кольори у відтінки сірого

from copy import deepcopy

import warnings, logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_default_handler()

logging.getLogger("transformers").setLevel(logging.ERROR)


import torch
from transformers import (
    AutoTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, DatasetDict

# === CONFIG ===
dataset = "dataset.csv"
dataset_path = "./datasets/" + dataset
output_dir = "./results/ofat_test/"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

local = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(local, use_fast=True)

# === DATA ===
df = pd.read_csv(dataset_path)
df['label'] = df['sent'].apply(lambda x: 1 if x + 1 >= 1 else 0)
train_df, val_df = train_test_split(df[['text','label']], test_size=0.2, random_state=42)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))
raw = DatasetDict({"train": train_ds, "validation": val_ds})

def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

tok_ds = raw.map(tok, batched=True, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

def model_init():
    return DistilBertForSequenceClassification.from_pretrained(
        local, local_files_only=True, num_labels=2
    )

# === BASE HYPERPARAMETERS ===
base_params = dict(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    warmup_ratio=0.1,
    num_train_epochs=3,
    lr_scheduler_type="linear",
    gradient_accumulation_steps=1
)

# === PARAMETERS TO TEST (only one changes each time) ===
param_grid = {
    "learning_rate": [5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4, 5e-4],
    "per_device_train_batch_size": [8, 16, 32],
    "weight_decay": [0.0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15],
    "warmup_ratio": [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
    "num_train_epochs": [2, 3, 4, 5, 6, 8],
    "lr_scheduler_type": ["linear", "cosine", "cosine_with_restarts", "polynomial"],
    "gradient_accumulation_steps": [1, 2, 4, 8],
}

results = []

# === MAIN LOOP ===
for param_name, values in param_grid.items():
    print(f"\n🔧 Testing parameter: {param_name}")

    for val in values:
        # set up current parameter set
        params = deepcopy(base_params)
        params[param_name] = val

        run_name = f"{param_name}_{val}"
        run_output = os.path.join(output_dir, run_name)
        os.makedirs(run_output, exist_ok=True)

        args = TrainingArguments(
            output_dir=run_output,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            weight_decay=params["weight_decay"],
            gradient_accumulation_steps=params["gradient_accumulation_steps"],
            warmup_ratio=params["warmup_ratio"],
            num_train_epochs=params["num_train_epochs"],
            lr_scheduler_type=params["lr_scheduler_type"],
            report_to="none",
            logging_steps=50,
            seed=42,
            disable_tqdm=True,
        )

        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=tok_ds["train"],  # subset for speed
            eval_dataset=tok_ds["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )

        print(f"  ▶ Training with {param_name}={val}")
        start = time.time()
        trainer.train()
        metrics = trainer.evaluate()
        elapsed = time.time() - start

        acc = metrics.get("eval_accuracy")
        f1  = metrics.get("eval_f1")

        print(f"    acc={acc:.4f}, f1={f1:.4f}, time={elapsed/60:.1f}min")

        results.append({
            "param": param_name,
            "value": val,
            "accuracy": acc,
            "f1": f1,
            "time_min": elapsed/60
        })

# === SAVE RESULTS ===
df_results = pd.DataFrame(results)
csv_path = os.path.join(output_dir, "ofat_results.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n✅ Results saved to {csv_path}")

# === PLOTS ===
for param in df_results["param"].unique():
    subset = df_results[df_results["param"] == param]

    plt.figure()
    # Чорно-білий стиль: різні маркери / лінії, без кольорів
    plt.plot(
        subset["value"], subset["accuracy"],
        marker="o", linestyle="-", color="black", label="Accuracy"
    )
    plt.plot(
        subset["value"], subset["f1"],
        marker="x", linestyle="--", color="gray", label="F1"
    )

    plt.xlabel(param)
    plt.ylabel("Score")
    plt.title(f"OFAT: {param}")
    plt.legend(frameon=False)
    plt.grid(True, linestyle=":", linewidth=0.6)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{param}_metrics_bw.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"📈 Saved {out_path}")






