import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import time
from datetime import datetime
from copy import deepcopy

import torch
from transformers import (
    AutoTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, DatasetDict

# NEW
import optuna
import matplotlib.pyplot as plt

dataset = "dataset.csv"
dataset_path = "./datasets/" + dataset
epochs = 5
runname = f"opt_distil_{dataset}_epochs_{epochs}"
logging_dir=f"./logs/{runname}/"
output_dir=f"./results/{runname}/"
hpo_dir = os.path.join(output_dir, "hpo_artifacts")
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(hpo_dir, exist_ok=True)

print(dataset)

# Device (для інфо)
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ⚡ FAST tokenizer з локального кешу
local = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(local, use_fast=True)

def model_init():
    return DistilBertForSequenceClassification.from_pretrained(
        local, local_files_only=True, num_labels=2
    )

print('start reading')
df = pd.read_csv(dataset_path)
print('end reading')

print('fill label start')
df['label'] = df['sent'].apply(lambda x: 1 if x + 1 >= 1 else 0)
print('end fill label')

print('train_test_split start')
train_df, val_df = train_test_split(df[['text','label']], test_size=0.2, random_state=42)
print('end train_test_split')

# 🔁 HF Datasets
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))
raw = DatasetDict({"train": train_ds, "validation": val_ds})

# 🚀 Токенізація
num_workers = 16
def tok(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
    )

print('start tokenizer (multiprocessing map)')
tok_ds = raw.map(
    tok,
    batched=True,
    batch_size=2048,
    num_proc=num_workers,
    remove_columns=["text"],
)
print('end tokenizer')

# Collator
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8
)

# Метрики (тепер і accuracy, і F1)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# Базові аргументи тренування (деякі значення буде підбирати Optuna)
base_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=epochs,            # може бути перезаписано HPO
    per_device_train_batch_size=16,     # може бути перезаписано HPO
    per_device_eval_batch_size=16,
    logging_dir=logging_dir,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="f1",         # стежимо за F1
    greater_is_better=True,
    seed=42,
    # fp16=True  # якщо CUDA Ampere+; або bf16=True там, де підтримується
)

# --- Підмножина для швидкого HPO (за потреби можна даунсемплити)
train_for_hpo = tok_ds["train"]
eval_for_hpo  = tok_ds["validation"]

# --------- Optuna: простір пошуку ---------
def suggest_params(trial: optuna.Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 6),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.15),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts", "polynomial"]
        ),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4]),
    }

def objective(trial: optuna.Trial):
    # 1) скопіювати базові аргументи і записати підібрані значення
    args = TrainingArguments(**deepcopy(base_args.to_dict()))
    params = suggest_params(trial)
    for k, v in params.items():
        setattr(args, k, v)

    # 2) локальний Trainer
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_for_hpo,
        eval_dataset=eval_for_hpo,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 3) тренування + оцінка
    trainer.train()
    metrics = trainer.evaluate()

    # 4) збережемо обидві метрики в атрибути тріалу (для CSV/графіків)
    # метрики у Transformers зазвичай мають ключі виду "eval_f1" і "eval_accuracy"
    eval_f1 = metrics.get("eval_f1", None)
    eval_acc = metrics.get("eval_accuracy", None)
    trial.set_user_attr("eval_f1", float(eval_f1) if eval_f1 is not None else None)
    trial.set_user_attr("eval_accuracy", float(eval_acc) if eval_acc is not None else None)

    # 5) повертай метрику-ціль (оптимізуємо F1)
    return eval_f1

# Створюємо та запускаємо дослідження
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
)
n_trials = 20  # збільшуй, якщо є ресурс
print('start HPO')
hpo_start = time.time()
print(f"🕓 HPO started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
hpo_end = time.time()
print(f"🏁 HPO finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️ HPO duration: {(hpo_end - hpo_start)/60:.2f} minutes")
print("Best params:", study.best_params)

# --------- CSV з усіма тріалами ---------
rows = []
for t in study.trials:
    row = {**t.params}
    row["trial_number"] = t.number
    row["value"] = t.value  # цільова метрика (F1)
    row["eval_f1"] = t.user_attrs.get("eval_f1")
    row["eval_accuracy"] = t.user_attrs.get("eval_accuracy")
    rows.append(row)

df_trials = pd.DataFrame(rows)

csv_path = os.path.join(hpo_dir, "hpo_trials.csv")
df_trials.to_csv(csv_path, index=False)
print(f"✅ HPO trials saved: {csv_path}")

# --------- Графіки: для кожного гіперпараметра — значення accuracy та F1 ---------
def plot_metrics_vs_param(df, param, out_dir):
    # підготуємо x та y
    x = df[param]
    y_f1 = df["eval_f1"]
    y_acc = df["eval_accuracy"]

    plt.figure()
    # для категоріальних параметрів matplotlib вміє малювати категорії напряму
    plt.scatter(x, y_f1, label="F1", marker="o")
    plt.scatter(x, y_acc, label="Accuracy", marker="x")
    plt.xlabel(param)
    plt.ylabel("Score")
    plt.title(f"Metrics vs {param}")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"metrics_vs_{param}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📈 Saved: {out_path}")

# будуємо список параметрів з колонки DataFrame
param_cols = [c for c in df_trials.columns if c not in {"trial_number", "value", "eval_f1", "eval_accuracy"}]
for p in param_cols:
    plot_metrics_vs_param(df_trials, p, hpo_dir)

# --------- Фінальне тренування з найкращими гіперпараметрами ---------
print('start training with best params')
train_start = time.time()
print(f"🕓 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

best_args = TrainingArguments(**{**base_args.to_dict(), **study.best_params})
final_trainer = Trainer(
    model_init=model_init,
    args=best_args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
final_trainer.train()

print('end training')
train_end = time.time()
elapsed = train_end - train_start
print(f"🏁 Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️ Training duration: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")

final_metrics = final_trainer.evaluate()
print("Final eval:", final_metrics)

save_dir = f"./trained/opt_distilbert_{dataset}/"
final_trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ Saved model & tokenizer to: {save_dir}")
