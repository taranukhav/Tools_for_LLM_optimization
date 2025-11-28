
# DistilBERT Hyperparameter Exploration
OFAT Grid Search (`ofat_distilbert.py`) + Optuna HPO (`opt_distilbert.py`)

This project contains two complementary scripts for tuning a **DistilBERT** text classification model:

1. **OFAT (One-Factor-At-a-Time) sweep** – systematic grid search over one hyperparameter at a time.
2. **Optuna-powered Hyperparameter Optimization (HPO)** – Bayesian-style search using Optuna with pruning and metric tracking.

Both scripts work on the same dataset and model family and share a lot of common setup.

---

## 1. Repository Structure

A typical layout looks like this:

```text
.
├── datasets/
│   └── dataset.csv        # put your dataset file
├── logs/
│   └── opt_distil_dataset.csv_epochs_5/        # created by opt_distilbert.py
├── results/
│   ├── ofat_test/                              # created by ofat_distilbert.py
│   └── opt_distil_dataset.csv_epochs_5/        # created by opt_distilbert.py
│       └── hpo_artifacts/
├── trained/
│   └── opt_distilbert_dataset.csv/             # final best model from opt_distilbert.py
├── ofat_distilbert.py                          # OFAT grid-search script (name is up to you)
├── opt_distilbert.py                           # Optuna HPO script
└── README.md
```

> Adjust file names to your actual script names if they differ.

---

## 2. Dataset Format (Shared by Both Scripts)

Both scripts expect a CSV file:

- Location: `./datasets/dataset.csv`  
- At minimum, the following columns:
  - `text` – the input text for classification.
  - `sent` – a numeric value used to derive the binary label.

Label creation in both scripts:

```python
df['label'] = df['sent'].apply(lambda x: 1 if x + 1 >= 1 else 0)
```

This effectively assigns label `1` for most values of `sent`.  
You will likely want to adapt this logic to your use case, for example:

```python
# Example: positive vs non-positive sentiment
df['label'] = df['sent'].apply(lambda x: 1 if x > 0 else 0)
```

The scripts then use only `["text", "label"]` for training and validation.

---

## 3. Common Dependencies

Both scripts rely on the same ecosystem:

```bash
pip install   torch   transformers   datasets   scikit-learn   pandas   numpy   matplotlib   optuna
```

> Make sure PyTorch is installed with the correct backend (CPU / CUDA / MPS).  
> You may also want `tensorboard` if you plan to inspect logs from `opt_distilbert.py`.

---

## 4. Model and Tokenizer (Shared)

Both scripts use the multilingual DistilBERT checkpoint:

- **Model/Tokenizer name**: `distilbert-base-multilingual-cased`

Tokenizer:

```python
local = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(local, use_fast=True)
```

Model initialization (fresh per run):

```python
def model_init():
    return DistilBertForSequenceClassification.from_pretrained(
        local,
        local_files_only=True,
        num_labels=2
    )
```

> `local_files_only=True` means the checkpoint **must already be in your local Hugging Face cache**.  
> If it is not, either:
> - run once without `local_files_only=True` (with internet), or  
> - manually download and point `local` to a local directory.

### Device Selection

Both scripts auto-detect the best available device:

```python
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Device: {device}")
```

Priority: **MPS** (Apple Silicon) → **CUDA** → **CPU**.

---

## 5. Script 1: OFAT Grid Search – `ofat_distilbert.py`

This script performs a **one-factor-at-a-time** hyperparameter exploration using Hugging Face’s `Trainer`.  
For each hyperparameter in a predefined grid:

1. Start from base hyperparameters.
2. Override exactly one parameter with a value from the grid.
3. Train and evaluate a model (with early stopping).
4. Log accuracy, F1, and runtime.
5. Append results to a global table.
6. Create grayscale plots.

### 5.1 Data Preparation

- Train/validation split (80/20):

```python
train_df, val_df = train_test_split(
    df[["text", "label"]],
    test_size=0.2,
    random_state=42
)
```

- Hugging Face Datasets:

```python
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))
raw = DatasetDict({"train": train_ds, "validation": val_ds})
```

- Tokenization:

```python
def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

tok_ds = raw.map(tok, batched=True, remove_columns=["text"])
```

- Padding collator:

```python
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8
)
```

### 5.2 Metrics

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }
```

### 5.3 Base Hyperparameters

```python
base_params = dict(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    warmup_ratio=0.1,
    num_train_epochs=3,
    lr_scheduler_type="linear",
    gradient_accumulation_steps=1,
)
```

### 5.4 OFAT Parameter Grid

Each parameter is varied separately:

```python
param_grid = {
    "learning_rate": [5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4, 5e-4],
    "per_device_train_batch_size": [8, 16, 32],
    "weight_decay": [0.0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15],
    "warmup_ratio": [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
    "num_train_epochs": [2, 3, 4, 5, 6, 8],
    "lr_scheduler_type": ["linear", "cosine", "cosine_with_restarts", "polynomial"],
    "gradient_accumulation_steps": [1, 2, 4, 8],
}
```

For each `(param_name, value)` pair:

- Copy `base_params`.
- Override `param_name` with `value`.
- Create `TrainingArguments` with early stopping (patience=1).
- Train and evaluate with `Trainer`.

### 5.5 Outputs (OFAT)

Output directory:

```python
output_dir = "./results/ofat_test/"
```

Per-run subfolders, e.g.:

```text
results/ofat_test/
├── learning_rate_2e-05/
├── learning_rate_3e-05/
├── weight_decay_0.01/
└── ...
```

#### Aggregated CSV

```python
df_results = pd.DataFrame(results)
csv_path = os.path.join(output_dir, "ofat_results.csv")
df_results.to_csv(csv_path, index=False)
```

Columns:

- `param`
- `value`
- `accuracy`
- `f1`
- `time_min`

#### Grayscale Metric Plots

For each hyperparameter `param`:

- Plot Accuracy and F1 vs `value` in grayscale.
- Saved as: `results/ofat_test/{param}_metrics_bw.png`

Example:

- `learning_rate_metrics_bw.png`
- `warmup_ratio_metrics_bw.png`

---

## 6. Script 2: Optuna HPO – `opt_distilbert.py`

This script performs **hyperparameter optimization** using [Optuna](https://optuna.org/).  
It uses Hugging Face’s `Trainer` as the inner training loop and optimizes **F1-score** on the validation set.

Key features:

- Optuna TPE sampler and median pruning.
- Parallel tokenization with `num_proc`.
- Logging to TensorBoard.
- Saving all trial results and plots under `hpo_artifacts/`.
- Final training with the best-found hyperparameters on the full train set.
- Saving the final model and tokenizer.

### 6.1 Setup and Paths

Configuration in the script:

```python
dataset = "dataset.csv"
dataset_path = "./datasets/" + dataset
epochs = 5
runname = f"opt_distil_{dataset}_epochs_{epochs}"

logging_dir = f"./logs/{runname}/"
output_dir  = f"./results/{runname}/"
hpo_dir     = os.path.join(output_dir, "hpo_artifacts")

os.makedirs(logging_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(hpo_dir, exist_ok=True)
```

Environment variable to allow tokenizer parallelism:

```python
os.environ["TOKENIZERS_PARALLELISM"] = "true"
```

### 6.2 Data Preparation and Tokenization

Same split logic as OFAT:

```python
train_df, val_df = train_test_split(
    df[["text","label"]],
    test_size=0.2,
    random_state=42
)
```

Converted to Hugging Face Datasets:

```python
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))
raw = DatasetDict({"train": train_ds, "validation": val_ds})
```

Tokenization with multiprocessing:

```python
num_workers = 16

def tok(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
    )

tok_ds = raw.map(
    tok,
    batched=True,
    batch_size=2048,
    num_proc=num_workers,
    remove_columns=["text"],
)
```

> You may need to reduce `num_workers` and/or `batch_size` depending on your CPU and RAM.

### 6.3 Base TrainingArguments

Some arguments are fixed, others are overwritten by Optuna:

```python
base_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=epochs,
    per_device_train_batch_size=16,  # Overridden by HPO
    per_device_eval_batch_size=16,
    logging_dir=logging_dir,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=42,
)
```

### 6.4 Hyperparameter Search Space

Optuna suggestion function:

```python
def suggest_params(trial: optuna.Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 6),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.15),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type",
            ["linear", "cosine", "cosine_with_restarts", "polynomial"],
        ),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [1, 2, 4]
        ),
    }
```

### 6.5 Objective Function

For each trial:

1. Copy base `TrainingArguments`.
2. Overwrite chosen hyperparameters.
3. Create a local `Trainer`.
4. Train with early stopping (patience=2).
5. Evaluate on validation set.
6. Store F1 and accuracy in trial attributes.
7. Return F1 as the objective value.

Optuna study:

```python
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
)

n_trials = 20
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
print("Best params:", study.best_params)
```

### 6.6 Trial CSV and Plots

After optimization:

```python
rows = []
for t in study.trials:
    row = {**t.params}
    row["trial_number"] = t.number
    row["value"] = t.value            # objective (F1)
    row["eval_f1"] = t.user_attrs.get("eval_f1")
    row["eval_accuracy"] = t.user_attrs.get("eval_accuracy")
    rows.append(row)

df_trials = pd.DataFrame(rows)
csv_path = os.path.join(hpo_dir, "hpo_trials.csv")
df_trials.to_csv(csv_path, index=False)
```

Per-parameter plots (F1 and accuracy vs hyperparameter value):

```python
def plot_metrics_vs_param(df, param, out_dir):
    x = df[param]
    y_f1 = df["eval_f1"]
    y_acc = df["eval_accuracy"]

    plt.figure()
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
```

All plots are saved under:

```text
./results/opt_distil_.../hpo_artifacts/
```

### 6.7 Final Training with Best Hyperparameters

After HPO finishes, the script trains a final model using `study.best_params` on the full tokenized train set:

```python
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
final_metrics = final_trainer.evaluate()
```

Model and tokenizer are saved to:

```python
save_dir = f"./trained/opt_distilbert_{dataset}/"
final_trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
```

Resulting directory example:

```text
trained/
└── opt_distilbert_dataset.csv/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```

---

## 7. How to Run

From the project root:

### 7.1 Run OFAT Grid Search

```bash
python ofat_distilbert.py
```

Artifacts will appear in:

- `./results/ofat_test/`
- `./results/ofat_test/ofat_results.csv`
- `./results/ofat_test/*_metrics_bw.png`

### 7.2 Run Optuna HPO

```bash
python opt_distilbert.py
```

Artifacts will appear in:

- `./logs/opt_distil_.../` (TensorBoard logs)
- `./results/opt_distil_.../hpo_artifacts/` (CSV + plots)
- `./trained/opt_distilbert_dataset.csv/` (final model)

---

## 8. Customization Tips

- **Dataset / path**: change `dataset` and `dataset_path` in both scripts.
- **Label logic**: adapt `df["label"] = ...` to your task.
- **Search space**: adjust `param_grid` (OFAT) and `suggest_params` (Optuna).
- **Number of trials**: increase `n_trials` in Optuna for more exhaustive search.
- **Parallelism**: tune `num_workers` and `batch_size` in tokenization to fit your hardware.
- **Early stopping**: modify patience in `EarlyStoppingCallback`.

---

## 9. Troubleshooting

- **Model not found with `local_files_only=True`**  
  Make sure `distilbert-base-multilingual-cased` is downloaded to your Hugging Face cache, or temporarily set `local_files_only=False`.

- **Out-of-memory (GPU/CPU)**  
  - Reduce `per_device_train_batch_size`.  
  - Increase `gradient_accumulation_steps`.  
  - Shorten `max_length` or use a smaller subset of the data.

- **Slow training**  
  - Shrink the OFAT grid / Optuna search space.  
  - Reduce `num_train_epochs`.  
  - Use a smaller dataset sample for quick debugging.

---

