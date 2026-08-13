import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models/finetuned", exist_ok=True)
os.makedirs("results/finetune", exist_ok=True)
os.makedirs("results/finetune/plots", exist_ok=True)

print("=" * 60)
print("   DISTILBERT FINE-TUNING — CUSTOMER SUPPORT")
print("=" * 60)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CONFIG = {
    "model_name"   : "distilbert-base-uncased",
    "max_length"   : 64,
    "batch_size"   : 16,
    "epochs"       : 3,
    "learning_rate": 2e-5,
    "warmup_steps" : 100,
    "seed"         : 42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device : {device}")
print(f"  Config : {CONFIG}")

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
print("\n📦 Loading dataset...")
df = pd.read_csv("data/finetune/finetune_3000.csv")
df.dropna(subset=["instruction", "category"], inplace=True)
df.drop_duplicates(subset=["instruction"], inplace=True)
print(f"  Records : {len(df)}")
print(f"  Intents : {df['category'].nunique()}")
print(f"\n  Intent distribution:")
print(df["category"].value_counts().to_string())

# ─────────────────────────────────────────
# LABEL ENCODING
# ─────────────────────────────────────────
le = LabelEncoder()
df["label"] = le.fit_transform(df["category"])
num_classes = len(le.classes_)
print(f"\n  Classes : {num_classes}")
pickle.dump(le, open("models/finetuned/label_encoder.pkl", "wb"))
print("  ✅ Label encoder saved")

# save intent mapping
intent_map = {i: label for i, label in enumerate(le.classes_)}
pd.DataFrame(list(intent_map.items()),
             columns=["id","intent"]).to_csv(
             "models/finetuned/intent_mapping.csv", index=False)

# ─────────────────────────────────────────
# TRAIN / VAL SPLIT
# ─────────────────────────────────────────
train_df, val_df = train_test_split(df, test_size=0.2,
                                     random_state=CONFIG["seed"],
                                     stratify=df["label"])
print(f"\n  Train : {len(train_df)} | Val : {len(val_df)}")

# ─────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────
class SupportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.tolist()
        self.labels    = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length    = self.max_len,
            padding       = "max_length",
            truncation    = True,
            return_tensors= "pt"
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ─────────────────────────────────────────
# TOKENIZER & MODEL
# ─────────────────────────────────────────
print("\n🤖 Loading DistilBERT tokenizer and model...")
tokenizer = DistilBertTokenizer.from_pretrained(CONFIG["model_name"])
model     = DistilBertForSequenceClassification.from_pretrained(
                CONFIG["model_name"], num_labels=num_classes)
model.to(device)
print(f"  ✅ Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")

# ─────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────
train_dataset = SupportDataset(train_df["instruction"], train_df["label"],
                                tokenizer, CONFIG["max_length"])
val_dataset   = SupportDataset(val_df["instruction"],   val_df["label"],
                                tokenizer, CONFIG["max_length"])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False)
print(f"\n  Train batches : {len(train_loader)}")
print(f"  Val batches   : {len(val_loader)}")

# ─────────────────────────────────────────
# OPTIMIZER & SCHEDULER
# ─────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], eps=1e-8)
total_steps = len(train_loader) * CONFIG["epochs"]
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps  = CONFIG["warmup_steps"],
    num_training_steps= total_steps
)

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("   TRAINING STARTED")
print("=" * 60)

history = {"train_loss":[], "val_loss":[], "val_acc":[], "val_f1":[]}
best_val_f1 = 0

for epoch in range(CONFIG["epochs"]):
    # ── TRAIN ──
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

    # ── VALIDATE ──
    model.eval()
    val_loss  = 0
    all_preds = []
    all_labels= []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            val_loss  += outputs.loss.item()
            preds      = torch.argmax(outputs.logits, dim=1)
            all_preds .extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc  = accuracy_score(all_labels, all_preds)
    val_f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    val_prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    val_rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

    history["train_loss"].append(avg_train_loss)
    history["val_loss"]  .append(avg_val_loss)
    history["val_acc"]   .append(val_acc)
    history["val_f1"]    .append(val_f1)

    print(f"\n{'─'*60}")
    print(f"  Epoch {epoch+1}/{CONFIG['epochs']} COMPLETE")
    print(f"  Train Loss : {avg_train_loss:.4f}")
    print(f"  Val Loss   : {avg_val_loss:.4f}")
    print(f"  Val Acc    : {val_acc:.4f}")
    print(f"  Val F1     : {val_f1:.4f}")
    print(f"  Val Prec   : {val_prec:.4f}")
    print(f"  Val Recall : {val_rec:.4f}")
    print(f"{'─'*60}\n")

    # save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model.save_pretrained("models/finetuned/best_model")
        tokenizer.save_pretrained("models/finetuned/best_model")
        print(f"  ✅ Best model saved (F1={best_val_f1:.4f})")

# ─────────────────────────────────────────
# FINAL EVALUATION
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("   FINAL EVALUATION")
print("=" * 60)
print(classification_report(all_labels, all_preds,
      target_names=le.classes_, zero_division=0))

# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
print("\n📊 Generating plots...")

# Loss curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("DistilBERT Fine-Tuning Results", fontsize=16, fontweight="bold")

axes[0].plot(range(1, CONFIG["epochs"]+1), history["train_loss"],
             "o-", color="#6C63FF", label="Train Loss", linewidth=2)
axes[0].plot(range(1, CONFIG["epochs"]+1), history["val_loss"],
             "s-", color="#FF6584", label="Val Loss",   linewidth=2)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Training vs Validation Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, CONFIG["epochs"]+1), history["val_acc"],
             "o-", color="#43D39E", label="Accuracy", linewidth=2)
axes[1].plot(range(1, CONFIG["epochs"]+1), history["val_f1"],
             "s-", color="#FFBE0B", label="F1 Score",  linewidth=2)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
axes[1].set_title("Validation Accuracy & F1")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig("results/finetune/plots/training_curves.png", dpi=150)
plt.close()
print("  📊 Training curves saved")

# Confusion matrix (top 10 intents only for readability)
top_intents  = df["category"].value_counts().head(10).index.tolist()
top_ids      = [list(le.classes_).index(i) for i in top_intents if i in le.classes_]
mask         = [l in top_ids for l in all_labels]
filtered_true= [all_labels[i] for i in range(len(all_labels)) if mask[i]]
filtered_pred= [all_preds[i]  for i in range(len(all_preds))  if mask[i]]

if filtered_true:
    cm = confusion_matrix(filtered_true, filtered_pred, labels=top_ids)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=top_intents, yticklabels=top_intents)
    plt.title("Confusion Matrix — Top 10 Intents", fontsize=14, fontweight="bold")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results/finetune/plots/confusion_matrix.png", dpi=150)
    plt.close()
    print("  📊 Confusion matrix saved")

# Per-intent F1 bar chart
report = classification_report(all_labels, all_preds,
         target_names=le.classes_, output_dict=True, zero_division=0)
intent_f1s = {k: v["f1-score"] for k, v in report.items()
              if k in le.classes_}
sorted_intents = sorted(intent_f1s.items(), key=lambda x: x[1], reverse=True)
names  = [x[0] for x in sorted_intents]
scores = [x[1] for x in sorted_intents]

plt.figure(figsize=(14, 6))
bars = plt.bar(names, scores,
               color=["#6C63FF" if s >= 0.7 else "#FF6584" for s in scores],
               edgecolor="white")
plt.axhline(y=0.7, color="orange", linestyle="--", alpha=0.7, label="0.7 threshold")
plt.xlabel("Intent"); plt.ylabel("F1 Score")
plt.title("Per-Intent F1 Score", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.legend(); plt.tight_layout()
plt.savefig("results/finetune/plots/per_intent_f1.png", dpi=150)
plt.close()
print("  📊 Per-intent F1 chart saved")

# ─────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────
results = {
    "model"      : CONFIG["model_name"],
    "epochs"     : CONFIG["epochs"],
    "best_val_f1": best_val_f1,
    "final_acc"  : history["val_acc"][-1],
    "final_f1"   : history["val_f1"][-1],
    "num_classes": num_classes,
    "train_size" : len(train_df),
    "val_size"   : len(val_df),
}
pd.DataFrame([results]).to_csv("results/finetune/finetune_results.csv", index=False)

print("\n" + "=" * 60)
print("   FINE-TUNING COMPLETE ✅")
print("=" * 60)
print(f"  Best Val F1  : {best_val_f1:.4f}")
print(f"  Final Acc    : {history['val_acc'][-1]:.4f}")
print(f"  Model saved  : models/finetuned/best_model/")
print(f"  Plots saved  : results/finetune/plots/")
print("=" * 60)