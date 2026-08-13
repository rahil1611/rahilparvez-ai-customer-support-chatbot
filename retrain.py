import pandas as pd
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from database import (get_all_conversations, Session, ModelMetrics)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("   AI CUSTOMER SUPPORT - MODEL RETRAINING")
print("=" * 60)

# ─────────────────────────────────────────
# LOAD ORIGINAL DATA
# ─────────────────────────────────────────
print("\n📦 Loading original datasets...")
sentiment_train = pd.read_csv("data/processed/sentiment_train.csv")
sentiment_test  = pd.read_csv("data/processed/sentiment_test.csv")

sentiment_train.dropna(subset=["clean_text", "sentiment_encoded"], inplace=True)
sentiment_test.dropna(subset=["clean_text", "sentiment_encoded"], inplace=True)

print(f"  Original train: {len(sentiment_train)} | Test: {len(sentiment_test)}")

# ─────────────────────────────────────────
# LOAD CONVERSATIONS FROM DATABASE
# ─────────────────────────────────────────
print("\n📦 Loading conversations from database...")
conversations = get_all_conversations()
conv_df = pd.DataFrame(conversations)

# Filter conversations with valid data
conv_df = conv_df.dropna(subset=["user_message", "sentiment"])
conv_df["clean_text"] = conv_df["user_message"]
conv_df["sentiment_encoded"] = conv_df["sentiment"].map({
    "positive": 2, "neutral": 1, "negative": 0
}).astype(int)

print(f"  New conversations: {len(conv_df)}")

# ─────────────────────────────────────────
# COMBINE DATASETS
# ─────────────────────────────────────────
print("\n🔄 Combining datasets...")
combined_train = pd.concat([sentiment_train, conv_df[["clean_text", "sentiment_encoded"]]], ignore_index=True)
combined_train.drop_duplicates(subset=["clean_text"], inplace=True)

print(f"  Combined train size: {len(combined_train)}")
print(f"  Sentiment distribution:")
print(combined_train["sentiment_encoded"].value_counts().sort_index())

# ─────────────────────────────────────────
# FEATURE ENGINEERING — TF-IDF
# ─────────────────────────────────────────
print("\n⚙️  Re-applying TF-IDF Vectorization...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                        sublinear_tf=True, min_df=2)
X_train = tfidf.fit_transform(combined_train["clean_text"].astype(str))
X_test  = tfidf.transform(sentiment_test["clean_text"].astype(str))
y_train = combined_train["sentiment_encoded"].astype(int)
y_test  = sentiment_test["sentiment_encoded"].astype(int)

print(f"  Feature matrix: {X_train.shape}")
pickle.dump(tfidf, open("models/tfidf_vectorizer.pkl", "wb"))
print("  ✅ Updated TF-IDF vectorizer saved")

# ─────────────────────────────────────────
# LOAD PREVIOUS METRICS
# ─────────────────────────────────────────
session = Session()
try:
    prev_metrics = session.query(ModelMetrics).order_by(ModelMetrics.timestamp.desc()).limit(4).all()
    prev_acc = {}
    for m in prev_metrics:
        prev_acc[m.model_name] = m.accuracy
    print(f"\n📊 Previous accuracies: {prev_acc}")
except:
    prev_acc = {}
finally:
    session.close()

# ─────────────────────────────────────────
# RETRAIN LOGISTIC REGRESSION
# ─────────────────────────────────────────
print("\n🤖 Retraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred, average="weighted")

print(f"  New Accuracy: {lr_acc:.4f} (Previous: {prev_acc.get('Logistic Regression', 'N/A')})")
print(f"  New F1 Score: {lr_f1:.4f}")

pickle.dump(lr_model, open("models/logistic_regression_tuned.pkl", "wb"))
print("  ✅ Logistic Regression retrained and saved")

# Save metrics
session = Session()
try:
    lr_metric = ModelMetrics(
        model_name="Logistic Regression",
        accuracy=lr_acc,
        f1_score=lr_f1,
        precision=precision_score(y_test, lr_pred, average="weighted"),
        recall=recall_score(y_test, lr_pred, average="weighted"),
        timestamp=datetime.now()
    )
    session.add(lr_metric)
    session.commit()
except Exception as e:
    print(f"  Error saving LR metrics: {e}")
finally:
    session.close()

# ─────────────────────────────────────────
# RETRAIN DECISION TREE
# ─────────────────────────────────────────
print("\n🌳 Retraining Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=5,
                                   min_samples_leaf=2, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred, average="weighted")

print(f"  New Accuracy: {dt_acc:.4f} (Previous: {prev_acc.get('Decision Tree', 'N/A')})")
print(f"  New F1 Score: {dt_f1:.4f}")

pickle.dump(dt_model, open("models/decision_tree_tuned.pkl", "wb"))
print("  ✅ Decision Tree retrained and saved")

# Save metrics
session = Session()
try:
    dt_metric = ModelMetrics(
        model_name="Decision Tree",
        accuracy=dt_acc,
        f1_score=dt_f1,
        precision=precision_score(y_test, dt_pred, average="weighted"),
        recall=recall_score(y_test, dt_pred, average="weighted"),
        timestamp=datetime.now()
    )
    session.add(dt_metric)
    session.commit()
except Exception as e:
    print(f"  Error saving DT metrics: {e}")
finally:
    session.close()

# ─────────────────────────────────────────
# FINE-TUNE DISTILBERT
# ─────────────────────────────────────────
print("\n🤖 Fine-tuning DistilBERT with new conversations...")

# Prepare data for DistilBERT
intent_data = conv_df.dropna(subset=["user_message", "intent"])
if len(intent_data) > 0:
    print(f"  Conversations with intent: {len(intent_data)}")

    # Load existing model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("models/finetuned/best_model")
    model = DistilBertForSequenceClassification.from_pretrained("models/finetuned/best_model")
    model.to(device)

    # Load label encoder
    le = pickle.load(open("models/finetuned/label_encoder.pkl", "rb"))
    intent_data["label"] = intent_data["intent"].map({v: k for k, v in enumerate(le.classes_)})
    intent_data = intent_data.dropna(subset=["label"])

    if len(intent_data) > 0:
        # Create dataset
        class RetrainDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len=64):
                self.texts = texts.tolist()
                self.labels = labels.tolist()
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                enc = self.tokenizer(
                    self.texts[idx],
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                return {
                    "input_ids": enc["input_ids"].flatten(),
                    "attention_mask": enc["attention_mask"].flatten(),
                    "labels": torch.tensor(self.labels[idx], dtype=torch.long)
                }

        dataset = RetrainDataset(intent_data["user_message"], intent_data["label"], tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Fine-tune
        optimizer = AdamW(model.parameters(), lr=1e-5)
        model.train()

        print("  Fine-tuning...")
        for epoch in range(1):  # One epoch fine-tuning
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch 1 Loss: {avg_loss:.4f}")

        # Save updated model
        model.save_pretrained("models/finetuned/best_model")
        tokenizer.save_pretrained("models/finetuned/best_model")
        print("  ✅ DistilBERT fine-tuned and saved")

        # Note: For simplicity, not calculating new accuracy here as it would require test data
    else:
        print("  No valid intent data for fine-tuning")
else:
    print("  No conversations with intent data")

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
print("\n" + "="*60)
print("   RETRAINING COMPLETE")
print("="*60)
print(f"  Logistic Regression: {lr_acc:.4f} accuracy")
print(f"  Decision Tree:      {dt_acc:.4f} accuracy")
print(f"  DistilBERT:         Fine-tuned on {len(intent_data) if 'intent_data' in locals() else 0} new samples")
print("\n  Models saved to models/ folder")
print("  Metrics saved to database")
print("="*60)</content>
<parameter name="filePath">c:\Users\DELL\Desktop\ai-chatbot\retrain.py