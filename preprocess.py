import pandas as pd
import numpy as np
import os
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords

os.makedirs("data/processed", exist_ok=True)

# ─────────────────────────────────────────
# 1. CUSTOMER SUPPORT DATASET
# ─────────────────────────────────────────
print("📦 Processing customer support dataset...")
df = pd.read_csv("data/customer_support.csv")

print(f"  Original shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# Drop duplicates and nulls
df.drop_duplicates(inplace=True)
df.dropna(subset=["instruction", "response"], inplace=True)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s?.!,']", "", text)   # remove special chars
    text = re.sub(r"\s+", " ", text).strip()           # remove extra spaces
    return text

df["clean_instruction"] = df["instruction"].apply(clean_text)
df["clean_response"]    = df["response"].apply(clean_text)

# Encode intent/category labels if they exist
if "intent" in df.columns:
    le = LabelEncoder()
    df["intent_encoded"] = le.fit_transform(df["intent"].astype(str))
    print(f"  Unique intents: {df['intent'].nunique()}")

if "category" in df.columns:
    le2 = LabelEncoder()
    df["category_encoded"] = le2.fit_transform(df["category"].astype(str))
    print(f"  Unique categories: {df['category'].nunique()}")

# Feature: text length
df["instruction_len"] = df["clean_instruction"].apply(len)
df["response_len"]    = df["clean_response"].apply(len)

# Feature: word count
df["instruction_words"] = df["clean_instruction"].apply(lambda x: len(x.split()))
df["response_words"]    = df["clean_response"].apply(lambda x: len(x.split()))

# Feature: has question mark
df["has_question"] = df["clean_instruction"].apply(lambda x: int("?" in x))

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("data/processed/support_train.csv", index=False)
test_df.to_csv("data/processed/support_test.csv",  index=False)
df.to_csv("data/processed/support_full.csv", index=False)

print(f"  ✅ Saved: {len(train_df)} train | {len(test_df)} test records")

# ─────────────────────────────────────────
# 2. SENTIMENT DATASET
# ─────────────────────────────────────────
print("\n📦 Processing sentiment dataset...")
df2 = pd.read_csv("data/sentiment_data.csv")

print(f"  Original shape: {df2.shape}")
print(f"  Columns: {df2.columns.tolist()}")

df2.drop_duplicates(inplace=True)
df2.dropna(inplace=True)

# Identify text column
text_col = None
for col in ["text", "review_body", "sentence", "review"]:
    if col in df2.columns:
        text_col = col
        break

if text_col is None:
    text_col = df2.select_dtypes(include="object").columns[0]

print(f"  Using text column: '{text_col}'")
df2["clean_text"] = df2[text_col].apply(clean_text)

# Identify label column
label_col = None
for col in ["label", "stars", "sentiment", "rating"]:
    if col in df2.columns:
        label_col = col
        break

if label_col:
    print(f"  Using label column: '{label_col}'")
    # Map to positive / negative / neutral
    unique_vals = sorted(df2[label_col].unique())
    print(f"  Unique labels: {unique_vals}")

    if df2[label_col].dtype in [float, int] or str(df2[label_col].dtype).startswith("int"):
        max_val = df2[label_col].max()
        if max_val >= 4:
            # 1-5 star rating
            df2["sentiment"] = df2[label_col].apply(
                lambda x: "positive" if x >= 4 else ("negative" if x <= 2 else "neutral")
            )
        else:
            # binary 0/1
            df2["sentiment"] = df2[label_col].apply(lambda x: "positive" if x == 1 else "negative")
    else:
        df2["sentiment"] = df2[label_col].astype(str).str.lower()

    le3 = LabelEncoder()
    df2["sentiment_encoded"] = le3.fit_transform(df2["sentiment"])
    print(f"  Sentiment distribution:\n{df2['sentiment'].value_counts()}")

# Feature engineering
df2["text_len"]   = df2["clean_text"].apply(len)
df2["word_count"] = df2["clean_text"].apply(lambda x: len(x.split()))
df2["has_exclaim"]= df2["clean_text"].apply(lambda x: int("!" in x))
df2["has_question"]= df2["clean_text"].apply(lambda x: int("?" in x))

stop_words = set(stopwords.words("english"))
df2["stopword_ratio"] = df2["clean_text"].apply(
    lambda x: len([w for w in x.split() if w in stop_words]) / (len(x.split()) + 1)
)

train2, test2 = train_test_split(df2, test_size=0.2, random_state=42)

train2.to_csv("data/processed/sentiment_train.csv", index=False)
test2.to_csv("data/processed/sentiment_test.csv",  index=False)
df2.to_csv("data/processed/sentiment_full.csv", index=False)

print(f"  ✅ Saved: {len(train2)} train | {len(test2)} test records")

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
print("\n" + "="*50)
print("✅ PREPROCESSING COMPLETE")
print("="*50)
print(f"  Support dataset  → {len(df)} records")
print(f"  Sentiment dataset → {len(df2)} records")
print("  All files saved in data/processed/")