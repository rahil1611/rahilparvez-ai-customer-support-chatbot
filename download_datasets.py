from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

print("Downloading customer support dataset...")
ds1 = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df1 = pd.DataFrame(ds1["train"])
df1.to_csv("data/customer_support.csv", index=False)
print(f"✅ Customer support dataset: {len(df1)} records saved")

print("Downloading sentiment dataset...")
ds2 = load_dataset("rotten_tomatoes")
df2 = pd.DataFrame(ds2["train"])
df2.to_csv("data/sentiment_data.csv", index=False)
print(f"✅ Sentiment dataset: {len(df2)} records saved")

print("\n✅ All datasets downloaded successfully!")
print("Files saved in the /data folder")