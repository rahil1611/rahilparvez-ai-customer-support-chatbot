import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

print("=" * 60)
print("   AI CUSTOMER SUPPORT - MODEL TRAINING")
print("=" * 60)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
print("\n📦 Loading datasets...")
sentiment_train = pd.read_csv("data/processed/sentiment_train.csv")
sentiment_test  = pd.read_csv("data/processed/sentiment_test.csv")
support_full    = pd.read_csv("data/processed/support_full.csv")

sentiment_train.dropna(subset=["clean_text", "sentiment_encoded"], inplace=True)
sentiment_test.dropna(subset=["clean_text", "sentiment_encoded"], inplace=True)

X_train_raw = sentiment_train["clean_text"].astype(str)
y_train     = sentiment_train["sentiment_encoded"].astype(int)
X_test_raw  = sentiment_test["clean_text"].astype(str)
y_test      = sentiment_test["sentiment_encoded"].astype(int)

print(f"  Train size: {len(X_train_raw)} | Test size: {len(X_test_raw)}")
print(f"  Classes: {sorted(y_train.unique())}")

# ─────────────────────────────────────────
# FEATURE ENGINEERING — TF-IDF
# ─────────────────────────────────────────
print("\n⚙️  Applying TF-IDF Vectorization...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                        sublinear_tf=True, min_df=2)
X_train = tfidf.fit_transform(X_train_raw)
X_test  = tfidf.transform(X_test_raw)
print(f"  Feature matrix: {X_train.shape}")
pickle.dump(tfidf, open("models/tfidf_vectorizer.pkl", "wb"))
print("  ✅ TF-IDF vectorizer saved")

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    cv   = cross_val_score(model, X_tr, y_tr, cv=5, scoring="f1_weighted")
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  CV F1     : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\n{classification_report(y_te, y_pred, zero_division=0)}")
    return model, y_pred, {"name": name, "accuracy": acc,
                           "precision": prec, "recall": rec,
                           "f1": f1, "cv_f1": cv.mean()}

def plot_confusion_matrix(name, y_te, y_pred, labels, filename):
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {name}", fontsize=14, fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"results/plots/{filename}", dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix saved: {filename}")

def plot_roc_curve(name, model, X_te, y_te, filename):
    classes = sorted(np.unique(y_te))
    if len(classes) == 2:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_te)[:, 1]
        else:
            y_score = model.decision_function(X_te)
        fpr, tpr, _ = roc_curve(y_te, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlim([0, 1]); plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {name}", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"results/plots/{filename}", dpi=150)
        plt.close()
        print(f"  📊 ROC curve saved: {filename} | AUC = {roc_auc:.4f}")
        return roc_auc
    else:
        y_bin = label_binarize(y_te, classes=classes)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_te)
        else:
            return None
        plt.figure(figsize=(8, 6))
        aucs = []
        colors = ["darkorange", "blue", "green", "red", "purple"]
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                     label=f"Class {cls} (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Multi-class) — {name}", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"results/plots/{filename}", dpi=150)
        plt.close()
        mean_auc = np.mean(aucs)
        print(f"  📊 ROC curve saved: {filename} | Mean AUC = {mean_auc:.4f}")
        return mean_auc

# ─────────────────────────────────────────
# MODEL 1 — LOGISTIC REGRESSION
# ─────────────────────────────────────────
print("\n🤖 MODEL 1: Logistic Regression")
lr_model = LogisticRegression(max_iter=1000, C=1.0,
                               solver="lbfgs")
lr_model, lr_preds, lr_metrics = evaluate_model(
    "Logistic Regression", lr_model, X_train, y_train, X_test, y_test)

label_names = sorted(sentiment_test["sentiment"].dropna().unique()) \
    if "sentiment" in sentiment_test.columns else [str(c) for c in sorted(y_test.unique())]

plot_confusion_matrix("Logistic Regression", y_test, lr_preds,
                      label_names, "cm_logistic_regression.png")
lr_auc = plot_roc_curve("Logistic Regression", lr_model,
                         X_test, y_test, "roc_logistic_regression.png")
lr_metrics["auc"] = lr_auc

pickle.dump(lr_model, open("models/logistic_regression.pkl", "wb"))
print("  ✅ Logistic Regression model saved")

# ─────────────────────────────────────────
# GRID SEARCH TUNING — LOGISTIC REGRESSION
# ─────────────────────────────────────────
print("\n🔍 Grid Search — Logistic Regression...")
lr_params = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "saga"]}
lr_gs = GridSearchCV(LogisticRegression(max_iter=1000),
                     lr_params, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=1)
lr_gs.fit(X_train, y_train)
print(f"  Best params : {lr_gs.best_params_}")
print(f"  Best CV F1  : {lr_gs.best_score_:.4f}")
best_lr = lr_gs.best_estimator_
pickle.dump(best_lr, open("models/logistic_regression_tuned.pkl", "wb"))

# ─────────────────────────────────────────
# MODEL 2 — DECISION TREE
# ─────────────────────────────────────────
print("\n🌳 MODEL 2: Decision Tree")
dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=5,
                                   min_samples_leaf=2, random_state=42)
dt_model, dt_preds, dt_metrics = evaluate_model(
    "Decision Tree", dt_model, X_train, y_train, X_test, y_test)

plot_confusion_matrix("Decision Tree", y_test, dt_preds,
                      label_names, "cm_decision_tree.png")
dt_auc = plot_roc_curve("Decision Tree", dt_model,
                         X_test, y_test, "roc_decision_tree.png")
dt_metrics["auc"] = dt_auc

pickle.dump(dt_model, open("models/decision_tree.pkl", "wb"))
print("  ✅ Decision Tree model saved")

# ─────────────────────────────────────────
# GRID SEARCH TUNING — DECISION TREE
# ─────────────────────────────────────────
print("\n🔍 Grid Search — Decision Tree...")
dt_params = {"max_depth": [10, 20, 30, None],
             "min_samples_split": [2, 5, 10],
             "min_samples_leaf": [1, 2, 4]}
dt_gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                     dt_params, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=1)
dt_gs.fit(X_train, y_train)
print(f"  Best params : {dt_gs.best_params_}")
print(f"  Best CV F1  : {dt_gs.best_score_:.4f}")
best_dt = dt_gs.best_estimator_
pickle.dump(best_dt, open("models/decision_tree_tuned.pkl", "wb"))

# ─────────────────────────────────────────
# COMPARISON VISUALIZATION
# ─────────────────────────────────────────
print("\n📊 Generating comparison charts...")
metrics_df = pd.DataFrame([lr_metrics, dt_metrics])
metrics_df.to_csv("results/model_comparison.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

# Bar chart
metric_cols = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metric_cols))
width = 0.35
bars1 = axes[0].bar(x - width/2,
                    [lr_metrics[m] for m in metric_cols], width,
                    label="Logistic Regression", color="#4C72B0")
bars2 = axes[0].bar(x + width/2,
                    [dt_metrics[m] for m in metric_cols], width,
                    label="Decision Tree", color="#DD8452")
axes[0].set_xticks(x)
axes[0].set_xticklabels(["Accuracy", "Precision", "Recall", "F1"], fontsize=11)
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel("Score")
axes[0].set_title("Metrics Comparison")
axes[0].legend()
for bar in bars1: axes[0].text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + 0.01,
                                f"{bar.get_height():.2f}", ha="center", fontsize=8)
for bar in bars2: axes[0].text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + 0.01,
                                f"{bar.get_height():.2f}", ha="center", fontsize=8)

# Radar-style summary
summary_metrics = ["accuracy", "precision", "recall", "f1", "cv_f1"]
lr_vals = [lr_metrics[m] for m in summary_metrics]
dt_vals = [dt_metrics[m] for m in summary_metrics]
axes[1].plot(summary_metrics, lr_vals, "o-", color="#4C72B0",
             label="Logistic Regression", linewidth=2)
axes[1].plot(summary_metrics, dt_vals, "s-", color="#DD8452",
             label="Decision Tree", linewidth=2)
axes[1].set_ylim(0, 1.1)
axes[1].set_title("Performance Profile")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticklabels(summary_metrics, rotation=15, fontsize=9)

plt.tight_layout()
plt.savefig("results/plots/model_comparison.png", dpi=150)
plt.close()
print("  📊 Comparison chart saved")

# Sentiment distribution
if "sentiment" in sentiment_train.columns:
    plt.figure(figsize=(8, 5))
    sentiment_train["sentiment"].value_counts().plot(
        kind="bar", color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="black")
    plt.title("Sentiment Distribution (Training Data)", fontsize=14, fontweight="bold")
    plt.xlabel("Sentiment"); plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results/plots/sentiment_distribution.png", dpi=150)
    plt.close()
    print("  📊 Sentiment distribution chart saved")

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("   TRAINING COMPLETE — FINAL RESULTS")
print("=" * 60)
for m in [lr_metrics, dt_metrics]:
    print(f"\n  {m['name']}")
    print(f"    Accuracy  : {m['accuracy']:.4f}")
    print(f"    Precision : {m['precision']:.4f}")
    print(f"    Recall    : {m['recall']:.4f}")
    print(f"    F1 Score  : {m['f1']:.4f}")
    print(f"    CV F1     : {m['cv_f1']:.4f}")
    if m.get('auc'):
        print(f"    AUC       : {m['auc']:.4f}")

winner = max([lr_metrics, dt_metrics], key=lambda x: x["f1"])
print(f"\n🏆 Best Model: {winner['name']} (F1 = {winner['f1']:.4f})")
print("\n✅ All models and results saved!")