# AI Customer Support Chatbot

An AI-powered customer support chatbot that combines intent classification, sentiment analysis, and generative response modeling behind a FastAPI backend, with a self-contained web chat UI and an admin dashboard.

## Overview

The system takes an incoming customer message and runs it through three models before responding:

1. A fine-tuned **DistilBERT** classifier detects the customer's **intent** (e.g. order, refund, shipping, account).
2. A **TF-IDF + Logistic Regression** pipeline detects the message's **sentiment** (positive / neutral / negative).
3. A fine-tuned **GPT-2** language model generates the actual reply, optionally grounded with the closest matching example from a customer-support dataset (via TF-IDF cosine similarity).

If a message is flagged as highly negative or contains anger/complaint keywords, the bot skips generation and returns a scripted escalation message instead of a model-generated one. Every turn is logged to a local database, which also doubles as a growing pool of training data for retraining the models later.

## Features

- REST chat endpoint (`POST /chat`) returning the reply plus sentiment, intent, and confidence scores for each
- Keyword- and confidence-based **escalation detection**, with an escalation reference ID
- Per-session conversation history (`GET /history/{session_id}`)
- Feedback submission endpoint (`POST /feedback`)
- Live model/accuracy stats endpoint (`GET /stats`)
- Model comparison results endpoint (`GET /results`)
- Database inspection endpoints for stored conversations and accumulated training data (`GET /database/*`)
- On-demand retraining trigger (`POST /retrain`), which re-runs `retrain.py` as a subprocess
- Static single-page chat UI (`static/index.html`) and a separate admin view (`static/admin.html`)

## Architecture

```
Browser (static/index.html, admin.html)
        │  HTTP (fetch)
        ▼
FastAPI app (app.py)
        │
        ├── Sentiment: TF-IDF vectorizer + tuned Logistic Regression (models/*.pkl)
        ├── Intent:    Fine-tuned DistilBERT classifier (models/finetuned/best_model)
        ├── Response:  Fine-tuned GPT-2, optionally grounded with a TF-IDF nearest-neighbor
        │              lookup against the cleaned support dataset
        └── database.py → SQLAlchemy ORM → SQLite (chatbot.db)
                 ├── conversations   (every user/bot turn, with scores)
                 ├── training_data   (instruction/response pairs, incl. from live chats)
                 └── model_metrics   (accuracy/F1 history across retrains)
```

`retrain.py` reloads the original datasets, combines them with new conversations pulled from `conversations`, and re-fits the sentiment models and fine-tunes DistilBERT on the combined data — this is what `POST /retrain` invokes.

## Technology Stack

- **Backend:** FastAPI, Pydantic, SQLAlchemy (SQLite)
- **ML / NLP:** PyTorch, Hugging Face Transformers (DistilBERT, GPT-2), scikit-learn (Logistic Regression, Decision Tree, TF-IDF)
- **Data processing:** pandas, numpy, NLTK (stopword removal)
- **Visualization (offline, for training reports):** matplotlib, seaborn
- **Frontend:** static HTML/CSS/JS (no framework), served directly by FastAPI's `StaticFiles`
- **Experimentation/training notebook:** Jupyter (`ai-chatbot.ipynb`) — originally run on a CUDA GPU workspace

## Project Structure

```
ai-chatbot/
├── app.py                     # FastAPI app: loads all models, defines all routes
├── database.py                 # SQLAlchemy models + DB helper functions (SQLite)
├── preprocess.py                # Cleans and splits the raw datasets into train/test CSVs
├── download_datasets.py         # Pulls the public Bitext support & Rotten Tomatoes datasets
├── train_models.py              # Trains + tunes the sentiment models (LogReg, Decision Tree)
├── retrain.py                   # Retrains sentiment models + fine-tunes DistilBERT on new data
├── generate_all_plots.py        # Regenerates evaluation plots from saved models
├── ai-chatbot.ipynb              # Original model training/fine-tuning notebook (TinyLlama → GPT-2, DistilBERT)
├── static/
│   ├── index.html               # Customer-facing chat UI
│   └── admin.html                # Admin dashboard
├── data/
│   ├── customer_support.csv      # Raw Bitext dataset (gitignored, regenerate via download_datasets.py)
│   ├── sentiment_data.csv        # Raw Rotten Tomatoes dataset (gitignored)
│   ├── processed/                # Cleaned train/test splits produced by preprocess.py (gitignored)
│   └── finetune/                 # DistilBERT fine-tuning data + script
├── models/
│   ├── tfidf_vectorizer.pkl, logistic_regression_tuned.pkl, decision_tree_tuned.pkl  # tracked (small)
│   ├── finetuned/                # Fine-tuned DistilBERT weights (gitignored — see Setup)
│   └── gpt2-finetuned/           # Fine-tuned GPT-2 weights (gitignored — see Setup)
├── results/                    # Evaluation plots and metrics CSVs (tracked)
└── chatbot.db                   # Runtime SQLite database (gitignored)
```

> No `requirements.txt` exists in the project yet — see Setup below for the packages actually imported by the code.

## How It Works

1. `download_datasets.py` fetches the raw datasets, `preprocess.py` cleans and splits them.
2. `train_models.py` trains and grid-search-tunes the sentiment models; DistilBERT intent classification and GPT-2 response generation are fine-tuned separately (see `ai-chatbot.ipynb` and `data/finetune/finetune.py`).
3. `app.py` loads all trained artifacts at startup and exposes them through FastAPI routes.
4. Each incoming `/chat` message is scored for sentiment and intent, checked against escalation rules, and answered either by GPT-2 (optionally grounded in a similar historical Q&A pair) or by the escalation script.
5. Every turn is persisted to SQLite via `database.py`, building up a dataset that `retrain.py` can later fold back into the models.

## Setup

Requires Python 3.11 and (for practical training/inference speed) a CUDA-capable GPU — the code checks `torch.cuda.is_available()` and falls back to CPU automatically.

```bash
pip install fastapi uvicorn pydantic sqlalchemy pandas numpy torch transformers scikit-learn matplotlib seaborn nltk datasets huggingface_hub
```

The trained deep-learning models (`models/finetuned/`, `models/gpt2-finetuned/`) are excluded from this repository by `.gitignore` due to size (hundreds of MB to ~1.4GB). To obtain them, either:

- Run the training pipeline yourself: `download_datasets.py` → `preprocess.py` → `train_models.py`, then run the DistilBERT/GPT-2 fine-tuning steps in `ai-chatbot.ipynb` (or `data/finetune/finetune.py`), or
- Copy an existing trained `models/finetuned/` and `models/gpt2-finetuned/` folder from wherever they were originally trained.

If you use the notebook, set your Hugging Face token as an environment variable rather than hardcoding it:

```bash
export HF_TOKEN=your_token_here   # do not commit this value
```

## Running the Project

```bash
uvicorn app:app --reload
```

Then open `http://localhost:8000/` for the chat UI, or `http://localhost:8000/static/admin.html` for the admin dashboard. Interactive API docs are available at `http://localhost:8000/docs` (FastAPI's built-in Swagger UI).

## Results

Measured on the held-out test splits (see `results/model_comparison.csv` and `results/finetune/finetune_results.csv`):

| Model | Task | Accuracy | F1 |
|---|---|---|---|
| DistilBERT (fine-tuned) | Intent classification (11 classes) | 96.74% | 0.9669 |
| Logistic Regression (tuned) | Sentiment (3-class) | 75.62% | 0.7562 |
| Decision Tree (tuned) | Sentiment (3-class) | 59.03% | 0.5887 |

The fine-tuned GPT-2 response generator was evaluated in `ai-chatbot.ipynb` by comparing generated responses to held-out expected responses via TF-IDF cosine similarity across 10 sample questions, averaging **64.26%** similarity (range 50.96%–71.45%).

## Future Improvements

- Add a `requirements.txt` / `pyproject.toml` pinning exact dependency versions
- Replace the in-memory `conversations` dict in `app.py` with persistent per-session storage so history survives a server restart
- Add automated tests for the FastAPI routes
- Move the hardcoded `/stats` accuracy figures to be read from `results/model_comparison.csv` and `results/finetune/finetune_results.csv` directly, so they can't drift from the actual latest metrics

## Author

Rahil Parvez Syed
