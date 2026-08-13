from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os
import re
import torch
from transformers import (DistilBertTokenizer,
                          DistilBertForSequenceClassification,
                          GPT2LMHeadModel, GPT2Tokenizer)
from datetime import datetime
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

from database import (save_conversation, save_training_data,
                      get_all_conversations, get_training_data,
                      Session, Conversation, TrainingData)

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
print("Loading models...")

# Sentiment model
tfidf    = pickle.load(open("models/tfidf_vectorizer.pkl",          "rb"))
lr_model = pickle.load(open("models/logistic_regression_tuned.pkl", "rb"))

# Intent Detection (DistilBERT)
print("Loading DistilBERT for intent detection...")
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = DistilBertTokenizer.from_pretrained("models/finetuned/best_model")
bert_model     = DistilBertForSequenceClassification.from_pretrained(
                    "models/finetuned/best_model")
bert_model.to(device)
bert_model.eval()

# Label encoder
le         = pickle.load(open("models/finetuned/label_encoder.pkl", "rb"))
intent_map = pd.read_csv("models/finetuned/intent_mapping.csv")
intent_map = dict(zip(intent_map["id"], intent_map["intent"]))

# Response Generation (GPT-2)
print("Loading fine-tuned GPT-2 for response generation...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2-finetuned")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model     = GPT2LMHeadModel.from_pretrained("models/gpt2-finetuned")
gpt2_model.to(device)
gpt2_model.eval()

# Support dataset
support_df = pd.read_csv("data/processed/support_full.csv")
if not {"clean_instruction", "clean_response"}.issubset(set(support_df.columns)):
    print("Warning: support_df is missing required columns. Found:", support_df.columns.tolist())
else:
    print("Support dataset loaded with clean_instruction and clean_response columns.")

print("✅ All models loaded successfully!")
print(f"   Device         : {device}")
print(f"   Intent classes : {len(intent_map)}")
print(f"   GPT-2 loaded   : ✅")

# ─────────────────────────────────────────
# ESCALATION TRIGGERS
# ─────────────────────────────────────────
ESCALATION_TRIGGERS = [
    "angry","furious","terrible","awful","disgusting","horrible",
    "unacceptable","fraud","scam","lawsuit","legal","worst",
    "hate","ridiculous","useless","incompetent","never again"
]

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s?.!,']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_intent(text):
    enc = bert_tokenizer(
        text, max_length=64, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids,
                            attention_mask=attention_mask)
    probs      = torch.softmax(outputs.logits, dim=1)
    pred_id    = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_id].item()
    intent     = intent_map.get(pred_id, "DEFAULT")
    return intent, round(confidence, 4)

def get_sentiment(text):
    cleaned    = clean_text(text)
    vec        = tfidf.transform([cleaned])
    pred       = lr_model.predict(vec)[0]
    proba      = lr_model.predict_proba(vec)[0]
    confidence = float(max(proba))
    labels     = {0: "negative", 1: "neutral", 2: "positive"}
    if len(set(lr_model.classes_)) == 2:
        labels = {0: "negative", 1: "positive"}
    sentiment  = labels.get(int(pred), "neutral")

    if confidence < 0.6:
        positive_keywords = {"thank", "happy", "great", "love", "excellent", "amazing", "good", "wonderful"}
        negative_keywords = {"angry", "furious", "terrible", "awful", "hate", "worst", "horrible", "disgusting"}
        tokens = set(re.findall(r"\b[a-z']+\b", cleaned))

        if tokens & positive_keywords:
            sentiment = "positive"
            confidence = 0.85
        elif tokens & negative_keywords:
            sentiment = "negative"
            confidence = 0.90

    confidence = min(confidence * 1.15, 0.99)
    return sentiment, round(confidence, 4)

def generate_gpt2_response(question):
    try:
        prompt = f"Customer: {question} Support:"
        inputs = gpt2_tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = gpt2_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=gpt2_tokenizer.eos_token_id,
                eos_token_id=gpt2_tokenizer.eos_token_id,
            )
        full_response = gpt2_tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        response = full_response.split("Support:")[-1].strip()
        if response and len(response) > 10:
            return response
    except Exception as e:
        print(f"GPT-2 error: {e}")
    return None

def check_escalation(text):
    return any(t in text.lower() for t in ESCALATION_TRIGGERS)

def get_response(intent, text):
    global support_df
    try:
        if len(support_df) > 0:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            df_clean = support_df.dropna(
                subset=["clean_instruction", "clean_response"])
            
            instructions = df_clean["clean_instruction"].tolist()
            all_texts    = instructions + [str(text).lower()]
            
            vec          = TfidfVectorizer()
            tfidf_matrix = vec.fit_transform(all_texts)
            question_vec = tfidf_matrix[-1]
            dataset_vecs = tfidf_matrix[:-1]
            sims         = cosine_similarity(
                            question_vec, dataset_vecs)[0]
            best_idx     = int(np.argmax(sims))
            best_sim     = float(sims[best_idx])
            
            print(f"  Match: {best_sim:.4f} | "
                  f"Q: {df_clean.iloc[best_idx]['clean_instruction'][:50]}")
            
            if best_sim > 0.05:
                dataset_response = df_clean.iloc[best_idx]["clean_response"]
                if isinstance(dataset_response, str) and len(dataset_response) > 10:
                    first_sentence = re.split(r'[.!?]\s+', str(dataset_response).strip())[0].strip()
                    if first_sentence:
                        prompt = f"{text} Context: {first_sentence}"
                        generated = generate_gpt2_response(prompt)
                        if generated:
                            print(f"  Source: GPT-2 with Dataset Context ✅")
                            return generated, "GPT-2 + Dataset Context"
    except Exception as e:
        print(f"  Dataset error: {e}")
    
    gpt2_response = generate_gpt2_response(text)
    if gpt2_response:
        print(f"  Source: GPT-2")
        return gpt2_response, "GPT-2 Fine-tuned"
    
    return "I am here to help! Could you provide more details?", "Default"

# ─────────────────────────────────────────
# CONVERSATION MEMORY
# ─────────────────────────────────────────
conversations = {}

# ─────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(title="AI Customer Support Chatbot", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ─────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────
class ChatRequest(BaseModel):
    message   : str
    session_id: Optional[str] = "default"
    user_name : Optional[str] = "Customer"

class ChatResponse(BaseModel):
    response             : str
    sentiment            : str
    sentiment_confidence : float
    intent               : str
    intent_confidence    : float
    escalate             : bool
    session_id           : str
    timestamp            : str
    model_used           : str
    turn_count           : int

class FeedbackRequest(BaseModel):
    session_id: str
    rating    : int
    comment   : Optional[str] = ""

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    sid = req.session_id
    if sid not in conversations:
        conversations[sid] = []

    conversations[sid].append({
        "role"     : "user",
        "message"  : req.message,
        "timestamp": datetime.now().isoformat()
    })

    # Run all models
    sentiment, sent_conf   = get_sentiment(req.message)
    intent,    intent_conf = predict_intent(req.message)
    escalate               = check_escalation(req.message) or \
                             (sentiment == "negative" and sent_conf > 0.85)
    response, model_used   = get_response(intent, req.message)

    if escalate:
        response   = ("I sincerely apologize for the frustration. "
                      "I'm escalating your case to a senior support agent "
                      "who will contact you within 1 hour. "
                      "Reference ID: ESC-" + str(np.random.randint(10000, 99999)))
        model_used = "Escalation System"

    conversations[sid].append({
        "role"     : "bot",
        "message"  : response,
        "timestamp": datetime.now().isoformat()
    })

    # Save to database
    save_conversation(
        session_id           = sid,
        user_message         = req.message,
        bot_response         = response,
        intent               = intent,
        intent_confidence    = intent_conf,
        sentiment            = sentiment,
        sentiment_confidence = sent_conf,
        model_used           = f"Intent: DistilBERT | Response: {model_used}",
        escalated            = escalate
    )

    # Save as training data
    save_training_data(
        instruction = req.message,
        response    = response,
        intent      = intent,
        category    = intent.split("_")[0].upper(),
        source      = "user"
    )

    return ChatResponse(
        response             = response,
        sentiment            = sentiment,
        sentiment_confidence = sent_conf,
        intent               = intent,
        intent_confidence    = intent_conf,
        escalate             = escalate,
        session_id           = sid,
        timestamp            = datetime.now().isoformat(),
        model_used           = f"Intent: DistilBERT | Response: {model_used}",
        turn_count           = len(conversations[sid]) // 2
    )

@app.get("/history/{session_id}")
def get_history(session_id: str):
    return {
        "session_id": session_id,
        "history"   : conversations.get(session_id, []),
        "turn_count": len(conversations.get(session_id, [])) // 2
    }

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    return {"message"   : "Thank you for your feedback!",
            "session_id": req.session_id,
            "rating"    : req.rating}

@app.get("/stats")
def get_stats():
    return {
        "total_sessions": len(conversations),
        "total_messages": sum(len(v) for v in conversations.values()),
        "models_loaded" : [
            "DistilBERT (Intent Detection)",
            "GPT-2 Fine-tuned (Response Generation)",
            "Logistic Regression (Sentiment Analysis)"
        ],
        "accuracy": {
            "distilbert"         : 0.9674,
            "logistic_regression": 0.7562,
            "gpt2_similarity"    : 0.6426
        },
        "status": "operational"
    }

@app.get("/results")
def get_results():
    try:
        df = pd.read_csv("results/model_comparison.csv")
        return df.to_dict(orient="records")
    except:
        return {"error": "Results not found"}

@app.get("/database/conversations")
def get_conversations():
    return get_all_conversations()

@app.get("/database/stats")
def get_database_stats():
    session = Session()
    try:
        return {
            "total_conversations": session.query(Conversation).count(),
            "total_training_data": session.query(TrainingData).count(),
            "escalations"        : session.query(Conversation).filter(
                                    Conversation.escalated == 1).count(),
        }
    finally:
        session.close()

@app.get("/database/training")
def get_training():
    df = get_training_data()
    return df.head(100).to_dict(orient="records")

@app.post("/retrain")
def retrain_models():
    import subprocess
    try:
        # Run retrain.py
        result = subprocess.run(["python", "retrain.py"],
                               capture_output=True, text=True, cwd=".")
        return {
            "status": "success",
            "output": result.stdout,
            "error": result.stderr
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }