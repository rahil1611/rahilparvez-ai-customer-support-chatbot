from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
import os

# ─────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────
Base = declarative_base()
engine = create_engine("sqlite:///chatbot.db", echo=False)
Session = sessionmaker(bind=engine)

# ─────────────────────────────────────────
# TABLES
# ─────────────────────────────────────────
class Conversation(Base):
    __tablename__ = "conversations"
    id                   = Column(Integer, primary_key=True)
    session_id           = Column(String(100))
    user_message         = Column(Text)
    bot_response         = Column(Text)
    intent               = Column(String(100))
    intent_confidence    = Column(Float)
    sentiment            = Column(String(50))
    sentiment_confidence = Column(Float)
    model_used           = Column(String(200))
    escalated            = Column(Integer, default=0)
    timestamp            = Column(DateTime, default=datetime.now)

class TrainingData(Base):
    __tablename__ = "training_data"
    id           = Column(Integer, primary_key=True)
    instruction  = Column(Text)
    response     = Column(Text)
    intent       = Column(String(100))
    category     = Column(String(100))
    source       = Column(String(50), default="user")
    timestamp    = Column(DateTime, default=datetime.now)

class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    id         = Column(Integer, primary_key=True)
    model_name = Column(String(100))
    accuracy   = Column(Float)
    f1_score   = Column(Float)
    precision  = Column(Float)
    recall     = Column(Float)
    timestamp  = Column(DateTime, default=datetime.now)

# ─────────────────────────────────────────
# CREATE TABLES
# ─────────────────────────────────────────
Base.metadata.create_all(engine)
print("✅ Database tables created!")

# ─────────────────────────────────────────
# DATABASE FUNCTIONS
# ─────────────────────────────────────────
def save_conversation(session_id, user_message, bot_response,
                      intent, intent_confidence, sentiment,
                      sentiment_confidence, model_used, escalated=False):
    session = Session()
    try:
        conv = Conversation(
            session_id           = session_id,
            user_message         = user_message,
            bot_response         = bot_response,
            intent               = intent,
            intent_confidence    = intent_confidence,
            sentiment            = sentiment,
            sentiment_confidence = sentiment_confidence,
            model_used           = model_used,
            escalated            = int(escalated),
            timestamp            = datetime.now()
        )
        session.add(conv)
        session.commit()
        return conv.id
    except Exception as e:
        session.rollback()
        print(f"DB Error: {e}")
        return None
    finally:
        session.close()

def save_training_data(instruction, response, intent, category, source="user"):
    session = Session()
    try:
        data = TrainingData(
            instruction = instruction,
            response    = response,
            intent      = intent,
            category    = category,
            source      = source,
            timestamp   = datetime.now()
        )
        session.add(data)
        session.commit()
        return data.id
    except Exception as e:
        session.rollback()
        print(f"DB Error: {e}")
        return None
    finally:
        session.close()

def save_model_metrics(model_name, accuracy, f1_score, precision, recall):
    session = Session()
    try:
        metrics = ModelMetrics(
            model_name = model_name,
            accuracy   = accuracy,
            f1_score   = f1_score,
            precision  = precision,
            recall     = recall,
            timestamp  = datetime.now()
        )
        session.add(metrics)
        session.commit()
        return metrics.id
    except Exception as e:
        session.rollback()
        print(f"DB Error: {e}")
        return None
    finally:
        session.close()

def get_all_conversations():
    session = Session()
    try:
        convs = session.query(Conversation).all()
        return [{
            "id"                  : c.id,
            "session_id"          : c.session_id,
            "user_message"        : c.user_message,
            "bot_response"        : c.bot_response,
            "intent"              : c.intent,
            "intent_confidence"   : c.intent_confidence,
            "sentiment"           : c.sentiment,
            "sentiment_confidence": c.sentiment_confidence,
            "model_used"          : c.model_used,
            "escalated"           : c.escalated,
            "timestamp"           : str(c.timestamp)
        } for c in convs]
    finally:
        session.close()

def get_training_data():
    session = Session()
    try:
        data = session.query(TrainingData).all()
        return pd.DataFrame([{
            "instruction": d.instruction,
            "response"   : d.response,
            "intent"     : d.intent,
            "category"   : d.category,
            "source"     : d.source,
            "timestamp"  : str(d.timestamp)
        } for d in data])
    finally:
        session.close()

def get_stats():
    session = Session()
    try:
        total_convs    = session.query(Conversation).count()
        total_training = session.query(TrainingData).count()
        escalations    = session.query(Conversation).filter(
                            Conversation.escalated == 1).count()
        intents        = session.query(
                            Conversation.intent,
                            sqlalchemy.func.count(Conversation.intent)
                         ).group_by(Conversation.intent).all()
        return {
            "total_conversations": total_convs,
            "total_training_data": total_training,
            "total_escalations"  : escalations,
            "intent_distribution": dict(intents)
        }
    finally:
        session.close()

def load_existing_dataset():
    """Load existing dataset into training_data table"""
    print("Loading existing dataset into database...")
    df = pd.read_csv("data/processed/support_full.csv")
    df.dropna(subset=["clean_instruction", "clean_response"], inplace=True)

    session = Session()
    count   = 0
    try:
        for _, row in df.iterrows():
            data = TrainingData(
                instruction = row["clean_instruction"],
                response    = row["clean_response"],
                intent      = row.get("intent", "unknown"),
                category    = row.get("category", "unknown"),
                source      = "dataset",
                timestamp   = datetime.now()
            )
            session.add(data)
            count += 1
            if count % 1000 == 0:
                session.commit()
                print(f"  Saved {count} records...")
        session.commit()
        print(f"✅ Loaded {count} records into database!")
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()

# Load existing dataset on first run
import sqlalchemy
if __name__ == "__main__":
    load_existing_dataset()
    print("\n📊 Database Stats:")
    session = Session()
    print(f"  Training data: {session.query(TrainingData).count()} records")
    print(f"  Conversations: {session.query(Conversation).count()} records")
    session.close()