#!/usr/bin/env python3
"""
College RAG Chatbot - Flask API version
- Embedding + FAISS RAG retrieval
- Persistent memory with FAISS
- Text generation using TinyLlama (or any model)
- Handles greetings, queries, polite fallback, and memory logging
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
from tqdm import tqdm
from flask import Flask, request, jsonify
from flask_cors import CORS
# -----------------------------
# Config
# -----------------------------
CSV_FILE = "colleges.csv"
PROCESSED_CSV = "colleges_processed.csv"
FAISS_INDEX_FILE = "colleges.index"
MEMORY_FILE = "memory.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL_NAME = "intfloat/e5-base-v2"
GEN_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TOP_K = 3
TOP_K_MEM = 3
BATCH_SIZE = 64
MAX_PROMPT_CHARS = 2000
MAX_CONTEXT_CHARS = 1200
SIMILARITY_THRESHOLD = 0.2
MEMORY_MAX_ENTRIES = 300

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger("CollegeRAG")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# -----------------------------
# Utility
# -----------------------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def safe_truncate(text: str, max_chars: int):
    if len(text) <= max_chars:
        return text
    t = text[:max_chars]
    if " " in t:
        t = t[:t.rfind(" ")]
    return t

# -----------------------------
# Greeting Handler
# -----------------------------
GREETINGS = {
    # 1–10: Greetings
    "hi": "Hello! How can I assist you with college or course info today?",
    "hello": "Hi there! Ask me about colleges, courses, or exams.",
    "hey": "Hey! Looking for college or course details?",
    "good morning": "Good morning! Ready to learn something new?",
    "good afternoon": "Good afternoon! What info can I get for you?",
    "good evening": "Good evening! How can I help with college info?",
    "welcome": "Welcome! I’m here to help with your academic questions.",
    "greetings": "Greetings! How may I assist your learning journey?",
    "namaste": "Namaste! How can I support your education goals?",
    "yo": "Hey there! Curious about colleges or courses?",

    # 11–20: Appreciation
    "thank you": "You're very welcome! 😊",
    "thanks": "Glad to help! 👍",
    "thankyou": "No problem! Always here to help.",
    "tysm": "You’re most welcome! Anything else you’d like to know?",
    "thx": "No worries! 😊",
    "appreciate it": "That means a lot! Let me know if you need more info.",
    "grateful": "Happy to assist your learning journey!",
    "thanks a lot": "Anytime! Keep exploring and learning.",
    "thank u": "You’re welcome! 😊",
    "thankyou so much": "You’re most welcome! I’m here whenever you need me.",

    # 21–30: Help & Support
    "help": "Sure! Tell me what you need help with — college, course, or admission?",
    "i need help": "Absolutely! What are you trying to find?",
    "can you help me": "Of course! Ask me about any college or course.",
    "assist me": "I’d be happy to help! What’s your query?",
    "what can you do": "I can help you find info about colleges, courses, and admissions.",
    "help me choose": "Sure! Tell me your interests, and I’ll suggest some colleges or programs.",
    "how to apply": "I can guide you through the application process! Which college or course?",
    "college suggestion": "Got it! Tell me your preferred field or location.",
    "course suggestion": "Sure! What kind of course are you interested in — UG, PG, or diploma?",
    "find college": "I can help with that! Which city or subject are you looking for?",

    # 31–40: Confirmation
    "ok": "Alright! 👍",
    "okay": "Okay! What next?",
    "sure": "Sure! Let's do this.",
    "yes": "Great! Tell me more.",
    "yeah": "Perfect! What do you want to know?",
    "yup": "Got it!",
    "done": "All set! Anything else?",
    "fine": "Good to hear! How can I help now?",
    "great": "Awesome! Let’s continue.",
    "cool": "Cool 😎 Let’s proceed!",

    # 41–50: Clarification Requests
    "what do you mean": "I meant about college or course details. Want me to explain again?",
    "explain": "Sure! Here’s a simpler explanation.",
    "repeat": "Got it! Let me rephrase that.",
    "say again": "No problem! I’ll say it again clearly.",
    "confused": "No worries — I’ll simplify that for you.",
    "not clear": "Let me clarify that for you.",
    "meaning": "Here’s what that means in simple terms.",
    "details": "Sure! Here are the full details.",
    "more info": "Alright! Here’s some more information for you.",
    "tell me more": "Of course! Let’s dive deeper into that.",

    # 51–60: Politeness & Courtesy
    "please": "Of course! Happy to assist.",
    "kindly": "Sure! I’ll take care of that.",
    "sorry": "No worries at all! Let’s fix it together.",
    "my bad": "All good! Happens to everyone.",
    "excuse me": "Yes! How can I assist?",
    "no problem": "Exactly! We’re good to go.",
    "never mind": "Alright! Let’s move on.",
    "its okay": "Glad to hear! 😊",
    "no worries": "Perfect! Let’s continue.",
    "thanks anyway": "Anytime! Wishing you the best.",

    # 61–70: Farewell
    "bye": "Goodbye! 👋 Keep learning and exploring.",
    "goodbye": "Bye! Hope to see you again soon.",
    "see you": "See you later! Keep working hard.",
    "take care": "You too! Stay curious.",
    "catch you later": "Sure! Come back anytime.",
    "see ya": "See ya! 👋",
    "later": "Later! Stay motivated.",
    "peace": "Peace out! ✌️",
    "tata": "Tata! 😊 Study well.",
    "good night": "Good night! Rest well and dream big.",

    # 71–80: Acknowledgment / Small Talk
    "nice": "Glad you liked it!",
    "awesome": "Awesome indeed! 😄",
    "amazing": "Right? It’s pretty cool!",
    "wow": "Wow! That’s exciting!",
    "great job": "Thanks! I try my best.",
    "good": "Good to hear!",
    "perfect": "Perfect! Let’s continue.",
    "interesting": "Yes, it really is!",
    "cool stuff": "Totally! 😎",
    "that helps": "Happy to hear that!",

    # 81–90: General Conversation
    "who are you": "I’m your educational assistant, here to help with college info!",
    "what can you do": "I can help you find colleges, courses, fees, and admission details.",
    "where are you from": "I live in the cloud 🌩️ — always online to help students.",
    "are you a bot": "Yes! A smart educational chatbot built to help learners.",
    "how are you": "I’m great! Ready to help you. How about you?",
    "what’s up": "Just helping students like you! How can I assist?",
    "do you know colleges": "Yes! I know many colleges, courses, and admission details.",
    "tell me a fact": "Did you know? The oldest university in the world is the University of Bologna (1088)!",
    "tell me something": "Sure! Did you know? Learning 15 minutes daily can double your knowledge in a year.",
    "you are smart": "Thank you! I’m learning from you too. 😊",

    # 91–100: Motivation & Learning
    "motivate me": "You’re capable of amazing things — keep going!",
    "study tips": "Study smart: Focus for 25 mins, rest 5 mins, repeat. 🧠",
    "exam tips": "Stay calm, revise well, and sleep enough — confidence is key!",
    "how to focus": "Turn off distractions, set small goals, and reward yourself!",
    "how to learn fast": "Use active recall and spaced repetition — works wonders!",
    "career advice": "Choose a field that excites you — success will follow.",
    "college life": "College is about learning, exploring, and growing — enjoy it!",
    "success tips": "Consistency beats talent — keep at it daily!",
    "daily routine": "Wake early, plan, learn, rest, and repeat — simple formula!",
    "thank you bot": "Always a pleasure! Keep shining in your studies. 🌟",
}


def handle_greeting(text: str) -> Optional[str]:
    t = text.strip().lower()
    for key in GREETINGS:
        if t == key or t.startswith(key + " ") or (" " + key + " " in t):
            return GREETINGS[key]
    return None

# -----------------------------
# Database + FAISS
# -----------------------------
def build_or_load_db():
    if os.path.exists(PROCESSED_CSV) and os.path.exists(FAISS_INDEX_FILE):
        logger.info("Loading processed CSV + FAISS index")
        df = pd.read_csv(PROCESSED_CSV)
        index = faiss.read_index(FAISS_INDEX_FILE)
        return df, index

    logger.info("Building FAISS index from CSV")
    df = pd.read_csv(CSV_FILE)
    df["combined"] = df.fillna("").astype(str).agg(" | ".join, axis=1)

    emb_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    texts = ["passage: " + t for t in df["combined"].tolist()]

    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding DB"):
        batch = texts[i:i+BATCH_SIZE]
        emb = emb_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    df.to_csv(PROCESSED_CSV, index=False)
    faiss.write_index(index, FAISS_INDEX_FILE)
    logger.info("Saved processed CSV and FAISS index")

    return df, index

# -----------------------------
# Memory
# -----------------------------
def load_memory_entries() -> List[dict]:
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

def append_memory_entry(entry: dict):
    entries = load_memory_entries()
    entries.append(entry)
    entries = entries[-MEMORY_MAX_ENTRIES:]
    with open(MEMORY_FILE, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

# -----------------------------
# Embedding Service
# -----------------------------
class EmbeddingService:
    def __init__(self, model_name=EMBED_MODEL_NAME, device=DEVICE):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str]):
        arr = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        arr = arr.astype("float32")
        faiss.normalize_L2(arr)
        return arr

# -----------------------------
# Load everything
# -----------------------------
df, db_index = build_or_load_db()
embedder = EmbeddingService()

try:
    gen_pipe = pipeline(
        "text-generation",
        model=GEN_MODEL,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=250,
        temperature=0.7,
        do_sample=True,
    )
    logger.info("Generation model loaded successfully.")
except Exception as e:
    gen_pipe = None
    logger.error(f"Failed to load generation model: {e}")

# -----------------------------
# RAG Function
# -----------------------------
def ask_bot(query: str):
    greet = handle_greeting(query)
    if greet:
        return greet

    q_emb = embedder.embed([f"query: {query}"])
    D, I = db_index.search(q_emb, TOP_K)
    contexts = []

    for score, idx in zip(D[0], I[0]):
        if score < SIMILARITY_THRESHOLD or idx < 0:
            continue
        contexts.append(df.iloc[idx]["combined"])

    if not contexts:
        return "I couldn’t find relevant info. Please give more details (college name / course / location)."

    context_block = "\n---\n".join([safe_truncate(c, MAX_CONTEXT_CHARS) for c in contexts])

    prompt = f"""
You are CollegeBot — factual and concise.
Use only the context below to answer accurately.

Context:
{context_block}

Question: {query}

Answer:
""".strip()

    if not gen_pipe:
        return "Generation model not loaded. Please check your setup."

    out = gen_pipe(prompt)
    result = out[0]["generated_text"].replace(prompt, "").strip()

    append_memory_entry({
        "id": int(time.time()*1000),
        "question": query,
        "answer": result,
        "timestamp": now_iso(),
    })

    return result

# -----------------------------
# Flask API
# -----------------------------
app = Flask(__name__)
CORS(app) 
@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "🎓 College RAG Chatbot API running!"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        answer = ask_bot(query)
        return jsonify({"query": query, "answer": answer})
    except Exception as e:
        logger.exception("Error during query processing")
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
