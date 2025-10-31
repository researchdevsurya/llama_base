#!/usr/bin/env python3

# meta-llama/Llama-3-8B-Instruct
"""
College RAG Chatbot - redesigned:
- fast batched embeddings
- normalized vectors + IndexFlatIP (inner product) for cosine search
- persistent FAISS index for DB and a small FAISS memory index for recent Q/A
- logging to file with timestamps and console output
- polite no-data reply when nothing relevant is found
- greeting handling
"""
# ! OWN PACKAGE
from responses import GREETINGS

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
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import faiss
from tqdm import tqdm
from time import sleep

# -----------------------------
# Config
# -----------------------------
CSV_FILE = "colleges.csv"
PROCESSED_CSV = "colleges_processed.csv"
FAISS_INDEX_FILE = "colleges.index"            # main DB index
MEMORY_INDEX_FILE = "memory.index"             # small memory index
MEMORY_FILE = "memory.jsonl"                   # persistent memory (Q/A entries)
LOG_FILE = "college_rag.log"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL_NAME = "intfloat/e5-base-v2"       # high-quality embedding model
GEN_MODEL = "meta-llama/Llama-3-8B-Instruct"
TOP_K = 3                                      # number DB hits
TOP_K_MEM = 3                                  # number memory hits
BATCH_SIZE = 64                                # embedding batch size
MAX_PROMPT_CHARS = 2000
MAX_CONTEXT_CHARS = 1200
MAX_RETRIES = 3
MEMORY_MAX_ENTRIES = 300                       # keep memory small (last N Q/A)
SIMILARITY_THRESHOLD = 0.2                     # inner product threshold (tune)

# -----------------------------
# Logging setup (file + console)
# -----------------------------
logger = logging.getLogger("CollegeRAG")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

# file handler
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info(f"Starting College RAG Bot on device: {DEVICE}")

# -----------------------------
# Utility helpers
# -----------------------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def timeit(fn):
    def wrapper(*a, **k):
        t0 = time.time()
        res = fn(*a, **k)
        logger.info(f"{fn.__name__} took {(time.time()-t0):.3f}s")
        return res
    return wrapper

def safe_truncate(text: str, max_chars: int):
    if len(text) <= max_chars:
        return text
    t = text[:max_chars]
    if " " in t:
        t = t[: t.rfind(" ")]
    return t

# -----------------------------
# Greeting handler
# -----------------------------


def handle_greeting(text: str) -> Optional[str]:
    t = text.strip().lower()
    for key in GREETINGS:
        if t == key or t.startswith(key + " ") or (" " + key + " " in t):
            return GREETINGS[key]
    return None

# -----------------------------
# Load / Build database & FAISS index
# -----------------------------
@timeit
def build_or_load_db():
    """
    Loads CSV, builds combined text column, computes embeddings and builds/saves FAISS index
    """
    if os.path.exists(PROCESSED_CSV) and os.path.exists(FAISS_INDEX_FILE):
        logger.info("Loading preprocessed CSV and FAISS index from disk.")
        df = pd.read_csv(PROCESSED_CSV)
        index = faiss.read_index(FAISS_INDEX_FILE)
        return df, index

    logger.info("Reading CSV and preparing DB.")
    df = pd.read_csv(CSV_FILE)
    # create combined text column (customize fields if you prefer)
    df["combined"] = df.fillna("").astype(str).agg(" | ".join, axis=1)

    logger.info("Computing embeddings for DB (batched).")
    emb_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    texts = ["passage: " + t for t in df["combined"].tolist()]

    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding DB"):
        batch = texts[i : i + BATCH_SIZE]
        emb = emb_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")

    # normalize embeddings => cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product search on normalized vectors = cosine
    index.add(embeddings)

    # save files
    logger.info("Saving processed CSV and FAISS index.")
    df.to_csv(PROCESSED_CSV, index=False)
    faiss.write_index(index, FAISS_INDEX_FILE)

    return df, index

# -----------------------------
# Memory: small persistent Q/A memory with FAISS index
# -----------------------------
def ensure_memory():
    # memory file exists and a memory index is present; load or create
    if not os.path.exists(MEMORY_FILE):
        open(MEMORY_FILE, "w").close()
    # memory index created at runtime; if file exists, load it
    if os.path.exists(MEMORY_INDEX_FILE):
        try:
            mem_index = faiss.read_index(MEMORY_INDEX_FILE)
            logger.info("Loaded memory FAISS index.")
            return mem_index
        except Exception as e:
            logger.warning("Failed to load memory index, rebuilding. " + str(e))
    # create empty memory index (dimension determined later after first embedding)
    return None

def load_memory_entries() -> List[dict]:
    entries = []
    if not os.path.exists(MEMORY_FILE):
        return entries
    with open(MEMORY_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries

def append_memory_entry(entry: dict):
    """
    Append a Q/A entry to memory (persist to file) and keep rolling size.
    entry: {id, question, answer, timestamp}
    """
    entries = load_memory_entries()
    entries.append(entry)
    # keep last MEMORY_MAX_ENTRIES
    entries = entries[-MEMORY_MAX_ENTRIES :]
    with open(MEMORY_FILE, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    # we will rebuild memory index lazily (on next query) for simplicity

# -----------------------------
# Fast embedding helper (batched)
# -----------------------------
class EmbeddingService:
    def __init__(self, model_name=EMBED_MODEL_NAME, device=DEVICE):
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: List[str], batch_size=BATCH_SIZE) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            arr = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embs.append(arr)
        embs = np.vstack(embs).astype("float32")
        faiss.normalize_L2(embs)
        return embs

# -----------------------------
# Text generation pipeline (with retries)
# -----------------------------


@timeit
def load_generation_pipeline():
    device_id = 0 if torch.cuda.is_available() else -1
    logger.info("Loading LLaMA 8B generation model pipeline.")
    
    model_name = GEN_MODEL

    # Load model + tokenizer explicitly for better control
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
        max_new_tokens=350,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    return gen_pipe

# -----------------------------
# Prepare everything
# -----------------------------
df, db_index = build_or_load_db()
embedding_service = EmbeddingService()
# prepare main DB embeddings dimension info
example_emb = embedding_service.embed_texts(["example"])
dim = example_emb.shape[1]
if db_index.ntotal == 0:
    logger.warning("DB index is empty - check your CSV or embedding step.")

# memory index lazy
memory_index = ensure_memory()

# generation pipeline
try:
    gen_pipe = load_generation_pipeline()
except Exception as e:
    logger.error("Failed to load generation model pipeline: " + str(e))
    gen_pipe = None

# -----------------------------
# RAG Query function
# -----------------------------
def ask_bot(query: str, top_k: int = TOP_K, top_k_mem: int = TOP_K_MEM) -> str:
    t_start = time.time()
    logger.info(f"Query received: {query}")

    # greeting short-circuit
    greet_reply = handle_greeting(query)
    if greet_reply:
        logger.info("Greeting handled by greeting module.")
        return greet_reply

    # encode query
    q_text = f"query: {query}"
    q_emb = embedding_service.embed_texts([q_text])  # (1, dim)

    # search DB
    D_db, I_db = db_index.search(q_emb, top_k)  # inner product scores
    contexts = []
    for score, idx in zip(D_db[0], I_db[0]):
        if idx < 0:
            continue
        if score < SIMILARITY_THRESHOLD:
            # below threshold => not relevant: skip
            logger.debug(f"DB hit below threshold (score={score:.3f}) - skipping idx {idx}")
            continue
        text = str(df.iloc[idx]["combined"])
        contexts.append((float(score), text))

    # search memory
    mem_contexts = []
    mem_entries = load_memory_entries()
    if mem_entries:
        # rebuild memory index from mem_entries embeddings on the fly for a small memory set
        mem_texts = [e.get("question","") + " ||| " + e.get("answer","") for e in mem_entries]
        mem_embs = embedding_service.embed_texts([f"memory: {t}" for t in mem_texts])
        mem_index = faiss.IndexFlatIP(dim)
        mem_index.add(mem_embs)
        D_mem, I_mem = mem_index.search(q_emb, top_k_mem)
        for s, idx in zip(D_mem[0], I_mem[0]):
            if idx < 0:
                continue
            if s < SIMILARITY_THRESHOLD:
                continue
            mem_contexts.append((float(s), mem_texts[idx]))
    else:
        logger.debug("No memory entries found.")

    # Merge memory (higher priority) + DB contexts into final context block
    # sort by score descending
    combined = sorted(mem_contexts + contexts, key=lambda x: x[0], reverse=True)
    # take top N contexts (avoid too long)
    if combined:
        ctx_texts = []
        char_count = 0
        for score, txt in combined:
            truncated = safe_truncate(txt, MAX_CONTEXT_CHARS)
            if char_count + len(truncated) > MAX_CONTEXT_CHARS * 2:  # limit overall context size
                break
            ctx_texts.append(f"[score:{score:.3f}] {truncated}")
            char_count += len(truncated)
        context_block = "\n---\n".join(ctx_texts)
    else:
        context_block = ""

    if not context_block:
        # No relevant info found - return polite, non-speculative reply as requested
        logger.info("No relevant context found in DB or memory for this query.")
        answer = (
            "I couldn't find any relevant information in my college database or memory to answer that. "
            "If you'd like, provide more details (college name / program / location / keywords) and I'll try again."
        )
        # record this interaction lightly to memory as unsuccessful lookup (optional)
        append_memory_entry(
            {
                "id": int(time.time() * 1000),
                "question": query,
                "answer": "",
                "timestamp": now_iso(),
                "note": "no_data_found",
            }
        )
        logger.info(f"Responded (no-data) in {(time.time()-t_start):.3f}s")
        return answer

    # Build prompt
    prompt = f"""
You are CollegeBot — concise, professional, and factual.

You have access to the following verified database excerpts and recent memory:
{context_block}

User Question: {query}

Instructions:
- Answer only based on the provided excerpts. Do NOT hallucinate.
- Use plain language, be concise, and helpful.
- If exact info isn't present, say so politely.
-dont repeat the explaoin here or question or any context 
Answer:
""".strip()

    prompt = safe_truncate(prompt, MAX_PROMPT_CHARS)

    # Generate with retry
    response_text = ""
    if gen_pipe is None:
        logger.error("Generation pipeline is not available. Returning fallback message.")
        return (
            "Sorry — the text generation model is not available in this environment. "
            "Please check the model installation and try again."
        )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Generating response (attempt {attempt})")
            # show tiny thinking spinner
            time.sleep(0.2)
            out = gen_pipe(prompt)
            # pipelines may return a list of dicts; 'generated_text' may contain prompt+answer
            generated = out[0].get("generated_text", "") if isinstance(out[0], dict) else str(out[0])
            # Remove prompt prefix if present
            if generated.startswith(prompt):
                answer = generated[len(prompt):].strip()
            else:
                answer = generated.strip()
            # some models may include "Answer:" label - strip it
            # if "Answer:" in answer:
                # answer = answer.split("Answer:")[-1].strip()
            response_text = answer
            break
        except Exception as e:
            logger.warning(f"Generation failed on attempt {attempt}: {e}")
            if attempt == MAX_RETRIES:
                logger.exception("Max retries reached for generation.")
                response_text = (
                    "I'm sorry — I couldn't generate an answer due to an internal error. "
                    "Please try again."
                )
            else:
                sleep(0.8)

    # Save to memory: Store question + answer for future retrieval
    mem_entry = {
        "id": int(time.time() * 1000),
        "question": query,
        "answer": response_text,
        "timestamp": now_iso(),
    }
    append_memory_entry(mem_entry)
    # record in log
    logger.info(f"Generated answer in {(time.time()-t_start):.3f}s; saved to memory id={mem_entry['id']}")

    # final formatting
    return response_text

# -----------------------------
# Interactive loop
# -----------------------------
def main_loop():
    print("🎓 College RAG Chatbot — type 'exit' or 'quit' to stop.")
    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break
        try:
            answer = ask_bot(q)
            print("\n🎯 Answer:")
            print("-" * 60)
            print(answer)
            print("-" * 60 + "\n")
        except Exception as e:
            logger.exception("Unhandled error in chat loop: " + str(e))
            print("⚠️  Something went wrong. Check the log for details.")

if __name__ == "__main__":
    main_loop()
