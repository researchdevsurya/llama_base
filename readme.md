# College RAG Chatbot - Code Workflow

## 📋 Overview
A Retrieval-Augmented Generation (RAG) chatbot that answers questions about colleges using a CSV database, vector search, and AI text generation.

---

## 🔧 System Architecture

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  1. GREETING CHECK              │
│  Is it a greeting/common phrase?│
└────────┬───────────────┬────────┘
         │ Yes           │ No
         ▼               ▼
    ┌────────┐    ┌──────────────┐
    │ Return │    │ 2. EMBEDDING │
    │Greeting│    │ Convert query│
    └────────┘    │ to vector    │
                  └──────┬───────┘
                         │
        ┌────────────────┴────────────────┐
        │                                  │
        ▼                                  ▼
┌──────────────────┐            ┌──────────────────┐
│ 3a. SEARCH DB    │            │ 3b. SEARCH MEMORY│
│ Find similar     │            │ Find similar Q/A │
│ college records  │            │ from past chats  │
└────────┬─────────┘            └────────┬─────────┘
         │                               │
         └───────────┬───────────────────┘
                     │
                     ▼
         ┌────────────────────┐
         │ 4. FILTER & MERGE  │
         │ Keep only relevant │
         │ results (threshold)│
         └─────────┬──────────┘
                   │
         ┌─────────┴──────────┐
         │ No                  │ Yes
         ▼                     ▼
    ┌────────────┐    ┌────────────────┐
    │ Return     │    │ 5. BUILD PROMPT│
    │"No Data"   │    │ Context + Query│
    │ Message    │    └────────┬───────┘
    └────────────┘             │
                               ▼
                    ┌───────────────────┐
                    │ 6. GENERATE ANSWER│
                    │ AI Model produces │
                    │ response text     │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ 7. SAVE TO MEMORY │
                    │ Store Q/A for     │
                    │ future retrieval  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ 8. RETURN ANSWER  │
                    │ Display to user   │
                    └───────────────────┘
```

---

## 🚀 Step-by-Step Workflow

### **INITIALIZATION PHASE** (Runs once at startup)

#### Step 1: Load Configuration
- Set file paths (CSV, FAISS indexes, logs)
- Configure models (embedding, generation)
- Set parameters (top-k results, batch size, thresholds)

#### Step 2: Setup Logging
- Create file logger → `college_rag.log`
- Create console logger → Terminal output
- Log timestamps and events

#### Step 3: Build/Load Database
```
IF processed CSV + FAISS index exist:
    → Load from disk (fast)
ELSE:
    → Read colleges.csv
    → Combine all columns into "combined" text
    → Generate embeddings (batched for speed)
    → Normalize vectors (for cosine similarity)
    → Build FAISS index (IndexFlatIP)
    → Save to disk for future use
```

#### Step 4: Initialize Memory System
```
IF memory.jsonl exists:
    → Load past Q/A entries
IF memory.index exists:
    → Load memory FAISS index
ELSE:
    → Create empty memory (builds on first query)
```

#### Step 5: Load AI Models
- **Embedding Model**: `intfloat/e5-base-v2` (converts text → vectors)
- **Generation Model**: `TinyLlama-1.1B-Chat` (generates answers)

---

### **QUERY PROCESSING PHASE** (Runs for each user question)

#### Step 6: Receive User Query
```python
User: "What are the top engineering colleges in Mumbai?"
```

#### Step 7: Check for Greetings
```
IF query matches greeting patterns:
    → Return pre-defined friendly response
    → SKIP to Step 14 (no RAG needed)
```

**Greeting Examples:**
- "hi" → "Hello! How can I assist you?"
- "thank you" → "You're very welcome! 😊"
- "bye" → "Goodbye! 👋 Keep learning."

#### Step 8: Embed Query
```python
query_text = "query: What are the top engineering colleges in Mumbai?"
query_vector = embedding_service.embed_texts([query_text])
# Result: [0.123, -0.456, 0.789, ...] (384-dim vector)
```

#### Step 9: Search Database (FAISS)
```
1. Search DB index with query vector
2. Get top-k results (default: 3)
3. Each result has:
   - Score (similarity: 0.0 to 1.0)
   - Index (row in CSV)
   
Example:
  [Score: 0.85] → Row 42: "IIT Bombay | Mumbai | Engineering | ..."
  [Score: 0.72] → Row 108: "VJTI | Mumbai | Engineering | ..."
  [Score: 0.68] → Row 215: "SPIT | Mumbai | Engineering | ..."
```

#### Step 10: Search Memory (Recent Q/A)
```
1. Load past Q/A entries from memory.jsonl
2. Rebuild memory FAISS index (small, so fast)
3. Search memory with query vector
4. Get top-k_mem results (default: 3)

Example:
  [Score: 0.91] → "Q: Best engineering colleges? A: IIT Bombay ranks..."
  [Score: 0.76] → "Q: Mumbai colleges? A: VJTI is highly regarded..."
```

#### Step 11: Filter Results by Threshold
```
SIMILARITY_THRESHOLD = 0.2 (configurable)

For each result (DB + Memory):
    IF score >= 0.2:
        → Keep it
    ELSE:
        → Discard (not relevant enough)
```

#### Step 12: Build Context Block
```
1. Merge DB + Memory results
2. Sort by score (highest first)
3. Truncate texts to avoid overflow
4. Create formatted context:

Context:
---
[score:0.91] Q: Best engineering colleges? A: IIT Bombay ranks #1...
---
[score:0.85] IIT Bombay | Mumbai | Engineering | Fees: 2L/year...
---
[score:0.76] Q: Mumbai colleges? A: VJTI is highly regarded...
---
```

#### Step 13: Generate Answer

**IF NO CONTEXT FOUND:**
```python
return "I couldn't find any relevant information in my database. 
        Please provide more details (college name/program/location)."
```

**IF CONTEXT FOUND:**
```
1. Build prompt:
   - System instruction: "You are CollegeBot, concise and factual"
   - Context block (from Step 12)
   - User question
   - Rules: "Answer only from context, don't hallucinate"

2. Send to AI generation model
3. Retry up to 3 times if fails
4. Extract answer from model output
5. Clean up formatting
```

**Example Prompt:**
```
You are CollegeBot — concise, professional, and factual.

Context:
[score:0.85] IIT Bombay | Mumbai | Engineering | Fees: 2L/year...
[score:0.72] VJTI | Mumbai | Engineering | Fees: 80K/year...

User Question: What are the top engineering colleges in Mumbai?

Instructions:
- Answer only based on the provided context
- Be concise and helpful
- Don't hallucinate

Answer:
```

**Model Output:**
```
Based on the data, the top engineering colleges in Mumbai include:
1. IIT Bombay - Premier institution with ₹2L/year fees
2. VJTI - Highly regarded with ₹80K/year fees
```

#### Step 14: Save to Memory
```python
memory_entry = {
    "id": 1698765432000,
    "question": "What are the top engineering colleges in Mumbai?",
    "answer": "Based on the data, the top engineering colleges...",
    "timestamp": "2025-10-30T10:30:45Z"
}

→ Append to memory.jsonl
→ Keep only last 300 entries (rolling window)
```

#### Step 15: Return Answer to User
```
🎯 Answer:
------------------------------------------------------------
Based on the data, the top engineering colleges in Mumbai include:
1. IIT Bombay - Premier institution with ₹2L/year fees
2. VJTI - Highly regarded with ₹80K/year fees
------------------------------------------------------------
```

---

## 🗂️ Data Flow Summary

```
CSV File (colleges.csv)
    ↓ [Read & Process]
Preprocessed CSV (colleges_processed.csv)
    ↓ [Generate Embeddings]
FAISS Index (colleges.index) → Fast Vector Search
    ↓
User Query → Embedding → Search → Context
    ↓
AI Model → Generate Answer
    ↓
memory.jsonl (Q/A History) → Future Retrieval
```

---

## 🛠️ Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Embeddings** | SentenceTransformer (e5-base-v2) | Convert text → vectors |
| **Vector Search** | FAISS (IndexFlatIP) | Fast similarity search |
| **Generation** | TinyLlama-1.1B-Chat | Generate natural answers |
| **Database** | Pandas + CSV | Store college data |
| **Memory** | JSONL + FAISS | Store past Q/A |
| **Logging** | Python logging | Track events |

---

## 📊 Performance Features

1. **Batched Embeddings**: Process 64 texts at once (faster than one-by-one)
2. **Normalized Vectors**: Use cosine similarity via inner product (efficient)
3. **Persistent Indexes**: Load pre-built FAISS indexes (skip re-computation)
4. **Threshold Filtering**: Discard low-quality results (cleaner answers)
5. **Memory System**: Learn from past conversations (improves over time)
6. **Retry Logic**: Handle generation failures gracefully

---

## 🎯 Usage Example

```bash
$ python college_rag_chatbot.py

🎓 College RAG Chatbot — type 'exit' or 'quit' to stop.

You: hi
🎯 Answer: Hello! How can I assist you with college or course info today?

You: What are the fees for IIT Bombay?
[Searches DB → Finds IIT Bombay record → Generates answer]
🎯 Answer: IIT Bombay's tuition fees are approximately ₹2 lakh per year...

You: thank you
🎯 Answer: You're very welcome! 😊

You: exit
👋 Goodbye!
```

---

## 📝 Configuration Guide

### Adjust Performance
```python
TOP_K = 3              # More results = better context (slower)
BATCH_SIZE = 64        # Larger = faster (needs more memory)
SIMILARITY_THRESHOLD = 0.2  # Higher = stricter (fewer results)
```

### Change Models
```python
EMBED_MODEL_NAME = "intfloat/e5-base-v2"  # Swap for different embeddings
GEN_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Use larger LLMs
```

### Memory Settings
```python
MEMORY_MAX_ENTRIES = 300  # Keep last N Q/A pairs
TOP_K_MEM = 3            # How many memory results to retrieve
```

---

## 🐛 Error Handling

1. **No Data Found**: Returns polite "no information" message
2. **Generation Failure**: Retries 3 times, then shows error message
3. **Index Missing**: Rebuilds from CSV automatically
4. **Memory Corruption**: Rebuilds memory index on next query

---

## 📂 File Structure

```
project/
├── college_rag_chatbot.py      # Main code
├── colleges.csv                # Input data
├── colleges_processed.csv      # Preprocessed data
├── colleges.index              # DB FAISS index
├── memory.jsonl                # Q/A history
├── memory.index                # Memory FAISS index
└── college_rag.log             # Logs
```

---

## 🎓 Summary

This chatbot uses **RAG (Retrieval-Augmented Generation)** to answer college-related questions:

1. **Retrieve** relevant info from database + memory
2. **Augment** the query with context
3. **Generate** natural language answer using AI

The system is fast (batched operations), persistent (saves indexes), and learns over time (memory system). Perfect for building educational assistants! 🚀