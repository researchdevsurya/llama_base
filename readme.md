

````markdown
# 🎓 TEM RAG Chatbot - Developer Explanation

This project is a **Retrieval-Augmented Generation (RAG) chatbot** that answers college-related questions.  
It combines a **FAISS-based search** for retrieving relevant data from a CSV file and a **TinyLlama text-generation model** for generating natural language answers.

---

## 🧠 What This Bot Does

- Loads a **colleges dataset** (CSV file).  
- Converts each record into embeddings using **SentenceTransformer**.  
- Stores those embeddings inside a **FAISS index** (for fast search).  
- When a user asks a question:
  - It finds similar text from the database.
  - Adds context from previous memory.
  - Sends all this to a text generation model (TinyLlama) to form a clean, factual answer.
- Handles **greetings and polite messages**.
- Keeps a **small memory** of past Q/A.
- Logs all activity with timestamps.

---

## ⚙️ Configuration Section

```python
CSV_FILE = "colleges.csv"
PROCESSED_CSV = "colleges_processed.csv"
FAISS_INDEX_FILE = "colleges.index"
MEMORY_INDEX_FILE = "memory.index"
MEMORY_FILE = "memory.jsonl"
LOG_FILE = "college_rag.log"
````

* These are all filenames used to store processed data and logs.
* `FAISS_INDEX_FILE` stores vector embeddings.
* `MEMORY_FILE` stores Q/A memory.

**Models:**

```python
EMBED_MODEL_NAME = "intfloat/e5-base-v2"
GEN_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

* `e5-base-v2` is used for embeddings.
* `TinyLlama` generates text answers.

---

## 🧩 Logging Setup

The bot logs every event both in the **console** and a **log file**.

```python
logger = logging.getLogger("CollegeRAG")
fh = logging.FileHandler(LOG_FILE)
ch = logging.StreamHandler()
```

This helps track:

* How long tasks take.
* What queries were asked.
* Any errors or warnings.

---

## 🧾 Utility Functions

### `now_iso()`

Returns the current time in ISO format.

### `timeit(fn)`

Decorator to log how long a function takes to run.

### `safe_truncate(text, max_chars)`

Truncates long text neatly (so it doesn’t break words).

---

## 💬 Greeting Handler

The bot can recognize over **100 common phrases** like:

* “hi”, “hello”, “thanks”, “bye”, “help”, “how are you”, etc.

and reply with friendly messages.

### Example

```python
if user says "hi" → bot says "Hello! How can I assist you with college info today?"
```

Handled by:

```python
def handle_greeting(text: str)
```

---

## 🧮 Database & FAISS Index

### Function: `build_or_load_db()`

This part:

1. Loads the CSV file.
2. Combines all columns into one big text field.
3. Creates embeddings in batches using the SentenceTransformer model.
4. Normalizes embeddings for cosine similarity.
5. Saves everything into a FAISS index for fast retrieval.

**FAISS** = Facebook AI Similarity Search
It’s super fast at finding the most similar text.

---

## 🧠 Memory System

### Files Used:

* `memory.jsonl` → Stores past Q/A.
* `memory.index` → Vector index for memory retrieval.

### Functions:

* `ensure_memory()` – Makes sure memory files exist.
* `load_memory_entries()` – Loads past Q/A.
* `append_memory_entry()` – Saves new Q/A to memory.

This lets the bot “remember” previous answers temporarily.

---

## ⚡ Embedding Service

Class: `EmbeddingService`

This handles encoding (creating embeddings) efficiently in batches.

```python
class EmbeddingService:
    def embed_texts(self, texts):
        ...
```

Every piece of text (like a college record or a user question) is converted into a **vector of numbers** that represents its meaning.

---

## ✍️ Text Generation

### Function: `load_generation_pipeline()`

Loads the **TinyLlama** text-generation model using the `transformers` pipeline.

* `temperature=0.7` → controls creativity.
* `max_new_tokens=250` → limits output size.

If CUDA (GPU) is available, it runs faster.

---

## 🔍 ask_bot() – The Core Function

This is the **main brain** of the chatbot.

### Steps:

1. **Greeting Check**
   If the user says "hello" or "thanks", reply instantly using the greeting dictionary.

2. **Embedding Query**
   Converts the user’s question into a vector.

3. **FAISS Search**
   Finds the top relevant database records and memory entries.

4. **Context Building**
   Joins the found records into a short readable context block.

5. **No Match Case**
   If nothing relevant is found, gives a polite message:

   > “I couldn’t find any relevant information. Please provide more details.”

6. **Prompt Creation**
   Builds a clean instruction prompt for the LLM:

   * Only use verified data.
   * Be factual.
   * No repetition or hallucination.

7. **Model Generation**
   Sends the prompt to the generation model (`TinyLlama`).
   Retries up to 3 times if there’s an error.

8. **Save to Memory**
   Stores question + answer for future searches.

---

## 🧑‍💻 Main Loop (Interactive Mode)

### Function: `main_loop()`

This keeps the bot running in the terminal.

```python
while True:
    q = input("You: ")
    answer = ask_bot(q)
    print(answer)
```

Type `exit` or `quit` to stop the bot.

---

## 🗂️ File Outputs Summary

| File Name                | Purpose                                   |
| ------------------------ | ----------------------------------------- |
| `colleges.csv`           | Main college dataset                      |
| `colleges_processed.csv` | Preprocessed dataset with combined text   |
| `colleges.index`         | FAISS index for DB embeddings             |
| `memory.jsonl`           | Stores previous Q/A for short-term memory |
| `memory.index`           | FAISS index for memory                    |
| `college_rag.log`        | Logs all events and errors                |

---

## 💡 Developer Notes

* **Batch embedding** improves speed for large CSVs.
* **FAISS.normalize_L2()** enables cosine similarity search.
* **SIMILARITY_THRESHOLD** (default 0.2) controls how relevant matches must be.
* **Memory entries** are limited to 300 to avoid slowing down the bot.
* **TinyLlama** is a small and fast model, perfect for local usage.

---

## 🧰 Requirements

Install all dependencies:

```bash
pip install pandas numpy torch sentence-transformers transformers faiss-cpu tqdm
```

(Optional GPU version):

```bash
pip install faiss-gpu
```

Run the bot:

```bash
python college_rag_bot.py
```

---

## 🧾 Summary

✅ Uses **SentenceTransformer** for semantic search
✅ Uses **FAISS** for vector similarity
✅ Uses **TinyLlama** for answer generation
✅ Handles greetings and polite replies
✅ Maintains small memory for better context
✅ Fully runs offline (if models are cached)

---

### 👨‍💻 Developer Tip

If you want to turn this into an API, you can easily wrap the `ask_bot()` function inside a Flask or FastAPI endpoint.

---

**Author:** Surya
**Project:** College RAG Chatbot (Local Intelligent Assistant for Colleges)

