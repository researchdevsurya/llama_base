### Replace the small `TinyLlama` model in your **College RAG Chatbot** with a **LLaMA 8B** model (e.g., *Meta-Llama-3-8B-Instruct*).


---

### 🧠 Step 1: Choose the Right LLaMA 8B Model

Depending on your environment and what you can run:

| Model                                   | Hugging Face ID                                           | Notes |
| --------------------------------------- | --------------------------------------------------------- | ----- |
| `meta-llama/Llama-3-8B-Instruct`        | Official Meta LLaMA 3 Instruct (best choice if available) |       |
| `NousResearch/Meta-Llama-3-8B-Instruct` | Mirror, often faster to download                          |       |
| `meta-llama/Llama-2-8b-chat-hf`         | LLaMA 2 version (fallback if you can’t use LLaMA 3)       |       |

---

### 🧩 Step 2: Update the Model in Your Script

Find this line:

```python
GEN_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

Change it to:

```python
GEN_MODEL = "meta-llama/Llama-3-8B-Instruct"
```

---

### ⚙️ Step 3: Adjust the Text Generation Pipeline

Right now, your generation pipeline is created like this:

```python
gen_pipe = pipeline(
    "text-generation",
    model=GEN_MODEL,
    device=device_id,
    max_new_tokens=250,
    temperature=0.7,
    do_sample=True,
)
```

That’s fine for `TinyLlama`, but for **LLaMA-3-8B**, you should use the `"text-generation"` **task** with **bfloat16** or **float16** precision to avoid running out of memory.

Here’s an improved version:

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

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
```

---

### 💾 Step 4: (Optional) Adjust Prompt Style

LLaMA models respond best to structured chat prompts.
You can wrap your existing `prompt` like this:

```python
prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are CollegeBot — concise, professional, and factual. You answer user queries only using the provided context.
<|eot_id|><|start_header_id|>user<|end_header_id|>
User Question: {query}
Here is relevant context:
{context_block}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""".strip()
```

That will help LLaMA 3 stay in role.

---

### 🧮 Step 5: Ensure You Have Enough GPU Memory

* LLaMA 8B typically requires **at least 16 GB of VRAM** (e.g., RTX 4090, A100, or cloud instance).
* On smaller GPUs, use **quantized versions** (for example, 4-bit or 8-bit):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
```

That lets you run LLaMA 8B on ~8–12GB GPUs.

---

### ✅ Summary of Key Edits

| Section                      | Change                                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| `GEN_MODEL`                  | `"meta-llama/Llama-3-8B-Instruct"`                                                             |
| `load_generation_pipeline()` | Use `AutoModelForCausalLM`, `AutoTokenizer`, `torch_dtype=torch.bfloat16`, `device_map="auto"` |
| Prompt format                | Use chat-friendly structure for better LLaMA 3 output                                          |
| GPU tips                     | Use 4-bit quantization if needed (`BitsAndBytesConfig`)                                        |

---

