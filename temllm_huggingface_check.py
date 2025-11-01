import torch
from transformers import pipeline

# ✅ Model ID (Meta official)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# ✅ Load pipeline with optimized settings
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,       # required for Meta models
    use_auth_token=True           # uses your Hugging Face login token
)

# ✅ Define the conversation
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# ✅ Generate response
outputs = pipe(
    messages,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)

# ✅ Print the assistant’s latest reply cleanly
assistant_reply = outputs[0]["generated_text"][-1]["content"]
print("🤖 Pirate Bot:", assistant_reply)
