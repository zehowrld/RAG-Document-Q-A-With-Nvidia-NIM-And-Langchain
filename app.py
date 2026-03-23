import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load security keys from .env
load_dotenv()
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_KEY
)

if not NVIDIA_KEY:
  print("❌ ERROR: NVIDIA_API_KEY not found in .env file!")
else:
  print("🚀 Connecting to NVIDIA NIM (Llama-3-70b)...")
  completion = client.chat.completions.create(
    model="meta/llama3-70b-instruct",
    messages=[{"role":"user","content":"Explain the benefit of RAG in one sentence."}],
    temperature=0.5,
    top_p=1,
    max_tokens=1024,
    stream=True
  )

  for chunk in completion:
    if chunk.choices[0].delta.content is not None:
      print(chunk.choices[0].delta.content, end="")

