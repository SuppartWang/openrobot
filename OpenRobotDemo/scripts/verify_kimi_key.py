"""Quick script to verify Kimi API key validity."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("KIMI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ No API key found in environment. Please check your .env file.")
    sys.exit(1)

print(f"🔑 Found API key: {api_key[:10]}...{api_key[-4:]}")
print(f"   Length: {len(api_key)} chars")

from openai import OpenAI

client = OpenAI(
    api_key=api_key,
    base_url="https://api.kimi.com/coding/v1",
    default_headers={"User-Agent": "KimiCLI/1.30.0"},
)

try:
    resp = client.chat.completions.create(
        model="kimi-latest",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
        timeout=15,
    )
    print("✅ Kimi API is working!")
    print("   Response:", resp.choices[0].message.content)
except Exception as e:
    print(f"❌ Kimi API test failed: {type(e).__name__}: {e}")
    sys.exit(1)
