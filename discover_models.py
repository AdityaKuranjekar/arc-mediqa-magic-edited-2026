import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print(f"🔍 Connecting to Google GenAI with key: {api_key[:10]}...")

try:
    client = genai.Client(api_key=api_key)
    print("\n✅ Available Models for your Project:")
    print("-" * 50)
    for model in client.models.list():
        # Strip the 'models/' prefix for your GEMINI_MODEL variable
        clean_name = model.name.replace("models/", "")
        print(f"ID: {clean_name:<30} | Supported: {model.supported_actions}")
    print("-" * 50)
except Exception as e:
    print(f"❌ Error listing models: {e}")
