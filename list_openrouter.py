
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("No OpenRouter API Key found")
    exit(1)

response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

if response.status_code == 200:
    models = response.json()['data']
    print(f"Found {len(models)} models")
    for m in models:
        # filter for free models or gemini
        if 'free' in m['id'] or 'gemini' in m['id']:
            print(f"- {m['id']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
