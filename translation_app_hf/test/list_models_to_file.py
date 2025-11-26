import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found.")
else:
    genai.configure(api_key=api_key)
    with open("available_models.txt", "w") as f:
        for m in genai.list_models():
            f.write(f"{m.name}\n")
    print("Models written to available_models.txt")
