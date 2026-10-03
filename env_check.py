import os
import classifier

print("GEMINI_API_KEY present:", bool(os.environ.get("GEMINI_API_KEY")))
print("OPENAI_API_KEY present:", bool(os.environ.get("OPENAI_API_KEY")))
