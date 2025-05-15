env_vars = [
    'OPENAI_API_KEY',
    'MONGO_DB_NAME',
    'MONGO_URL',
    'NEO4J_URI',
    'NEO4J_USERNAME',
    'NEO4J_PASSWORD',
    "NEO4J_GPT_MODEL",
    "CHAT_GPT_MODEL",
    ]

import os
from dotenv import load_dotenv

load_dotenv()

for var in env_vars:
    print(var, " - ", os.environ[var])