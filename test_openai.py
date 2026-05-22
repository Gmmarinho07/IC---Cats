from dotenv import load_dotenv
from openai import OpenAI
import os

# carrega o .env
load_dotenv()

# pega a chave
api_key = os.getenv("OPENAI_API_KEY")

# cria cliente
client = OpenAI(api_key=api_key)

# faz requisição
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": "Olá"}
    ]
)

print(response.choices[0].message.content)