from openai import OpenAI
from dotenv import load_dotenv
import anthropic
import json
import os

load_dotenv()

# =====================================================
# CLIENTES
# =====================================================

gpt_client = OpenAI()

claude_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# =====================================================
# PROMPT
# =====================================================

def build_prompt(text):

    return f"""
Extract catalyst names explicitly mentioned in the abstract.

Rules:
- Use only information present in the text.
- Do not infer.
- Do not guess.
- Return ONLY valid JSON.

Format:

{{
    "catalysts": []
}}

Abstract:

{text}
"""

# =====================================================
# LIMPEZA
# =====================================================

def clean_json(content):

    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    return json.loads(content)

# =====================================================
# GPT
# =====================================================

def gpt_catalyst(text):

    prompt = build_prompt(text)

    response = gpt_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content

    return clean_json(content)

# =====================================================
# CLAUDE
# =====================================================

def claude_catalyst(text):

    prompt = build_prompt(text)

    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.content[0].text

    return clean_json(content)