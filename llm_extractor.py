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
# PROMPTS
# =====================================================

def build_catalyst_prompt(text):

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


def build_metal_support_prompt(text):

    return f"""
Extract the active metal(s) and catalyst support explicitly mentioned in the abstract.

Definitions

Active metal:
The catalytically active metallic element
(examples: Ru, Cu, Ni, Pd, Pt, Ag, Co).

Support:
The material supporting the active metal
(examples: MgO, Al2O3, SiO2, Hydroxyapatite, MgAl-LDO).

Rules

- Use ONLY information explicitly present in the abstract.
- Do NOT infer.
- Do NOT guess.
- If the catalyst is written as Ru/MgO:
    metal = ["Ru"]
    support = "MgO"
- If there are multiple metals, return all of them.
- If no active metal exists, return [].
- If no support is explicitly identifiable, return null.
- Return ONLY valid JSON.
Do not provide explanations.

Do not think aloud.

Do not revise your answer.

Produce exactly one JSON object and stop immediately after the closing brace.

Format

{{
    "metal": [],
    "support": null
}}

Abstract:

{text}
"""


# =====================================================
# LIMPEZA
# =====================================================

import json
import re

def clean_json(content):

    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    decoder = json.JSONDecoder()

    match = re.search(r"\{", content)

    if not match:
        raise ValueError("Nenhum JSON encontrado.")

    start = match.start()

    obj, _ = decoder.raw_decode(content[start:])

    return obj

# =====================================================
# GPT - AGENT 1
# =====================================================

def gpt_catalyst(text):

    prompt = build_catalyst_prompt(text)

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

    return clean_json(
        response.choices[0].message.content
    )


# =====================================================
# GPT - AGENT 2
# =====================================================

def gpt_metal_support(text):

    prompt = build_metal_support_prompt(text)

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

    return clean_json(
        response.choices[0].message.content
    )


# =====================================================
# CLAUDE - AGENT 1
# =====================================================

def claude_catalyst(text):

    prompt = build_catalyst_prompt(text)

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
    

    return clean_json(
        response.content[0].text
    )


# =====================================================
# CLAUDE - AGENT 2
# =====================================================

def claude_metal_support(text):

    prompt = build_metal_support_prompt(text)

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