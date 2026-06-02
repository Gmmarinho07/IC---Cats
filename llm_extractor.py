from openai import OpenAI
from dotenv import load_dotenv
import json
import anthropic

load_dotenv()

client = OpenAI()


def agent_1_catalyst(text):

    prompt = f"""
Extract catalyst names explicitly mentioned in the abstract.

Rules:
- Return only valid JSON
- Do not explain
- Do not infer
- Do not guess

Format:

{{
  "catalysts": []
}}

Abstract:

{text}
"""

    response = client.chat.completions.create(
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

    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    return json.loads(content)


def agent_2_catalyst(text):

    prompt = f"""
You are a catalyst extraction system.

Identify catalyst materials explicitly used or evaluated in the study.

Rules:
- Use only information present in the abstract
- Ignore reaction products
- Ignore supports unless they are part of the catalyst name
- Do not infer abbreviations
- Return only valid JSON

Format:

{{
  "catalysts": []
}}

Abstract:

{text}
"""

    response = client.chat.completions.create(
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

    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    return json.loads(content)

def agent_3_metal_support(text):

    prompt = f"""
You are a catalyst decomposition system.

Extract ONLY the metal phase and catalyst support explicitly mentioned in the abstract.

Rules:
- Use only information present in the text.
- Do not infer.
- Do not guess.
- If a field is missing, return null.
- Return ONLY valid JSON.

Examples:

Catalyst:
Ni/HAP

Output:
{{
    "metal": ["Ni"],
    "support": "HAP"
}}

Catalyst:
Cu-Ni/Al2O3

Output:
{{
    "metal": ["Cu", "Ni"],
    "support": "Al2O3"
}}

Abstract:

{text}
"""

    response = client.chat.completions.create(
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

    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    return json.loads(content)