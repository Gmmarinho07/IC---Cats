from openai import OpenAI
from dotenv import load_dotenv

import json

load_dotenv()

client = OpenAI()

def extract_with_llm(text):

    prompt = f"""
You are a scientific extraction system.

Extract ONLY explicitly stated information.

Return ONLY valid JSON.

If information is missing:
return null.

Schema:

{{
    "catalyst": "",
    "temperature_C": null,
    "pressure_bar": null,
    "conversion_pct": null,
    "main_product": ""
}}

Text:

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
    print(content)

    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    return json.loads(content)