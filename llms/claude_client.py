import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def generate(
        prompt,
        model="claude-sonnet-4-6",
        temperature=0,
        max_tokens=500
):

    response = client.messages.create(

        model=model,

        temperature=temperature,

        max_tokens=max_tokens,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text