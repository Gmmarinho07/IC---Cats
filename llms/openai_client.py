from openai import OpenAI

client = OpenAI()


def generate(prompt, model="gpt-4o-mini", temperature=0):

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content