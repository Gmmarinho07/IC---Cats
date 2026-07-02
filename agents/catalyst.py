from prompts.catalyst_prompt import build_catalyst_prompt

from llms.openai_client import generate as gpt_generate
from llms.claude_client import generate as claude_generate

from utils.json_utils import clean_json


# =====================================================
# MODELOS DISPONÍVEIS
# =====================================================

GENERATORS = {

    "gpt": gpt_generate,

    "claude": claude_generate

}


# =====================================================
# AGENT 1
# =====================================================

def extract(text, model):

    prompt = build_catalyst_prompt(text)

    if model not in GENERATORS:

        raise ValueError(
            f"Modelo '{model}' não suportado."
        )

    response = GENERATORS[model](prompt)

    return clean_json(response)