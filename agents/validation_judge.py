
"""
agents/validation_judge.py

Agente 3 — Juiz de validação científica.

Recebe:
- Contexto do artigo
- JSON produzido pelos agentes de extração
- Modelo responsável pelo julgamento

Retorna:
- Decisões de validação
- Evidências textuais
- Justificativas
- Resumo das decisões
"""


from prompts.validation_prompt import build_validation_prompt

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
# VALIDAÇÃO DO RESULTADO
# =====================================================

VALID_DECISIONS = {
    "correct",
    "partial",
    "incorrect",
    "not_supported"
}


def validate_response(response):

    if not isinstance(response, dict):
        raise ValueError(
            "A resposta do juiz não é um objeto JSON válido."
        )

    if "validation" not in response:
        raise ValueError(
            "A resposta do juiz não contém a chave 'validation'."
        )

    if "summary" not in response:
        raise ValueError(
            "A resposta do juiz não contém a chave 'summary'."
        )

    validation = response["validation"]
    summary = response["summary"]

    if not isinstance(validation, list):
        raise ValueError(
            "O campo 'validation' deve ser uma lista."
        )

    if not isinstance(summary, dict):
        raise ValueError(
            "O campo 'summary' deve ser um objeto."
        )

    # Verifica os itens de validação

    for index, item in enumerate(validation):

        if not isinstance(item, dict):
            raise ValueError(
                f"Item de validação {index} inválido."
            )

        required_fields = {
            "entity_type",
            "extracted_value",
            "field",
            "decision",
            "evidence",
            "reason"
        }

        missing_fields = required_fields - item.keys()

        if missing_fields:
            raise ValueError(
                f"Item {index} sem campos obrigatórios: "
                f"{sorted(missing_fields)}"
            )

        if item["decision"] not in VALID_DECISIONS:
            raise ValueError(
                f"Decisão inválida no item {index}: "
                f"{item['decision']}"
            )

    # Recalcula o resumo a partir dos itens retornados.
    # Evita confiar cegamente nas contagens do modelo.

    calculated_summary = {
        decision: sum(
            1
            for item in validation
            if item["decision"] == decision
        )
        for decision in VALID_DECISIONS
    }

    response["summary"] = calculated_summary

    return response


# =====================================================
# EXECUÇÃO DO JUIZ
# =====================================================

def validate(context, extraction, model):

    if not isinstance(context, str) or not context.strip():
        raise ValueError(
            "O contexto do artigo está vazio ou inválido."
        )

    if not isinstance(extraction, dict):
        raise ValueError(
            "A extração deve ser um objeto JSON."
        )

    model = model.lower().strip()

    if model not in GENERATORS:
        raise ValueError(
            f"Modelo juiz '{model}' não suportado. "
            f"Modelos disponíveis: {list(GENERATORS)}"
        )

    # ---------------------------------------------
    # CONSTRUÇÃO DO PROMPT
    # ---------------------------------------------

    prompt = build_validation_prompt(
        context,
        extraction
    )

    # ---------------------------------------------
    # CHAMADA AO MODELO
    # ---------------------------------------------

    print(
        f"\n[AGENT 3] Starting validation with "
        f"{model.upper()}..."
    )

    response = GENERATORS[model](prompt)

    if not response:
        raise ValueError(
            "O modelo juiz retornou uma resposta vazia."
        )

    # ---------------------------------------------
    # TRATAMENTO DO JSON
    # ---------------------------------------------

    parsed_response = clean_json(response)

    # ---------------------------------------------
    # VERIFICAÇÃO DA ESTRUTURA
    # ---------------------------------------------

    validated_response = validate_response(
        parsed_response
    )

    print(
        f"[AGENT 3] Validation completed with "
        f"{model.upper()}."
    )

    print(
        "[AGENT 3] Summary:",
        validated_response["summary"]
    )

    return validated_response
