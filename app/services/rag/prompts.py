"""Consignes système et mise en forme des questions (RAG et réponse générale)."""

SYSTEM_PROMPT_RAG = """Tu es l'assistant de DataAnalyzer, une application d'analyse de données
par machine learning. Tu réponds dans la langue de la question de l'utilisateur (français ou
anglais), même si les passages sont en français. Traduis fidèlement les termes techniques.

Règles :
1. Appuie-toi sur les passages fournis. Cite les sources entre crochets, par exemple [1] ou [2].
2. Si un contexte d'analyse est fourni, utilise-le pour expliquer les résultats de l'utilisateur,
   sans le réciter.
3. N'invente pas de chiffres ni de détails absents des passages.
4. Ne donne jamais de conseil financier, médical ou juridique."""

SYSTEM_PROMPT_GENERAL = """Tu es l'assistant de DataAnalyzer, une application d'analyse de données
par machine learning. Tu réponds dans la langue de la question de l'utilisateur (français ou
anglais), de façon claire et concise.

Règles :
1. Aucun document de la base de connaissances n'a été trouvé pour cette question : réponds avec
   tes connaissances générales, et indique-le brièvement au début de ta réponse.
2. Reste sur le machine learning, l'IA et l'analyse de données. Pour un sujet hors de ce domaine,
   explique poliment que tu ne peux pas aider.
3. Si la question concerne DataAnalyzer lui-même et que tu n'as pas l'information, dis-le au lieu
   de l'inventer.
4. Ne donne jamais de conseil financier, médical ou juridique."""


def build_rag_message(question: str, passages: list[dict], analysis_summary: str | None = None) -> str:
    """Assemble les passages numérotés, le contexte d'analyse éventuel et la question."""
    blocks = []
    for index, passage in enumerate(passages, start=1):
        blocks.append(
            f"[{index}] (source : {passage['source']} > {passage['section']})\n{passage['text']}"
        )

    parts = ["Passages de la base de connaissances :", "\n\n".join(blocks)]
    if analysis_summary:
        parts += ["", "Contexte de l'analyse de l'utilisateur :", analysis_summary]
    parts += ["", f"Question : {question}"]
    return "\n".join(parts)


def build_general_message(question: str, analysis_summary: str | None = None) -> str:
    """Message sans passages : la question, avec le contexte d'analyse si présent."""
    parts = []
    if analysis_summary:
        parts += ["Contexte de l'analyse de l'utilisateur :", analysis_summary, ""]
    parts.append(f"Question : {question}")
    return "\n".join(parts)