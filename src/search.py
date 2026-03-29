import os
from dotenv import load_dotenv

from langchain_postgres import PGVector
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(question=None):
    for k in ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
        if not os.getenv(k):
            raise RuntimeError(f"Environment variable {k} is not set.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=PROMPT_TEMPLATE,
    )

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True
    )

    chain = prompt | llm

    # print("[1/4] Vetorizando a pergunta...")
    results = store.similarity_search_with_score(question, k=10)
    # print(f"[2/4] {len(results)} chunks recuperados do banco vetorial.")

    context = "\n\n".join([doc.page_content for doc, _ in results])

    # print("[3/4] Montando prompt e chamando a LLM...")
    result = chain.invoke({"contexto": context, "pergunta": question})
    # print("[4/4] Resposta recebida.")

    return result
  