# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão e busca semântica de documentos PDF via CLI, utilizando LangChain, OpenAI e PostgreSQL com pgVector.

## Como funciona

```
document.pdf → chunks → embeddings → PGVector (PostgreSQL)
                                            ↓
                              pergunta do usuário (CLI)
                                            ↓
                              busca semântica (k=10)
                                            ↓
                              prompt + LLM → resposta
```

1. **Ingestão** (`ingest.py`): lê o PDF, divide em chunks, gera embeddings e armazena no banco vetorial.
2. **Busca** (`search.py`): vetoriza a pergunta, busca os chunks mais relevantes e monta a chain com o LLM.
3. **Chat** (`chat.py`): interface CLI que recebe perguntas do usuário e exibe as respostas.

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Chave de API da OpenAI

## Configuração

**1. Clone o repositório e crie o ambiente virtual:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Configure as variáveis de ambiente:**

```bash
cp .env.example .env
```

Preencha o `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=rag
```

## Execução

**1. Suba o banco de dados:**

```bash
docker compose up -d
```

**2. Execute a ingestão do PDF:**

```bash
python src/ingest.py
```

**3. Inicie o chat:**

```bash
python src/chat.py
```

## Exemplo de uso

```
Chat iniciado. Digite 'sair' para encerrar.

Você: Qual o faturamento da Empresa SuperTechIABrazil?

Assistente: O faturamento foi de 10 milhões de reais.

Você: Qual é a capital da França?

Assistente: Não tenho informações necessárias para responder sua pergunta.

Você: sair
Encerrando chat.
```

## Tecnologias

- [LangChain](https://www.langchain.com/)
- [OpenAI](https://platform.openai.com/)
- [PostgreSQL + pgVector](https://github.com/pgvector/pgvector)
- [Docker](https://www.docker.com/)
