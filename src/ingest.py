import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = Path(__file__).parent.parent / "document.pdf"

def ingest_pdf():
    for k in ("OPENAI_API_KEY", "OPENAI_EMBEDDING_MODEL", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
        if not os.getenv(k):
            raise RuntimeError(f"Environment variable {k} is not set.")

    docs = PyPDFLoader(str(PDF_PATH)).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    ).split_documents(docs)

    if not splits:
        raise SystemExit(0)

    enriched_splits = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]

    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True
    )

    ids = [f"doc-{i}" for i in range(len(enriched_splits))]
    store.add_documents(documents=enriched_splits, ids=ids)


if __name__ == "__main__":
    ingest_pdf()