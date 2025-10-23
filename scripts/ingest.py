from app.rag import ingest_corpus
from app.settings import settings

if __name__ == "__main__":
    docs, chunks = ingest_corpus(settings.docs_dir, settings.index_dir)
    print(f"Indexed documents: {docs}, chunks: {chunks}")