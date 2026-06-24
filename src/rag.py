import os
from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm

import PyPDF2
import psycopg
from openai import OpenAI
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

DB_URL = os.getenv(
    "DB_URL",
    "postgresql+psycopg://ai:ai@localhost:5432/ai",
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")

CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200

# Retrieval tuning
VECTOR_K        = 20
BM25_K          = 20
RERANK_TOP_N    = 8
MULTI_QUERY_N   = 3      # original + N variants
HISTORY_TURNS   = 5      # how many past turns RAGChain keeps
HISTORY_FOR_REWRITE = 3  # how many turns are fed into the rewriter

# OpenAI — chat + embeddings. Base URL is overridable for OpenAI-compatible
# endpoints (Azure, local vLLM, etc.).
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL           = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL     = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Empty key is tolerated at construction; calls fail gracefully (caught below).
_openai = OpenAI(base_url=OPENAI_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", ""))

# Reranker — multilingual cross-encoder. The only local model left; OpenAI has
# no rerank endpoint, so we keep this for precise ranking.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

EMBEDDING_MODEL = OpenAIEmbeddings(
    model=EMBED_MODEL,
    base_url=OPENAI_BASE_URL,
    api_key=os.getenv("OPENAI_API_KEY", ""),
)

# -------------------------------------------------------
# Global state
# -------------------------------------------------------
_rag_chain = None
_llm_invoke = None


# -------------------------------------------------------
# LLM
# -------------------------------------------------------

def _call_llm(prompt: str, max_tokens: int = 512) -> str:
    try:
        resp = _openai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
            top_p=0.9,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[LLM error: {e}]"


# -------------------------------------------------------
# Text normalization
# -------------------------------------------------------

# Apostrophes and quotes come in several Unicode variants:
#   ' ' (U+2018/U+2019, curly), ` (U+0060), ʻ ʼ (U+02BB/U+02BC), ′ ‵ (primes)...
# Users type a plain ' (U+0027). This mismatch weakens BM25 (exact token match)
# and embeddings. We fold every variant to a single ' (U+0027) — at both index
# and query time.
_APOSTROPHE_VARIANTS = "ʻʼ‘’`´′‵"
_APOSTROPHE_TABLE = {ord(c): "'" for c in _APOSTROPHE_VARIANTS}


def _normalize_text(text: str) -> str:
    """Fold apostrophe/quote variants to a single ' (U+0027)."""
    return text.translate(_APOSTROPHE_TABLE) if text else text


# -------------------------------------------------------
# Document Loading
# -------------------------------------------------------

def _load_pdf(path: Path) -> str:
    reader = PyPDF2.PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return _normalize_text(text)


def _load_txt(path: Path) -> str:
    return _normalize_text(path.read_text(encoding="utf-8"))


def _load_documents_from_folder(folder_path: str) -> List[Document]:
    """Read all .txt and .pdf files in a folder and return a list of Documents."""
    data_path = Path(folder_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = list(data_path.glob("*.txt")) + list(data_path.glob("*.pdf"))
    if not files:
        raise ValueError(f"No .txt or .pdf files found in folder: {folder_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    all_docs: List[Document] = []
    for file in tqdm(files, desc="Reading files"):
        try:
            if file.suffix.lower() == ".pdf":
                text = _load_pdf(file)
            else:
                text = _load_txt(file)

            raw_doc = Document(page_content=text, metadata={"source": file.name})
            chunks = splitter.split_documents([raw_doc])

            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "parent_file": file.name,
                    "chunk_index": i,
                })
            all_docs.extend(chunks)
        except Exception as e:
            print(f"Error ({file.name}): {e}")

    print(f"Prepared {len(all_docs)} chunks in total.")
    return all_docs


def _fetch_all_docs_from_pgvector(collection_name: str) -> List[Document]:
    """Read all chunks from an existing pgvector collection (for BM25)."""
    pg_url = DB_URL.replace("postgresql+psycopg://", "postgresql://", 1)
    docs: List[Document] = []
    try:
        with psycopg.connect(pg_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT e.document, e.cmetadata
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                    """,
                    (collection_name,),
                )
                for content, meta in cur.fetchall():
                    docs.append(Document(page_content=_normalize_text(content or ""), metadata=meta or {}))
    except Exception as e:
        print(f"Error fetching chunks for BM25: {e}")
    return docs


# -------------------------------------------------------
# Query transforms
# -------------------------------------------------------

def _query_rewrite_with_history(query: str, history: List[Tuple[str, str]]) -> str:
    """Rewrite the query, resolving pronouns to concrete names from the history."""
    if not history:
        return _query_rewrite_plain(query)

    recent = history[-HISTORY_FOR_REWRITE:]
    hist_block = "\n\n".join(
        f"User: {q}\nAnswer: {a[:200]}" for q, a in recent
    )

    prompt = (
        "Given the conversation history, rewrite the user's last question into a "
        "standalone semantic search query. Replace pronouns (he, she, it, they, "
        "this, that) with the concrete names they refer to in the history. "
        "If the question is already complete and unambiguous, return it unchanged. "
        "Output only the rewritten question, with no comments.\n\n"
        f"Conversation history:\n{hist_block}\n\n"
        f"Last question: {query}\n\nRewritten question:"
    )
    rewritten = _call_llm(prompt, max_tokens=80)
    return rewritten or query


def _query_rewrite_plain(query: str) -> str:
    prompt = (
        "Rewrite the user query so that it becomes a good semantic search query. "
        "Make it a short descriptive sentence. Do not add comments or explanation.\n\n"
        f"query: {query}\n\nrewritten:"
    )
    return _call_llm(prompt, max_tokens=64) or query


def _multi_query(query: str, n: int = MULTI_QUERY_N) -> List[str]:
    """Generate n semantic variants of a single question. The original is included too."""
    prompt = (
        f"Generate {n} different semantic search variants of the question below. "
        "Write each on a new line, without numbering or comments. "
        "The variants must mean the same thing but use different words and "
        "phrasings. Do not repeat the original question.\n\n"
        f"Original question: {query}\n\n"
        "Variants:"
    )
    resp = _call_llm(prompt, max_tokens=200)
    variants: List[str] = []
    for line in resp.split("\n"):
        cleaned = line.strip().lstrip("-•*").strip()
        # numeric prefix (1. , 2) , 3 - )
        while cleaned and cleaned[0].isdigit():
            cleaned = cleaned[1:]
        cleaned = cleaned.lstrip(".)- ").strip()
        if cleaned and cleaned.lower() != query.lower():
            variants.append(cleaned)
    variants = variants[:n]
    # prepend the original question so it is always present
    return [query] + variants


# -------------------------------------------------------
# Reranker (cross-encoder)
# -------------------------------------------------------

class CrossEncoderReranker:
    """Multilingual cross-encoder reranker (BAAI/bge-reranker-v2-m3).

    The default FlashrankRerank model (ms-marco-MiniLM) is English-only and
    ranked multilingual candidates poorly. This wrapper uses a strong
    multilingual cross-encoder for precise reranking.
    """

    def __init__(self, model_name: str = RERANKER_MODEL, top_n: int = RERANK_TOP_N):
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device, max_length=1024)
        self.top_n = top_n

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        if not documents:
            return []
        scores = self.model.predict([(query, d.page_content) for d in documents])
        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]


# -------------------------------------------------------
# RAGChain
# -------------------------------------------------------

class RAGChain:
    def __init__(self, hybrid_retriever, reranker, prompt):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.prompt = prompt
        self.history: List[Tuple[str, str]] = []

    def reset_history(self) -> None:
        self.history = []

    def invoke(self, query: str) -> str:
        query = _normalize_text(query)
        resolved = _normalize_text(_query_rewrite_with_history(query, self.history))
        variants = [_normalize_text(v) for v in _multi_query(resolved)]

        seen = set()
        candidates: List[Document] = []
        for q in variants:
            try:
                fetched = self.hybrid_retriever.invoke(q)
            except Exception as e:
                print(f"Retrieval error ('{q[:40]}...'): {e}")
                continue
            for d in fetched:
                key = (
                    d.metadata.get("parent_file", ""),
                    d.metadata.get("chunk_index", -1),
                    d.page_content[:80],
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(d)

        if not candidates:
            answer = "No relevant information found."
        else:
            try:
                top_docs = self.reranker.compress_documents(candidates, resolved)
            except Exception as e:
                print(f"Rerank error: {e}")
                top_docs = candidates[:RERANK_TOP_N]

            context = "\n\n".join(d.page_content for d in top_docs)
            final_prompt = self.prompt.format(context=context, query=query)
            answer = _call_llm(final_prompt)

        self.history.append((query, answer))
        if len(self.history) > HISTORY_TURNS:
            self.history = self.history[-HISTORY_TURNS:]
        return answer


# -------------------------------------------------------
# Public API
# -------------------------------------------------------

def initialize_rag(
    input_path: str,
    collection_name: str = COLLECTION_NAME,
    pre_delete_collection: bool = False,
):
    """
    Initialize the RAG system.

    Args:
        input_path: Path to a file (PDF/TXT) or a folder.
        collection_name: PGVector collection name.
        pre_delete_collection: If True, wipe previous data and re-ingest.
    """
    global _rag_chain

    # 1. Documents
    path = Path(input_path)
    if path.is_dir():
        docs = _load_documents_from_folder(input_path)
    elif path.suffix.lower() == ".pdf":
        text = _load_pdf(path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        docs = splitter.create_documents([text], metadatas=[{"source": path.name}])
    elif path.suffix.lower() == ".txt":
        text = _load_txt(path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        docs = splitter.create_documents([text], metadatas=[{"source": path.name}])
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    # 2. PGVector
    print("Writing vectors to PGVector...")
    vector_store = PGVector.from_documents(
        documents=docs,
        embedding=EMBEDDING_MODEL,
        connection=DB_URL,
        collection_name=collection_name,
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )

    print(f"Database ready: '{collection_name}'")
    _build_chain(vector_store, docs)
    return _rag_chain


def connect_existing(collection_name: str = COLLECTION_NAME):
    """Connect to an existing PGVector collection (without re-ingesting)."""
    global _rag_chain

    vector_store = PGVector(
        connection=DB_URL,
        embeddings=EMBEDDING_MODEL,
        collection_name=collection_name,
        use_jsonb=True,
    )

    print(f"Database connected: '{collection_name}'")
    docs = _fetch_all_docs_from_pgvector(collection_name)
    print(f"Loaded {len(docs)} chunks for BM25.")
    _build_chain(vector_store, docs)
    return _rag_chain


def _build_chain(vector_store, all_docs: List[Document]):
    global _rag_chain

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": VECTOR_K})

    if all_docs:
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = BM25_K
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6],
        )
    else:
        # No BM25 corpus available — vector retriever only
        print("Warning: no chunks for BM25, using vector search only.")
        base_retriever = vector_retriever

    reranker = CrossEncoderReranker(RERANKER_MODEL, top_n=RERANK_TOP_N)

    prompt_template = """Answer the user's question using only the context below.
If the answer is not in the context, say 'No relevant information found.'

Context:
{context}

Question: {query}

Answer:"""
    custom_prompt = PromptTemplate.from_template(prompt_template)

    _rag_chain = RAGChain(base_retriever, reranker, custom_prompt)


def return_response(text: str) -> str:
    global _rag_chain
    if _rag_chain is None:
        return "RAG system not initialized. Call initialize_rag() or connect_existing() first."
    return _rag_chain.invoke(text)


def reset_history() -> None:
    """Clear the conversation history (useful when switching topics)."""
    global _rag_chain
    if _rag_chain is not None:
        _rag_chain.reset_history()


# -------------------------------------------------------
# CLI / standalone
# -------------------------------------------------------

if __name__ == "__main__":
    print("=== RAG system ===")
    print("1 - Ingest new data (folder or file)")
    print("2 - Connect to an existing database")
    choice = input("Choice: ").strip()

    if choice == "1":
        input_path = input("Enter the file or folder path: ").strip()
        col = input(f"Collection name [{COLLECTION_NAME}]: ").strip() or COLLECTION_NAME
        overwrite = input("Wipe existing data and re-ingest? (y/n): ").strip().lower() == "y"
        initialize_rag(input_path, collection_name=col, pre_delete_collection=overwrite)
    elif choice == "2":
        col = input(f"Collection name [{COLLECTION_NAME}]: ").strip() or COLLECTION_NAME
        connect_existing(col)
    else:
        print("Invalid choice.")
        exit(1)

    print("\nType a question (quit: 'exit', clear history: 'reset'):")
    while True:
        query = input("\nQuestion: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "reset":
            reset_history()
            print("Conversation history cleared.")
            continue
        print(f"\nAnswer: {return_response(query)}")
