
from pathlib import Path
from typing import List, TypedDict

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PDF_PATH = Path("./documents/ML Book.pdf")
INDEX_PATH = Path("./faiss_index")
BATCH_SIZE = 8

OLLAMA_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b"


# --------------------------------------------------
# 2. Initialize local models
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_URL,
)

llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0,
    base_url=OLLAMA_URL,
)


# --------------------------------------------------
# 3. Load existing FAISS index or build it once
# --------------------------------------------------

def build_or_load_vector_store() -> FAISS:
    faiss_file = INDEX_PATH / "index.faiss"
    metadata_file = INDEX_PATH / "index.pkl"

    # Reuse the index if both required files exist.
    if faiss_file.exists() and metadata_file.exists():
        print("Loading existing FAISS index...")

        vector_store = FAISS.load_local(
            str(INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        print("FAISS index loaded successfully.")
        return vector_store

    # Build a new index if one doesn't exist.
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH.resolve()}"
        )

    print(f"Loading PDF: {PDF_PATH}")

    docs = PyPDFLoader(str(PDF_PATH)).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_documents(docs)

    # Replace invalid Unicode characters, if any.
    for doc in chunks:
        doc.page_content = (
            doc.page_content
            .encode("utf-8", errors="replace")
            .decode("utf-8")
        )

    # Reject empty or unreadable documents.
    if not chunks or not any(
        doc.page_content.strip() for doc in chunks
    ):
        raise ValueError(
            "No readable text found in the PDF."
        )

    print(f"Total chunks: {len(chunks)}")
    print("Building FAISS index using local embeddings...")

    # Create the initial FAISS index from the first batch.
    first_batch = chunks[:BATCH_SIZE]

    vector_store = FAISS.from_documents(
        documents=first_batch,
        embedding=embeddings,
    )

    # Process and merge remaining batches.
    for start in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        end = start + len(batch)

        print(
            f"Embedding chunks {start + 1}-{end} "
            f"of {len(chunks)}"
        )

        batch_store = FAISS.from_documents(
            documents=batch,
            embedding=embeddings,
        )

        vector_store.merge_from(batch_store)

    # Save the completed index.
    INDEX_PATH.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(INDEX_PATH))

    print(f"Successfully indexed {len(chunks)} chunks.")
    print(f"FAISS index saved to: {INDEX_PATH.resolve()}")

    return vector_store


vector_store = build_or_load_vector_store()


# --------------------------------------------------
# 4. Configure the retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)


# --------------------------------------------------
# 5. Define LangGraph state
# --------------------------------------------------

class State(TypedDict):
    question: str
    docs: List[Document]
    answer: str


# --------------------------------------------------
# 6. Retrieve relevant documents
# --------------------------------------------------

def retrieve(state: State) -> dict:
    question = state["question"]

    retrieved_docs = retriever.invoke(question)

    return {"docs": retrieved_docs}


# --------------------------------------------------
# 7. Generate an answer using the local LLM
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the provided context. "
            "If the context does not contain the answer, say "
            "'I don't know based on the provided document.' "
            "Do not invent facts."
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Context:\n{context}",
        ),
    ]
)


def generate(state: State) -> dict:
    context = "\n\n".join(
        doc.page_content for doc in state["docs"]
    )

    response = (
        prompt | llm
    ).invoke(
        {
            "question": state["question"],
            "context": context,
        }
    )

    return {"answer": response.content}


# --------------------------------------------------
# 8. Build the LangGraph workflow
# --------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

rag_app = graph_builder.compile()


# --------------------------------------------------
# 9. Ask questions about the PDF
# --------------------------------------------------

def ask_question(question: str) -> str:
    result = rag_app.invoke(
        {
            "question": question,
            "docs": [],
            "answer": "",
        }
    )

    return result["answer"]


# --------------------------------------------------
# 10. Interactive question-answering loop
# --------------------------------------------------

if __name__ == "__main__":
    print("\nLocal Agentic RAG is ready!")
    print("Ask questions about ML Book.pdf.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not question:
                continue

            answer = ask_question(question)

            print(f"\nAnswer: {answer}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as exc:
            print(f"\nError: {exc}\n")
