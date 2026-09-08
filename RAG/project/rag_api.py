import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ============================= Setup =============================
load_dotenv()

app = FastAPI(title="stranger things RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "stranger_things")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOP_K = int(os.getenv("TOP_K", 3))

model = SentenceTransformer(EMBEDDING_MODEL)

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)


# =========================== Schemas ===========================

class QueryRequest(BaseModel):
    query: str


class Source(BaseModel):
    book_name: str
    page_number: int
    score: float


class QueryResponse(BaseModel):
    query: str
    route: str
    answer: str
    sources: list[Source]


# =========================== Endpoints ===========================

@app.get("/")
def root():
    return {"name": "stranger things RAG API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    # TODO 2: Router Prompt
    ROUTER_SYSTEM_PROMPT = """You are an intent classification assistant for a stranger things knowledge system.
Analyze the user's input and respond with EXACTLY ONE word from the following options:
- 'retrieve': If the user is asking a specific factual question about the stranger things universe, characters, lore, spells, or plot details.
- 'chitchat': If the user is engaging in casual conversation, greetings, general small talk, or expressing personal feelings/opinions related to stranger things (e.g., "Hi", "How are you?", "I love Hermione").
- 'off-topic': If the user is asking about topics completely unrelated to stranger things (e.g., math problems, coding, general world news, sports).

Output ONLY the category word in lowercase with no punctuation or extra text."""

    route = gemini_llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=request.query),
    ]).text.strip().lower()

    if route not in {"retrieve", "chitchat", "off-topic"}:
        route = "off-topic"

    if route == "chitchat":
        # TODO 3: Chitchat Prompt
        CHITCHAT_SYSTEM_PROMPT = """You are a friendly, enthusiastic stranger things AI assistant. 
Respond warmly and conversationally to the user's greeting or general conversation. 
Incorporate subtle, fun stranger things thematic references where appropriate, but keep the response brief, helpful, and welcoming."""

        response = gemini_llm.invoke([
            SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
            HumanMessage(content=request.query),
        ])

        return QueryResponse(
            query=request.query,
            route=route,
            answer=response.text,
            sources=[],
        )

    if route == "off-topic":
        return QueryResponse(
            query=request.query,
            route=route,
            answer="I can only answer questions about the stranger things universe.",
            sources=[],
        )

    query_vector = model.encode(
        [f"query: {request.query}"],
        normalize_embeddings=True,
    )[0].tolist()

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    ).points

    context = "\n\n".join(
        f"Book: {result.payload.get('book_name', 'Unknown')}\n"
        f"Page: {result.payload.get('page_number', 0)}\n"
        f"Content: {result.payload.get('content', '')}"
        for result in results
    )

    # TODO 4: RAG Prompt
    RAG_SYSTEM_PROMPT = """You are an accurate stranger things assistant.
Answer the user's question using ONLY the factual context provided below. 

Guidelines:
- Rely strictly on the clear facts in the context. Do not extrapolate, speculate, or assume external knowledge.
- If the provided context does NOT contain enough information to answer the question, state clearly: "I do not know."
- Keep your answer direct, accurate, and concise."""

    response = gemini_llm.invoke([
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{request.query}"
        ),
    ])

    return QueryResponse(
        query=request.query,
        route=route,
        answer=response.text,
        sources=[
            Source(
                book_name=result.payload.get("book_name", "Unknown"),
                page_number=result.payload.get("page_number", 0),
                score=result.score,
            )
            for result in results
        ],
    )

    
import uvicorn

if __name__ == "__main__":
    uvicorn.run("rag_api:app", host="127.0.0.1", port=8000, reload=True)