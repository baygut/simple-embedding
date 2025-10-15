from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

app = FastAPI(
    title="KadranAI Service API",
    description="API for generating text embeddings using a multilingual model.",
    version="1.0.0"
)

model = SentenceTransformer("intfloat/multilingual-e5-small")


class SingleTextRequest(BaseModel):
    text: str = Field(..., description="Single text to embed", example="Hello world")


class MultipleTextsRequest(BaseModel):
    texts: list[str] = Field(..., description="List of texts to embed", example=["Hello world", "How are you?"])


class SingleEmbeddingResponse(BaseModel):
    embedding: list[float] = Field(..., description="Embedding vector for the input text")


class MultipleEmbeddingsResponse(BaseModel):
    embeddings: list[list[float]] = Field(..., description="List of embedding vectors for the input texts")


@app.post(
    "/embedding",
    response_model=SingleEmbeddingResponse,
    summary="Generate embedding for a single text",
    description="Takes a single text string and returns its embedding vector"
)
def embed_single(request: SingleTextRequest) -> SingleEmbeddingResponse:
    embedding = model.encode(request.text).tolist()
    return SingleEmbeddingResponse(embedding=embedding)


@app.post(
    "/embeddings",
    response_model=MultipleEmbeddingsResponse,
    summary="Generate embeddings for multiple texts",
    description="Takes a list of text strings and returns their embedding vectors"
)
def embed_multiple(request: MultipleTextsRequest) -> MultipleEmbeddingsResponse:
    embeddings = model.encode(request.texts).tolist()
    return MultipleEmbeddingsResponse(embeddings=embeddings)