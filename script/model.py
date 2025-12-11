from openai import OpenAI
from typing import List, Dict
from langchain.embeddings.base import Embeddings
from config import LLM_API_KEY, LLM_BASE_URL, EMBED_API_KEY, EMBED_BASE_URL, LLM_MODEL, EMBED_MODEL

llm_client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)

embedding_client = OpenAI(
    api_key=EMBED_API_KEY,
    base_url=EMBED_BASE_URL,
)

def get_response(messages: List[Dict[str, str]]):
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=16384,
        timeout=60,
        temperature=0.0,
    )
    return response.choices[0].message.content

def get_embedding(text):
    response = embedding_client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return response.data[0].embedding

class LCEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        chunks_per_doc = [chunk_text(t, CHUNK_SIZE, CHUNK_OVERLAP) for t in texts]
        flat_chunks: List[str] = []
        counts: List[int] = []
        for cs in chunks_per_doc:
            counts.append(len(cs))
            flat_chunks.extend(cs)
        flat_embs = embed_texts_batch(flat_chunks, batch_size=16)
        out: List[List[float]] = []
        i = 0
        for c in counts:
            embs = flat_embs[i:i + c]
            i += c
            out.append(aggregate_embeddings(embs))
        return out

    def embed_query(self, text: str) -> List[float]:
        return get_embedding(text)
