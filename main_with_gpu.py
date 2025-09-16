import os, time
import openai, uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance



app = FastAPI(title="RAG with FastAPI and Qdrant")
collection_name =  os.getenv("QDRANT_COLLECTION", "rag_collection")
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("HF_LLM_MODEL", "GPT_LLM_MODEL")
default_k = 3

try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None
    OPENAI_API_KEY = None

try:
    HF_AVAIBLE = True
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except Exception:
    HF_AVAIBLE = False


# Check embedding & LLM model
time_init_embed = time.time()
print(f"Embedding model: {embedding_model_name}")
embedder = SentenceTransformer(embedding_model_name)
embedding_dim = embedder.get_sentence_embedding_dimension()
print(f"Embedding model loaded in {time.time() - time_init_embed:.2f} s")

if QDRANT_API_KEY:
    qdrant = QdrantClient(url=qdrant_url, api_key=QDRANT_API_KEY)
else:
    qdrant = QdrantClient(url=qdrant_url)

collection_list = [col.name for col in qdrant.get_collections().collections]
if collection_name not in collection_list:
    print(f"Creating Qdrant collection: {collection_name}")
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE))
    
HF_PIPELINE = None
if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY
    print("LLM Backend: OpenAI")
elif HF_AVAIBLE:
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
        HF_PIPELINE = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=2048,temperature=0)
    except Exception as e:
        HF_PIPELINE = None
        print(f"Failed to intialize HuggingFace pipeline: {e}")
else:
    print("No LLM backend available!")

class DocInsert(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

class QueryInsert(BaseModel):
    question: str
    top_k: Optional[int] = default_k
    use_openai: Optional[bool] = False

def embed_text(text: List[str]) -> List[List[float]]:
    return embedder.encode(text, show_progress_bar=False, convert_to_numpy=True).tolist()

def docs_upsert(docs: List[DocInsert]):
    ids = [doc.id for doc in docs]
    texts = [doc.text for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vectors = embed_text(texts)
    points = []

    for i, v in enumerate(ids):
        points.append({"id": v, "vector": vectors[i], "payload": {"text": texts[i], **(metadatas[i] or {})}})
    qdrant.upsert(collection_name=collection_name, points=points)

def similiar_searching(question: str, top_k: int):
    question_vector = embed_text([question])[0]
    search_result = qdrant.search(collection_name=collection_name, query_vector=question_vector, limit=top_k)
    results = []

    for sr in search_result:
        results.append({"id": sr.id, "score": sr.score, "text": sr.payload.get("text"), "payload": sr.payload})
    return results

def prompt_building(question: str, contexts: List[Dict[str, Any]]) -> str:
    content_texts = []
    for i, ct in enumerate(contexts):
        content_texts.append(f"===== Document {i+1} (id: {ct.get('id')}) ===== \n{ct.get('text')}\n")

    ctext = "\n".join(content_texts)
    prompt = ("You are a helpful AI assistant. Use the following context to answer the question.\n"
    f"Context:\n{ctext}\n\nQuestion:\n{question}\n\nAnswer:\n")
    return prompt

def call_llm(prompt: str, use_openai: bool = False) -> str:
    if use_openai and OPENAI_API_KEY and openai:
        time_init = time.time()
        try:
            response = openai.chat.completions.create(
                model="your model gpt",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.2
            )
            
            time_final = time.time() - time_init
            return (response.choices[0].message['content'].strip(), time_final)
        
        except Exception as e:
            return RuntimeError(f"OpenAI request failed: {e}, time taken: {time.time() - time_init}")
    
    if HF_PIPELINE:
        call_hf_pipeline = HF_PIPELINE(prompt, max_length=2048, do_sample=False)

        if isinstance(call_hf_pipeline, list) and len(call_hf_pipeline) > 0:
            return call_hf_pipeline[0]['generated_text'].strip()

        return str(call_hf_pipeline)
    raise RuntimeError("No LLM backend available!")

# API Endpoints
@app.post("/ingest")
def ingest_docs(doc: DocInsert):
    try:
        docs_upsert([doc])
        return {"Status": "Success", "Message": f"id: {doc.id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_docs(query: QueryInsert):
    try:
        top_k = query.top_k or default_k
        contexts = similiar_searching(query.question, top_k)
        prompt = prompt_building(query.question, contexts)
        answer, time_exc = call_llm(prompt, use_openai=query.use_openai)
        
        return {"Question": query.question, "Answer": answer, "Contexts": contexts, "Time execution": time_exc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
def status_check():
    return {"Collection": collection_name, "Qdrant": qdrant_url, "Embedding Model": embedding_model_name, "LLM Model": LLM_MODEL if (OPENAI_API_KEY or HF_PIPELINE) else "None"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)