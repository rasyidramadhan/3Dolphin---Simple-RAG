import os, time
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from openai import OpenAI



app = Flask(__name__)
collection_name = os.getenv("QDRANT_COLLECTION", "rag_collection")
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
llm_model = os.getenv("OPENAI_LLM_MODEL")

embedder = SentenceTransformer(embedding_model_name)
embedding_dim = embedder.get_sentence_embedding_dimension()
qdrant = QdrantClient(url=qdrant_url)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

collections = [col.name for col in qdrant.get_collections().collections]
if collection_name not in collections:
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )

def embed_text(text):
    return embedder.encode([text])[0].tolist()

@app.route("/", methods=['GET'])
def home():
    return jsonify({"message": "RAG API is running", "endpoints": ["/ingest", "/query", "/health"]})

@app.route('/ingest', methods=['POST'])
def ingest():
    try:
        data = request.get_json()
        doc_id = data.get('id')
        text = data.get('text')
        meta = data.get('meta', {})

        if not doc_id or not text:
            return jsonify({"error": "Both 'id' and 'text' are required."}), 400

        vector = embed_text(text)
        qdrant.upsert(
            collection_name=collection_name,
            points=[{
                "id": doc_id,
                "vector": vector,
                "payload": {"text": text, **meta}
            }]
        )
        return jsonify({"status": "success", "message": f"Document {doc_id} ingested."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        question = data.get('question')
        top_k = data.get('top_k', 3)

        if not question:
            return jsonify({"error": "The 'question' field is required."}), 400

        q_vector = embed_text(question)
        search_result = qdrant.search(
            collection_name=collection_name,
            query_vector=q_vector,
            limit=top_k
        )

        context = [{"id": sr.id, "score": sr.score, "text": sr.payload["text"]}
                   for sr in search_result]
        context_text = "\n".join(c["text"] for c in context)

        prompt = (
            "You are a helpful AI assistant. "
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question:\n{question}\n\nAnswer:"
        )

        time_init_llm = time.time()
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )
        answer = response.choices[0].message.content
        print(f"LLM response time: {time.time() - time_init_llm:.2f} s")

        return jsonify({"Question": question, "Answer": answer, "Context": context})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def status_check():
    return jsonify({
        "collection": collection_name,
        "qdrant": qdrant_url,
        "embedding_model": embedding_model_name,
        "llm_model": llm_model
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)