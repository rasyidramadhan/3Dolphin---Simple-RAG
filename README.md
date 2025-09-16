# Simple RAG Rasyid x 3Dolphin Project Python Development
Implement RAG system, connect it to LLM (OpenAI GPT/HuggingFace) and Qdrant as the vector database.

## Initiation and run project
Download file code (example: main_with_cpu) and follow these commands:
1. Environment initiation
2. Check your environment (example: echo $env:OPENAI_API_KEY)
3. Run Qdrant Docker locally on Powershell/CMD and check
4. Run code: python main_with_cpu.py
5. Check output JSON Qdrant, like this: ({"title":"qdrant - vector search engine","version":"1.15.4","commit":"xxxxxxxxxxxxxx"}
)
6. Test on another terminal: curl http://localhost:xxxx/health
7. Try ingest data and query.
