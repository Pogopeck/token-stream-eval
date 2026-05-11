import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# Initialize ChromaDB (local storage)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="semantic_cache")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

async def get_semantic_cache(question: str):
    try:
        query_vector = embeddings.embed_query(question)
        results = collection.query(query_embeddings=[query_vector], n_results=1)
        if results['distances'] and results['distances'][0][0] < 0.1:
            return results['documents'][0][0]
    except Exception as e:
        # If the cache fails, we just return None and let the RAG continue
        # This is 'Graceful Degradation'
        print(f"Cache lookup failed: {e}")
    return None
    # Convert question to vector
    query_vector = embeddings.embed_query(question)
    
    # Search for similar questions
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=1
    )
    
    # If the similarity is very high (distance is low), return the cached answer
    if results['distances'] and results['distances'][0][0] < 0.1:
        return results['documents'][0][0]
    return None

def save_to_cache(question: str, answer: str):
    query_vector = embeddings.embed_query(question)
    collection.add(
        embeddings=[query_vector],
        documents=[answer],
        ids=[question]
    )