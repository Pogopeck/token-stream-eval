from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

INPUT_PRICE_PER_1M = 0.30 
OUTPUT_PRICE_PER_1M = 1.25

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), max_retries=6, # Automatically retry 6 times
    timeout=60, streaming=True)

async def generate_rag_answer(question: str, context: str) -> str:
    system_prompt = (
        "You are a helpful assistant. Answer the question based ONLY on the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    user_prompt = f"Context: {context}\n\nQuestion: {question}"
    
    response = await llm.ainvoke([
        ("system", system_prompt),
        ("human", f"Context: {context}\n\nQuestion: {question}")
    ])
    # LEVEL 2: PROGRAMMATIC TRACKING
    usage = response.usage_metadata
    in_t = usage.get("input_tokens", 0)
    out_t = usage.get("output_tokens", 0)
    
    cost = ((in_t * INPUT_PRICE_PER_1M) + (out_t * OUTPUT_PRICE_PER_1M)) / 1_000_000
    return response.content, in_t + out_t, cost