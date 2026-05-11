# app/modules/router.py
from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

class RouteQuery(BaseModel):
    category: Literal["greeting", "rag_required", "off_topic"] = Field(
        description="The category of the user query."
    )

async def semantic_router(query: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_retries=6, # Automatically retry 6 times
    timeout=60)
    structured_llm = llm.with_structured_output(RouteQuery)
    
    prompt = f"Categorize this user query for a RAG system: {query}"
    # Use ainvoke for production async support
    result = await structured_llm.ainvoke(prompt)
    return result.category