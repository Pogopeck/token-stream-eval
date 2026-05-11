from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict, Literal

# The 'Judge' output
class HallucinationCheck(BaseModel):
    is_hallucinated: bool = Field(description="True if the answer contains facts not in the context.")
    reasoning: str = Field(description="Explanation of why it is or isn't hallucinated.")
    confidence: float = Field(description="Confidence score between 0 and 1.")

# The State object that moves through LangGraph
class AgentState(TypedDict):
    question: str
    category: str
    context: str
    answer: str
    is_hallucinated: bool
    retry_count: int
    logs: List[str]
    total_tokens: int
    total_cost: float
    faithfulness_score: float
    cache_hit: bool
    stream_id: str